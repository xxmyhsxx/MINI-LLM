"""Qwen2.5-VL 多模态模型实现。"""

import itertools
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from minillm.models.qwen2 import Qwen2ForCausalLM
from minillm.vision.vision_encoder import Qwen2_5_VisionTransformer


class Qwen2_5_VLForConditionalGeneration(nn.Module):
    """Qwen2.5-VL 模型，包含视觉编码器和语言模型。"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.visual = Qwen2_5_VisionTransformer(config.vision_config)
        self.language_model = Qwen2ForCausalLM(config.text_config)
        self._cached_input_embeds = None
        self._cached_seq_len = 0
        self.rope_deltas = None

    def get_vision_position_ids(
        self,
        start_position: int,
        grid_thw: torch.Tensor,
        spatial_merge_size: int,
        device: torch.device,
        time_interval: int = 1,
    ) -> torch.Tensor:
        """计算单张图像的 3D 位置编码。"""
        llm_grid_t = int(grid_thw[0].item())
        llm_grid_h = int(grid_thw[1].item()) // spatial_merge_size
        llm_grid_w = int(grid_thw[2].item()) // spatial_merge_size
        image_seq_length = llm_grid_t * llm_grid_h * llm_grid_w
        position_width = torch.arange(start_position, start_position + llm_grid_w, device=device).repeat(
            llm_grid_h * llm_grid_t
        )
        position_height = torch.arange(start_position, start_position + llm_grid_h, device=device).repeat_interleave(
            llm_grid_w * llm_grid_t
        )
        position_temporal = torch.full((image_seq_length,), start_position, device=device, dtype=torch.long)
        position_temporal = position_temporal * time_interval
        return torch.stack([position_temporal, position_height, position_width], dim=0)

    def _get_rope_index_single_image(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """单 batch 单图像场景的快速位置编码路径。"""
        if input_ids.shape[0] != 1 or image_grid_thw.shape[0] != 1:
            return None
        image_token_id = getattr(self.config, "image_token_id", 151655)
        current_input_ids = input_ids[0]
        image_mask = current_input_ids == image_token_id
        if not image_mask.any():
            return None
        image_positions = torch.nonzero(image_mask, as_tuple=False).flatten()
        if image_positions.numel() == 0:
            return None
        start_idx = int(image_positions[0].item())
        end_idx = int(image_positions[-1].item()) + 1
        if end_idx - start_idx != image_positions.numel():
            return None

        spatial_merge_size = self.config.vision_config.spatial_merge_size
        tokens_per_second = self.config.vision_config.tokens_per_second
        grid_thw = image_grid_thw[0]

        prefix_len = start_idx
        suffix_len = current_input_ids.numel() - end_idx
        prefix_pos = torch.arange(prefix_len, device=input_ids.device).view(1, -1).expand(3, -1)
        vision_pos = self.get_vision_position_ids(
            prefix_len,
            grid_thw,
            spatial_merge_size,
            input_ids.device,
            time_interval=tokens_per_second,
        )
        current_pos = prefix_len + max(int(grid_thw[1].item()), int(grid_thw[2].item())) // spatial_merge_size
        suffix_pos = torch.arange(suffix_len, device=input_ids.device).view(1, -1).expand(3, -1) + current_pos
        llm_positions = torch.cat([prefix_pos, vision_pos, suffix_pos], dim=1)
        position_ids = llm_positions.unsqueeze(1)
        rope_deltas = torch.tensor([[llm_positions.max() + 1 - current_input_ids.numel()]], device=input_ids.device)
        return position_ids, rope_deltas

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """计算 Qwen2.5-VL 的 3D mRoPE 位置编码。"""
        fast_path = self._get_rope_index_single_image(input_ids, image_grid_thw)
        if fast_path is not None and attention_mask is None:
            return fast_path

        spatial_merge_size = self.config.vision_config.spatial_merge_size
        tokens_per_second = self.config.vision_config.tokens_per_second
        position_ids = torch.zeros(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        mrope_position_deltas = []
        image_iter = iter(image_grid_thw)
        image_token_id = getattr(self.config, "image_token_id", 151655)

        for batch_idx, current_input_ids in enumerate(input_ids):
            current_mask = attention_mask[batch_idx].bool() if attention_mask is not None else None
            if current_mask is not None:
                current_input_ids = current_input_ids[current_mask]

            modality_types = torch.where(current_input_ids == image_token_id, 1, 0)
            input_type_group = []
            for key, group in itertools.groupby(enumerate(modality_types.tolist()), lambda x: x[1]):
                group = list(group)
                start_index = group[0][0]
                end_index = group[-1][0] + 1
                input_type_group.append((key, start_index, end_index))

            current_pos = 0
            llm_pos_ids_list = []
            for modality_type, start_idx, end_idx in input_type_group:
                if modality_type == 0:
                    text_len = end_idx - start_idx
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + current_pos
                    )
                    current_pos += text_len
                else:
                    grid_thw = next(image_iter)
                    vision_position_ids = self.get_vision_position_ids(
                        current_pos,
                        grid_thw,
                        spatial_merge_size,
                        input_ids.device,
                        time_interval=tokens_per_second,
                    )
                    llm_pos_ids_list.append(vision_position_ids)
                    current_pos += max(int(grid_thw[1].item()), int(grid_thw[2].item())) // spatial_merge_size

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            if current_mask is not None:
                position_ids[:, batch_idx, current_mask] = llm_positions.to(position_ids.device)
            else:
                position_ids[:, batch_idx] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(current_input_ids))

        rope_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, rope_deltas

    def get_position_offset(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: torch.LongTensor | None,
    ) -> int:
        """返回多模态 decode 阶段的额外位置偏移量。"""
        if image_grid_thw is None:
            return 0
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        _, rope_deltas = self.get_rope_index(input_ids=input_ids, image_grid_thw=image_grid_thw)
        return int(rope_deltas[0, 0].item())

    def compute_3d_position_ids(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: torch.LongTensor | None,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """计算当前 step 的位置编码。"""
        if image_grid_thw is not None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )
            self.rope_deltas = rope_deltas
            return position_ids[:, 0]

        if inputs_embeds.ndim == 3:
            _, seq_len, _ = inputs_embeds.shape
        else:
            seq_len = inputs_embeds.shape[0]
        return torch.arange(seq_len, device=inputs_embeds.device, dtype=torch.long)

    def forward_hidden(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor | None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """返回可直接送入语言头的隐藏状态。"""
        if pixel_values is not None and image_grid_thw is not None:
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
            hidden_states = self._merge_inputs(input_ids, pixel_values, image_grid_thw)
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                inputs_embeds=hidden_states,
            )
            hidden_states = hidden_states.squeeze(0)
            return self.language_model.model.forward_embeds(hidden_states, position_ids)

        if input_ids.ndim == 2:
            assert input_ids.shape[0] == 1, "当前仅支持单请求多模态 prefill"
            input_ids = input_ids.squeeze(0)

        hidden_states = self.language_model.model.embed_tokens(input_ids)
        if positions is None:
            positions = torch.arange(input_ids.shape[0], device=input_ids.device, dtype=torch.long)
        return self.language_model.model.forward_embeds(hidden_states, positions)

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor | None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
    ):
        """前向传播，兼容旧接口并返回 logits。"""
        del use_cache
        hidden_states = self.forward_hidden(input_ids, positions, pixel_values, image_grid_thw)
        logits = self.compute_logits(hidden_states)
        if logits.ndim == 2 and logits.shape[0] == 1:
            return logits.unsqueeze(0)
        if pixel_values is not None or image_grid_thw is not None:
            return logits[-1:, :].unsqueeze(0)
        return logits

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """将隐藏状态映射到词表 logits。"""
        if hidden_states.dtype != self.language_model.lm_head.weight.dtype:
            hidden_states = hidden_states.to(self.language_model.lm_head.weight.dtype)
        return self.language_model.compute_logits(hidden_states)

    def _reset_cache(self):
        """重置兼容字段。"""
        self._cached_input_embeds = None
        self._cached_seq_len = 0
        self.rope_deltas = None

    def _merge_inputs(
        self,
        input_ids: torch.LongTensor,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """构建融合视觉特征后的输入 embedding。"""
        if pixel_values is not None and image_grid_thw is not None:
            vision_embeds = self.visual(pixel_values, image_grid_thw)
            hidden_states = self.language_model.model.embed_tokens(input_ids)
            image_token_id = getattr(self.config, "image_token_id", 151655)
            vision_mask = input_ids == image_token_id
            if vision_mask.any():
                assert vision_mask.sum().item() == vision_embeds.shape[0], (
                    f"图像占位 token 数量与视觉特征数量不匹配: "
                    f"placeholders={vision_mask.sum().item()}, vision_embeds={vision_embeds.shape[0]}"
                )
                hidden_states = hidden_states.clone()
                hidden_states[vision_mask] = vision_embeds
            return hidden_states
        return self.language_model.model.embed_tokens(input_ids)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.LongTensor,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        eos_token_id: int | list[int] | None = None,
        ignore_eos: bool = False,
    ):
        """兼容旧接口的简单生成实现。"""
        seq_len = input_ids.shape[-1]
        position_offset = self.get_position_offset(input_ids, image_grid_thw)
        self._reset_cache()

        for step in range(max_new_tokens):
            if step == 0:
                current_input_ids = input_ids
                current_positions = None
                current_pixel_values = pixel_values
                current_image_grid_thw = image_grid_thw
            else:
                current_input_ids = input_ids[:, -1:]
                current_positions = torch.tensor(
                    [seq_len - 1 + position_offset],
                    device=input_ids.device,
                    dtype=torch.long,
                )
                current_pixel_values = None
                current_image_grid_thw = None

            logits = self.forward(
                input_ids=current_input_ids,
                positions=current_positions,
                pixel_values=current_pixel_values,
                image_grid_thw=current_image_grid_thw,
                use_cache=True,
            )
            next_token_logits = logits[:, -1, :]

            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float("-inf")
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float("-inf")
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            seq_len += 1

            if not ignore_eos:
                if eos_token_id is None:
                    eos_token_id = getattr(self.config.text_config, "eos_token_id", None)
                if eos_token_id is not None:
                    eos_ids = [eos_token_id] if isinstance(eos_token_id, int) else eos_token_id
                    if next_token.item() in eos_ids:
                        break

        return input_ids
