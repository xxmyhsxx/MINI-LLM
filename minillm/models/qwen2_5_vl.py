"""Qwen2.5-VL multimodal model implementation."""

import itertools

import torch
import torch.nn as nn
from typing import Optional, Tuple

from minillm.models.qwen2 import Qwen2ForCausalLM
from minillm.vision.vision_encoder import Qwen2_5_VisionTransformer
from minillm.utils.context import set_context, reset_context


class Qwen2_5_VLForConditionalGeneration(nn.Module):
    """Qwen2.5-VL model with vision encoder and language model."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Vision encoder
        self.visual = Qwen2_5_VisionTransformer(config.vision_config)

        # Language model (reuse Qwen2 implementation)
        self.language_model = Qwen2ForCausalLM(config.text_config)

        # 缓存融合视觉特征的输入 embedding，供解码阶段使用
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
        """计算 Qwen2.5-VL 的 3D mRoPE position_ids 和 rope_deltas。"""
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

    def compute_3d_position_ids(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: torch.LongTensor | None,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """计算当前 step 的 3D 位置编码。"""
        if image_grid_thw is not None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )
            self.rope_deltas = rope_deltas
            return position_ids[:, 0]

        batch_size, seq_len, _ = inputs_embeds.shape
        position_ids = torch.arange(seq_len, device=inputs_embeds.device)
        position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
        return position_ids[:, 0]


    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
    ):
        """Forward pass with vision and text inputs.

        Args:
            input_ids: Text token IDs with vision placeholders
            positions: Position indices for RoPE
            pixel_values: Image patches (seq_len, channels)
            image_grid_thw: Grid dimensions (num_images, 3) for [t, h, w]
            use_cache: If True, reuse cached embeddings for tokens already processed
        """
        batch_size, seq_len = input_ids.shape

        if use_cache and self._cached_input_embeds is not None and pixel_values is None and image_grid_thw is None:
            new_embeds = self.language_model.model.embed_tokens(input_ids)
            hidden_states = new_embeds
            position_ids = self._get_decode_position_ids(input_ids, hidden_states)
            self._cached_seq_len += input_ids.shape[1]
        else:
            hidden_states = self._merge_inputs(input_ids, pixel_values, image_grid_thw)
            if use_cache:
                self._cached_input_embeds = hidden_states
                self._cached_seq_len = seq_len
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                inputs_embeds=hidden_states,
            )

        hidden_states = hidden_states.squeeze(0)
        hidden_states = self.language_model.model.forward_embeds(hidden_states, position_ids)

        # Take only the last token and compute logits directly
        # hidden_states shape: (seq_len, hidden_size)
        last_hidden = hidden_states[-1:, :]  # (1, hidden_size)

        # Compute logits directly without going through LMHead's prefill logic
        import torch.nn.functional as F
        logits = F.linear(last_hidden, self.language_model.lm_head.weight)  # (1, vocab_size)
        logits = logits.unsqueeze(0)  # (1, 1, vocab_size) for compatibility

        return logits

    def _reset_cache(self):
        """清除 embedding 缓存"""
        self._cached_input_embeds = None
        self._cached_seq_len = 0
        self.rope_deltas = None

    def _get_decode_position_ids(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """计算 decode 阶段新增 token 的位置编码。"""
        batch_size = input_ids.shape[0]
        base_position = self._cached_seq_len
        if self.rope_deltas is not None:
            base_position += int(self.rope_deltas[0, 0].item())
        last_position = torch.full(
            (3, batch_size, 1),
            base_position,
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        return last_position[:, 0]

    def _merge_inputs(
        self,
        input_ids: torch.LongTensor,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """构建融合视觉特征后的输入 embedding。"""
        batch_size, seq_len = input_ids.shape
        if pixel_values is not None and image_grid_thw is not None:
            vision_outputs = self.visual(pixel_values, image_grid_thw)
            vision_embeds = vision_outputs
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
        """Generate text with vision context."""
        batch_size, seq_len = input_ids.shape

        # 开始生成前重置缓存
        self._reset_cache()

        for step in range(max_new_tokens):
            if step == 0:
                current_input_ids = input_ids
                positions = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
                cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=input_ids.device)
                slot_mapping = torch.arange(seq_len, dtype=torch.int32, device=input_ids.device)
                set_context(
                    is_prefill=True,
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=seq_len,
                    max_seqlen_k=seq_len,
                    slot_mapping=slot_mapping,
                )
                current_pixel_values = pixel_values
                current_image_grid_thw = image_grid_thw
            else:
                current_input_ids = input_ids[:, -1:]
                positions = torch.tensor([seq_len - 1], device=input_ids.device, dtype=torch.long)
                slot = seq_len - 1
                slot_mapping = torch.tensor([slot], dtype=torch.int32, device=input_ids.device)
                context_lens = torch.tensor([seq_len], dtype=torch.int32, device=input_ids.device)
                num_blocks = (seq_len + 255) // 256
                block_tables = torch.arange(num_blocks, dtype=torch.int32, device=input_ids.device).view(1, -1)
                set_context(
                    is_prefill=False,
                    slot_mapping=slot_mapping,
                    context_lens=context_lens,
                    block_tables=block_tables,
                )
                current_pixel_values = None
                current_image_grid_thw = None

            try:
                logits = self.forward(
                    input_ids=current_input_ids,
                    positions=positions,
                    pixel_values=current_pixel_values,
                    image_grid_thw=current_image_grid_thw,
                    use_cache=True,
                )
            finally:
                reset_context()

            next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)

            # Sample next token
            if temperature > 0:
                next_token_logits = next_token_logits / temperature

                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')

                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            seq_len += 1

            # Check for EOS
            if not ignore_eos:
                # Resolve eos_token_id if not provided
                if eos_token_id is None:
                    eos_token_id = getattr(self.config.text_config, 'eos_token_id', None)
                
                # Make it a list for easier checking
                if eos_token_id is not None:
                    eos_ids = [eos_token_id] if isinstance(eos_token_id, int) else eos_token_id
                    if next_token.item() in eos_ids:
                        break

        return input_ids
