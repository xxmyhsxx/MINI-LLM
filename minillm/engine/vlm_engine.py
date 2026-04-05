"""VLM Engine — 视觉语言模型推理引擎。"""

import json
from dataclasses import fields
from pathlib import Path
from time import perf_counter

import torch
from PIL import Image
from transformers import AutoConfig, AutoTokenizer

from minillm.config import Config
from minillm.engine.block_manager import BlockManager
from minillm.engine.sequence import Sequence
from minillm.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from minillm.sampling_params import SamplingParams
from minillm.utils.context import reset_context, set_context
from minillm.utils.vlm_loader import load_vlm_model
from minillm.vision.image_processor import Qwen2VLImageProcessor


class VLMEngine:
    """视觉语言模型推理引擎。

    与 LLMEngine 不同，VLMEngine 专门处理多模态输入（图像+文本）。
    当前实现为简化版本，使用自己的模型实现。

    Attributes:
        model: VLM 模型
        tokenizer: 分词器
        image_processor: 图像处理器
    """

    def __init__(self, model_path: str, device: str = "cuda", **kwargs):
        """初始化 VLM 引擎。

        Args:
            model_path: 模型路径
            device: 设备（cuda 或 cpu）
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=True, trust_remote_code=True
        )

        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        self.runtime_config = Config(model_path, **config_kwargs)

        # 加载配置
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        preprocessor_path = Path(model_path) / "preprocessor_config.json"
        min_pixels, max_pixels = 56 * 56, 14 * 14 * 4 * 1280
        if preprocessor_path.exists():
            with open(preprocessor_path, "r", encoding="utf-8") as f:
                p_config = json.load(f)
                min_pixels = p_config.get("min_pixels", min_pixels)
                max_pixels = p_config.get("max_pixels", max_pixels)

        self.image_processor = Qwen2VLImageProcessor(
            patch_size=config.vision_config.patch_size,
            temporal_patch_size=config.vision_config.temporal_patch_size,
            merge_size=config.vision_config.spatial_merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        # 使用自己的模型实现
        self.model = Qwen2_5_VLForConditionalGeneration(config)

        # 加载权重（使用 VLM 专用加载器）
        load_vlm_model(self.model, model_path)

        # 转换为 bfloat16（Flash Attention 要求）
        self.model.to(device).to(torch.bfloat16)
        self.model.eval()

        # 分配 KV cache
        self.block_size = self.runtime_config.kvcache_block_size
        if self.runtime_config.cache_size_tokens is not None:
            self.num_blocks = self.runtime_config.cache_size_tokens // self.block_size
        else:
            self.num_blocks = 64
        self._peak_kv_cache_used_blocks = 0
        self.block_manager = BlockManager(self.num_blocks, self.block_size)
        self._allocate_kv_cache(config)
        self.runtime_config.enforce_eager = True
        self._warmup()

    def _warmup(self) -> None:
        """预热多模态推理热路径，降低首个请求的初始化开销。"""
        pixel_values = torch.zeros(
            (4, 3 * self.image_processor.temporal_patch_size * self.image_processor.patch_size * self.image_processor.patch_size),
            dtype=torch.bfloat16,
            device=self.device,
        )
        image_grid_thw = torch.tensor([[1, self.image_processor.merge_size, self.image_processor.merge_size]], dtype=torch.long, device=self.device)
        image_token_id = getattr(self.model.config, "image_token_id", 151655)
        input_ids = torch.tensor([[151644, image_token_id, 151645]], dtype=torch.long, device=self.device)
        sp = SamplingParams(temperature=0.0, top_k=0, top_p=1.0, max_tokens=1)
        try:
            _ = self.generate("warmup", None, sp, apply_chat_template=False)
            _ = self.model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                max_new_tokens=1,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                eos_token_id=[151645],
                ignore_eos=True,
            )
        except Exception:
            pass

    def _prepare_inputs(
        self,
        prompt: str | list[dict],
        images: list[Image.Image] | None,
        apply_chat_template: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """准备多模态输入。"""
        pixel_values = None
        image_grid_thw = None
        if images is not None and len(images) > 0:
            processed = self.image_processor(images)
            pixel_values = processed["pixel_values"].to(self.device).to(torch.bfloat16)
            image_grid_thw = processed["image_grid_thw"].to(self.device)

        if isinstance(prompt, str):
            if not apply_chat_template:
                prompt_text = prompt
            else:
                content_list = []
                if images is not None:
                    for _ in images:
                        content_list.append({"type": "image", "image": "dummy"})
                content_list.append({"type": "text", "text": prompt})
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": content_list},
                ]
                prompt_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            if not apply_chat_template:
                raise ValueError("当 prompt 为 list[dict] 格式时，必须设置 apply_chat_template=True")
            prompt_text = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

        raw_input_ids = self.tokenizer.encode(prompt_text)
        image_token_id = getattr(self.model.config, "image_token_id", 151655)
        if image_grid_thw is not None:
            assert image_token_id in raw_input_ids, f"聊天模板未生成图像占位符 {image_token_id}，无法注入图像特征"
            input_ids = []
            img_idx = 0
            for tid in raw_input_ids:
                if tid == image_token_id:
                    if img_idx < image_grid_thw.shape[0]:
                        grid = image_grid_thw[img_idx]
                        num_patches = grid[0] * grid[1] * grid[2]
                        merged_patches = num_patches.item() // 4
                        input_ids.extend([image_token_id] * merged_patches)
                        img_idx += 1
                    else:
                        input_ids.append(tid)
                else:
                    input_ids.append(tid)
        else:
            input_ids = raw_input_ids

        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        return input_ids, pixel_values, image_grid_thw

    def _resolve_eos_token_ids(self) -> list[int]:
        """收集所有可能的结束 token。"""
        eos_token_ids = []
        if hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
            if isinstance(self.tokenizer.eos_token_id, int):
                eos_token_ids.append(self.tokenizer.eos_token_id)
            elif isinstance(self.tokenizer.eos_token_id, list):
                eos_token_ids.extend(self.tokenizer.eos_token_id)
        if hasattr(self.tokenizer, "all_special_ids"):
            for tid in [151645, 151643]:
                if tid not in eos_token_ids:
                    eos_token_ids.append(tid)
        return eos_token_ids

    def _generate_core(
        self,
        prompt: str | list[dict],
        images: list[Image.Image] | None,
        sampling_params: SamplingParams,
        apply_chat_template: bool,
    ):
        """执行统一的 VLM 生成核心逻辑。"""
        input_ids, pixel_values, image_grid_thw = self._prepare_inputs(prompt, images, apply_chat_template)
        eos_token_ids = self._resolve_eos_token_ids()
        ignore_eos = getattr(sampling_params, "ignore_eos", False)

        self.reset_peak_memory_stats()
        seq = self._build_vlm_sequence(input_ids, sampling_params)
        start = perf_counter()
        first_token_time = None

        _, seq_len = input_ids.shape
        self.model._reset_cache()
        from minillm.utils.context import reset_context, set_context

        completion_token_ids = []
        decode_positions = torch.empty(1, dtype=torch.long, device=input_ids.device)
        decode_slot_mapping = torch.empty(1, dtype=torch.int32, device=input_ids.device)
        decode_context_lens = torch.empty(1, dtype=torch.int32, device=input_ids.device)
        decode_block_tables = None
        for step in range(sampling_params.max_tokens):
            if step == 0:
                current_input_ids = input_ids
                positions, cu_seqlens, slot_mapping = self._prepare_prefill_context(seq, input_ids)
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
                positions, slot_mapping, context_lens, block_tables = self._prepare_decode_context(
                    seq,
                    input_ids,
                    decode_positions,
                    decode_slot_mapping,
                    decode_context_lens,
                    decode_block_tables,
                )
                decode_block_tables = block_tables
                set_context(
                    is_prefill=False,
                    slot_mapping=slot_mapping,
                    context_lens=context_lens,
                    block_tables=block_tables,
                )
                current_pixel_values = None
                current_image_grid_thw = None

            try:
                with torch.inference_mode():
                    if step > 0 and not self.runtime_config.enforce_eager:
                        logits = self._run_decode_graph(
                            current_input_ids,
                            positions,
                            slot_mapping,
                            context_lens,
                            block_tables,
                        )
                    else:
                        logits = self.model.forward(
                            input_ids=current_input_ids,
                            positions=positions,
                            pixel_values=current_pixel_values,
                            image_grid_thw=current_image_grid_thw,
                            use_cache=True,
                        )
            finally:
                reset_context()

            next_token_logits = logits[:, -1, :]
            if sampling_params.temperature > 0:
                next_token_logits = next_token_logits / sampling_params.temperature
                if sampling_params.top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, sampling_params.top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                if sampling_params.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > sampling_params.top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            token_id = next_token.item()
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            seq.append_token(token_id)
            self.block_manager.may_append(seq)
            seq_len += 1
            self._peak_kv_cache_used_blocks = max(self._peak_kv_cache_used_blocks, len(self.block_manager.used_block_ids))
            completion_token_ids.append(token_id)

            if first_token_time is None:
                first_token_time = perf_counter() - start

            yield {
                "type": "token",
                "seq_id": 0,
                "token_id": token_id,
                "text": self.tokenizer.decode([token_id], skip_special_tokens=True),
                "generated_tokens": len(completion_token_ids),
            }

            if not ignore_eos and token_id in eos_token_ids:
                break

        total_time = perf_counter() - start
        ttft_seconds = total_time if first_token_time is None else first_token_time
        decode_time = max(total_time - ttft_seconds, 0.0)
        generated_tokens = len(completion_token_ids)
        overall_tokens_per_second = generated_tokens / total_time if total_time > 0 else 0.0
        decode_tokens_per_second = max(generated_tokens - 1, 0) / decode_time if decode_time > 0 else 0.0

        metrics = {
            "type": "metrics",
            "seq_id": 0,
            "prompt_tokens": input_ids.shape[1] - generated_tokens,
            "generated_tokens": generated_tokens,
            "ttft_seconds": ttft_seconds,
            "total_time_seconds": total_time,
            "overall_tokens_per_second": overall_tokens_per_second,
            "decode_tokens_per_second": decode_tokens_per_second,
            "text": self.tokenizer.decode(completion_token_ids, skip_special_tokens=True),
            "token_ids": completion_token_ids,
        }
        metrics.update(self.get_memory_profile())
        yield metrics

    @torch.inference_mode()
    def generate(
        self,
        prompt: str | list[dict],
        images: list[Image.Image] | None = None,
        sampling_params: SamplingParams | None = None,
        apply_chat_template: bool = True,
    ) -> dict:
        """生成响应。

        Args:
            prompt: 文本提示或对话格式的 list[dict]
            images: 图像列表（可选）
            sampling_params: 采样参数

        Returns:
            生成结果字典 {"text": str, "token_ids": list[int]}
        """
        if sampling_params is None:
            sampling_params = SamplingParams()

        metrics = None
        for event in self._generate_core(prompt, images, sampling_params, apply_chat_template):
            if event["type"] == "metrics":
                metrics = event
        assert metrics is not None
        return {
            "text": metrics["text"],
            "token_ids": metrics["token_ids"],
        }

    @torch.inference_mode()
    def batch_generate(
        self,
        requests: list[dict],
        sampling_params: SamplingParams | None = None,
        apply_chat_template: bool = True,
    ) -> list[dict]:
        """批量生成响应。

        Args:
            requests: 请求列表，每个请求格式为 {"text": str, "images": list[Image]}
            sampling_params: 采样参数
            apply_chat_template: 是否应用聊天模板

        Returns:
            生成结果列表，每个结果为 {"text": str, "token_ids": list[int]}
        """
        if sampling_params is None:
            sampling_params = SamplingParams()

        results = []
        for req in requests:
            prompt = req.get("text", "")
            images = req.get("images", None)

            # 每次请求前释放 KV Cache
            self.block_manager.free_all()
            self.model._reset_cache()

            result = self.generate(prompt, images, sampling_params, apply_chat_template)
            results.append(result)

        return results

    def get_kv_cache_bytes(self) -> int:
        return self.kv_cache.numel() * self.kv_cache.element_size()

    def get_model_bytes(self) -> int:
        parameter_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in self.model.buffers())
        return parameter_bytes + buffer_bytes

    def get_memory_profile(self) -> dict:
        kv_cache_total_bytes = self.get_kv_cache_bytes()
        kv_cache_total_blocks = self.num_blocks
        kv_cache_used_blocks = self._peak_kv_cache_used_blocks
        kv_cache_peak_used_blocks = self._peak_kv_cache_used_blocks
        kv_cache_used_bytes = 0
        kv_cache_peak_used_bytes = 0
        if kv_cache_total_blocks > 0:
            kv_cache_used_bytes = kv_cache_total_bytes * kv_cache_used_blocks // kv_cache_total_blocks
            kv_cache_peak_used_bytes = kv_cache_total_bytes * kv_cache_peak_used_blocks // kv_cache_total_blocks
        return {
            "model_bytes": self.get_model_bytes(),
            "kv_cache_total_bytes": kv_cache_total_bytes,
            "kv_cache_used_bytes": kv_cache_used_bytes,
            "kv_cache_peak_used_bytes": kv_cache_peak_used_bytes,
            "kv_cache_total_blocks": kv_cache_total_blocks,
            "kv_cache_used_blocks": kv_cache_used_blocks,
            "kv_cache_peak_used_blocks": kv_cache_peak_used_blocks,
            "cuda_memory_allocated_bytes": torch.cuda.memory_allocated(),
            "cuda_max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_memory_reserved_bytes": torch.cuda.memory_reserved(),
            "cuda_max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        }

    def reset_peak_memory_stats(self) -> None:
        torch.cuda.reset_peak_memory_stats()
        self._peak_kv_cache_used_blocks = 0

    def _build_vlm_sequence(self, input_ids: torch.Tensor, sampling_params: SamplingParams) -> Sequence:
        """基于输入 token 构建单序列并分配 paged KV blocks。"""
        seq = Sequence(input_ids[0].tolist(), sampling_params)
        seq.block_size = self.block_size
        self.block_manager.allocate(seq)
        self._peak_kv_cache_used_blocks = max(self._peak_kv_cache_used_blocks, len(self.block_manager.used_block_ids))
        return seq

    def _prepare_prefill_context(self, seq: Sequence, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """准备 prefill 上下文。"""
        seq_len = input_ids.shape[1]
        positions = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=input_ids.device)
        slot_mapping = torch.empty(seq_len, dtype=torch.int32, device=input_ids.device)
        offset = 0
        for block_idx, block_id in enumerate(seq.block_table):
            start = block_id * self.block_size
            if block_idx != len(seq.block_table) - 1:
                end = start + self.block_size
            else:
                end = start + seq.last_block_num_tokens
            chunk_len = end - start
            slot_mapping[offset:offset + chunk_len] = torch.arange(start, end, dtype=torch.int32, device=input_ids.device)
            offset += chunk_len
        return positions, cu_seqlens, slot_mapping

    def _prepare_decode_context(
        self,
        seq: Sequence,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        slot_mapping: torch.Tensor | None = None,
        context_lens: torch.Tensor | None = None,
        block_tables: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """准备 decode 上下文。"""
        seq_len = input_ids.shape[1]
        if positions is None:
            positions = torch.empty(1, dtype=torch.long, device=input_ids.device)
        if slot_mapping is None:
            slot_mapping = torch.empty(1, dtype=torch.int32, device=input_ids.device)
        if context_lens is None:
            context_lens = torch.empty(1, dtype=torch.int32, device=input_ids.device)
        if block_tables is None or block_tables.shape[1] != len(seq.block_table):
            block_tables = torch.empty((1, len(seq.block_table)), dtype=torch.int32, device=input_ids.device)

        positions[0] = seq_len - 1
        slot_mapping[0] = seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
        context_lens[0] = seq_len
        block_tables[0, :len(seq.block_table)] = torch.tensor(seq.block_table, dtype=torch.int32, device=input_ids.device)
        return positions, slot_mapping, context_lens, block_tables

    def generate_stream(
        self,
        prompt: str | list[dict],
        images: list[Image.Image] | None = None,
        sampling_params: SamplingParams | None = None,
        apply_chat_template: bool = True,
    ):
        if sampling_params is None:
            sampling_params = SamplingParams()
        yield from self._generate_core(prompt, images, sampling_params, apply_chat_template)

    def _allocate_kv_cache(self, config):
        """分配 KV cache（简化版本）。"""
        block_size = self.block_size
        num_blocks = self.num_blocks

        num_layers = config.text_config.num_hidden_layers
        num_kv_heads = config.text_config.num_key_value_heads
        head_dim = config.text_config.hidden_size // config.text_config.num_attention_heads

        self.kv_cache = torch.empty(
            2,
            num_layers,
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )

        layer_id = 0
        for module in self.model.language_model.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
