"""LLM Engine — 推理引擎入口类。"""

from dataclasses import fields
from time import perf_counter

from tqdm.auto import tqdm
from transformers import AutoTokenizer

from minillm.config import Config
from minillm.engine.model_runner import ModelRunner
from minillm.engine.scheduler import Scheduler
from minillm.engine.sequence import Sequence
from minillm.sampling_params import SamplingParams


class LLMEngine:
    """推理引擎入口类。

    管理请求生命周期：添加请求 → 调度 → 推理 → 收集结果。

    Attributes:
        model_runner: 模型运行器
        tokenizer: 分词器
        scheduler: 调度器
    """

    def __init__(self, model: str, **kwargs):
        """初始化推理引擎。

        Args:
            model: 模型路径
            **kwargs: 其他配置参数
        """
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.max_model_len = config.max_model_len
        self.model_runner = ModelRunner(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True, trust_remote_code=True)
        # 收集所有可能的 EOS token id
        eos_ids = []
        if self.tokenizer.eos_token_id is not None:
            if isinstance(self.tokenizer.eos_token_id, int):
                eos_ids.append(self.tokenizer.eos_token_id)
            else:
                eos_ids.extend(self.tokenizer.eos_token_id)
                
        # 兼容 Qwen 等模型的额外特殊结束符
        # Qwen: <|im_end|> -> 151645, <|endoftext|> -> 151643
        if hasattr(self.tokenizer, "all_special_ids"):
            for tid in [151645, 151643]:
                if tid in getattr(self.tokenizer, "all_special_ids", []) and tid not in eos_ids:
                    eos_ids.append(tid)
                    
        # 兜底：如果都没有，则用默认 0（虽然很少见）
        config.eos = eos_ids if eos_ids else 0
        self.scheduler = Scheduler(config)
        self._peak_kv_cache_used_blocks = 0

    def add_request(self, prompt: str | list[int] | list[dict], sampling_params: SamplingParams, apply_chat_template: bool = True) -> int:
        """添加一个推理请求。

        Args:
            prompt: 文本 prompt、对话列表或 token id 列表
            sampling_params: 采样参数
            apply_chat_template: 是否应用聊天模板

        Returns:
            序列 id
        """
        if isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], dict):
            # 对话列表格式
            prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            prompt = self.tokenizer.encode(prompt)
        elif isinstance(prompt, str):
            # 普通文本格式，尝试使用自动对话模板
            if apply_chat_template and hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template is not None:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
                try:
                    chat_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    prompt = self.tokenizer.encode(chat_prompt)
                except Exception:
                    prompt = self.tokenizer.encode(prompt)
            else:
                prompt = self.tokenizer.encode(prompt)
        required_len = len(prompt) + sampling_params.max_tokens
        assert required_len <= self.max_model_len, (
            f"请求长度超过 max_model_len: required={required_len}, max_model_len={self.max_model_len}. "
            f"请增大 max_model_len 或减小 max_tokens。"
        )
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq.seq_id

    def step_with_details(self) -> tuple[list[tuple[int, list[int]]], int, list[tuple[int, int]]]:
        """执行一步推理，并返回本步生成细节。

        Returns:
            (outputs, num_tokens, step_token_ids) 元组
            outputs 为完成的 (seq_id, token_ids) 列表
            step_token_ids 为本步生成的 (seq_id, token_id) 列表
        """
        seqs, is_prefill = self.scheduler.schedule()
        self._peak_kv_cache_used_blocks = max(
            self._peak_kv_cache_used_blocks,
            len(self.scheduler.block_manager.used_block_ids),
        )
        token_ids = self.model_runner.run(seqs, is_prefill)
        step_token_ids = [(seq.seq_id, token_id) for seq, token_id in zip(seqs, token_ids)]
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens, step_token_ids

    def step(self) -> tuple[list[tuple[int, list[int]]], int]:
        """执行一步推理。

        Returns:
            (outputs, num_tokens) 元组
            outputs 为完成的 (seq_id, token_ids) 列表
        """
        outputs, num_tokens, _ = self.step_with_details()
        return outputs, num_tokens

    def is_finished(self) -> bool:
        """检查是否所有请求处理完毕。"""
        return self.scheduler.is_finished()

    def reset_peak_memory_stats(self) -> None:
        """重置 CUDA 峰值显存统计。"""
        self.model_runner.reset_peak_memory_stats()

    def get_memory_profile(self) -> dict:
        """返回当前引擎显存画像。"""
        kv_cache_used_blocks = len(self.scheduler.block_manager.used_block_ids)
        return self.model_runner.get_memory_profile(kv_cache_used_blocks, self._peak_kv_cache_used_blocks)

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
        apply_chat_template: bool = True,
    ) -> list[dict]:
        """批量生成。

        Args:
            prompts: prompt 列表（文本或 token id）
            sampling_params: 采样参数（单个或列表）
            use_tqdm: 是否显示进度条
            apply_chat_template: 是否应用聊天模板

        Returns:
            生成结果列表，每项为 {"text": str, "token_ids": list[int]}
        """
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        assert len(sampling_params) == len(prompts), "sampling_params 数量必须与 prompts 数量一致"
        self._peak_kv_cache_used_blocks = 0
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp, apply_chat_template)
        outputs = {}
        prefill_throughput = decode_throughput = 0.0
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [
            {"text": self.tokenizer.decode(token_ids, skip_special_tokens=True), "token_ids": token_ids}
            for token_ids in outputs
        ]
        if use_tqdm:
            pbar.close()
        return outputs

    def generate_stream(self, prompt: str | list[int] | list[dict], sampling_params: SamplingParams, apply_chat_template: bool = True):
        """流式生成单个请求。

        Args:
            prompt: 文本 prompt 或 token id 列表
            sampling_params: 采样参数
            apply_chat_template: 是否应用聊天模板

        Yields:
            流式事件字典。
            token 事件格式：
            {
                "type": "token",
                "seq_id": int,
                "token_id": int,
                "text": str,
                "generated_tokens": int,
            }
            metrics 事件格式：
            {
                "type": "metrics",
                "seq_id": int,
                "prompt_tokens": int,
                "generated_tokens": int,
                "ttft_seconds": float,
                "total_time_seconds": float,
                "overall_tokens_per_second": float,
                "decode_tokens_per_second": float,
                "text": str,
                "token_ids": list[int],
            }
        """
        self._peak_kv_cache_used_blocks = 0
        prompt_token_ids = self.tokenizer.encode(prompt) if isinstance(prompt, str) else prompt
        seq_id = self.add_request(prompt, sampling_params, apply_chat_template)
        start = perf_counter()
        first_token_time = None
        completion_token_ids: list[int] = []

        while not self.is_finished():
            _, _, step_token_ids = self.step_with_details()
            step_finished_at = perf_counter()
            for current_seq_id, token_id in step_token_ids:
                if current_seq_id != seq_id:
                    continue
                completion_token_ids.append(token_id)
                if first_token_time is None:
                    first_token_time = step_finished_at - start
                yield {
                    "type": "token",
                    "seq_id": seq_id,
                    "token_id": token_id,
                    "text": self.tokenizer.decode([token_id], skip_special_tokens=True),
                    "generated_tokens": len(completion_token_ids),
                }

        total_time = perf_counter() - start
        ttft_seconds = total_time if first_token_time is None else first_token_time
        decode_time = max(total_time - ttft_seconds, 0.0)
        generated_tokens = len(completion_token_ids)
        overall_tokens_per_second = generated_tokens / total_time if total_time > 0 else 0.0
        decode_tokens_per_second = (
            max(generated_tokens - 1, 0) / decode_time if decode_time > 0 else 0.0
        )
        metrics = {
            "type": "metrics",
            "seq_id": seq_id,
            "prompt_tokens": len(prompt_token_ids),
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


class LLM(LLMEngine):
    """LLM 简化包装类。"""

    pass
