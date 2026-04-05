"""Qwen3 模型 benchmark：minillm vs nano-vllm vs HuggingFace PyTorch。

对比三个引擎在同一模型（Qwen3-0.6B）上的推理性能，包括吞吐量和延迟。

运行方式：
    python benchmarks/benchmark_qwen3.py --model /app/models/Qwen3-0.6B

依赖：需要 GPU，需要下载好的 Qwen3-0.6B 模型权重。
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from minillm.engine.llm_engine import LLM
from minillm.sampling_params import SamplingParams


@dataclass
class BenchmarkResult:
    """单个基准测试结果。"""

    engine: str
    batch_size: int
    input_length: int
    output_length: int
    elapsed_seconds: float
    generated_tokens: int
    throughput_tokens_per_second: float


def clear_cuda_cache() -> None:
    """清理 GPU 侧缓存，尽量减少相邻测例干扰。"""
    gc.collect()
    torch.cuda.empty_cache()


def build_prompt_token_ids(
    tokenizer: AutoTokenizer,
    batch_size: int,
    input_length: int,
) -> list[list[int]]:
    """构造固定长度且互不相同的输入 token。

    Args:
        tokenizer: 分词器
        batch_size: 批大小
        input_length: 每条输入的 token 数

    Returns:
        token id 列表的列表
    """
    base_token_ids = tokenizer.encode("hello benchmark sequence", add_special_tokens=False)
    assert base_token_ids
    prompts: list[list[int]] = []
    for i in range(batch_size):
        token_ids = [base_token_ids[(j + i) % len(base_token_ids)] for j in range(input_length)]
        token_ids[0] = base_token_ids[i % len(base_token_ids)]
        prompts.append(token_ids)
    return prompts


def benchmark_minillm(
    model_path: str,
    prompt_token_ids: list[list[int]],
    output_length: int,
    max_model_len: int,
) -> BenchmarkResult:
    """测试 minillm 推理性能。

    Args:
        model_path: 模型路径
        prompt_token_ids: 输入 token id 列表
        output_length: 生成 token 数
        max_model_len: 最大上下文长度

    Returns:
        基准测试结果
    """
    batch_size = len(prompt_token_ids)
    input_length = len(prompt_token_ids[0])
    llm = LLM(model=model_path, max_num_seqs=batch_size, max_model_len=max_model_len)
    sampling_params = SamplingParams(temperature=0, max_tokens=output_length, ignore_eos=True)

    torch.cuda.synchronize()
    start = perf_counter()
    outputs = llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    torch.cuda.synchronize()
    elapsed = perf_counter() - start
    generated_tokens = sum(len(output["token_ids"]) for output in outputs)
    del llm
    clear_cuda_cache()
    return BenchmarkResult(
        engine="minillm",
        batch_size=batch_size,
        input_length=input_length,
        output_length=output_length,
        elapsed_seconds=elapsed,
        generated_tokens=generated_tokens,
        throughput_tokens_per_second=generated_tokens / elapsed,
    )


def benchmark_nanovllm(
    model_path: str,
    prompt_token_ids: list[list[int]],
    output_length: int,
    max_model_len: int,
) -> BenchmarkResult:
    """测试 nano-vllm 推理性能。

    nano-vllm 的 SamplingParams 不支持 temperature=0（会断言失败），
    使用极小温度近似 greedy。
    由于 nano-vllm 初始化 torch.distributed 且无法清理，整个进程只能调用一次。

    Args:
        model_path: 模型路径
        prompt_token_ids: 输入 token id 列表
        output_length: 生成 token 数
        max_model_len: 最大上下文长度

    Returns:
        基准测试结果
    """
    from nanovllm import LLM as NanoLLM
    from nanovllm.sampling_params import SamplingParams as NanoSamplingParams

    batch_size = len(prompt_token_ids)
    input_length = len(prompt_token_ids[0])

    clear_cuda_cache()
    llm = NanoLLM(model_path, enforce_eager=False, max_model_len=max_model_len)
    sampling_params = [
        NanoSamplingParams(temperature=0.001, max_tokens=output_length, ignore_eos=True)
        for _ in range(batch_size)
    ]

    torch.cuda.synchronize()
    start = perf_counter()
    outputs = llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    torch.cuda.synchronize()
    elapsed = perf_counter() - start
    generated_tokens = sum(len(output["token_ids"]) for output in outputs)
    del llm
    clear_cuda_cache()
    return BenchmarkResult(
        engine="nano-vllm",
        batch_size=batch_size,
        input_length=input_length,
        output_length=output_length,
        elapsed_seconds=elapsed,
        generated_tokens=generated_tokens,
        throughput_tokens_per_second=generated_tokens / elapsed,
    )


def benchmark_hf(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_token_ids: list[list[int]],
    output_length: int,
) -> BenchmarkResult:
    """测试 HuggingFace PyTorch 推理性能。

    Args:
        model: HuggingFace 模型
        tokenizer: 分词器
        prompt_token_ids: 输入 token id 列表
        output_length: 生成 token 数

    Returns:
        基准测试结果
    """
    batch_size = len(prompt_token_ids)
    input_length = len(prompt_token_ids[0])
    input_ids = torch.tensor(prompt_token_ids, dtype=torch.long, device="cuda")
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        # warmup
        _ = model.generate(
            input_ids=input_ids[:1],
            attention_mask=attention_mask[:1],
            do_sample=False,
            max_new_tokens=1,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=None,
        )
        torch.cuda.synchronize()
        start = perf_counter()
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=output_length,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=None,
        )
        torch.cuda.synchronize()
    elapsed = perf_counter() - start
    generated_tokens = int(outputs.shape[0] * (outputs.shape[1] - input_ids.shape[1]))
    return BenchmarkResult(
        engine="huggingface_pytorch",
        batch_size=batch_size,
        input_length=input_length,
        output_length=output_length,
        elapsed_seconds=elapsed,
        generated_tokens=generated_tokens,
        throughput_tokens_per_second=generated_tokens / elapsed,
    )


def render_markdown(
    results: list[BenchmarkResult],
    model_path: str,
    hf_attn_implementation: str,
) -> str:
    """将结果渲染为 Markdown 表格。

    Args:
        results: 基准测试结果列表
        model_path: 模型路径
        hf_attn_implementation: HuggingFace 注意力实现

    Returns:
        Markdown 格式表格
    """
    lines = [
        "# Qwen3 Benchmark Report",
        "",
        f"- Model: `{model_path}`",
        f"- HuggingFace Attention: `{hf_attn_implementation}`",
        "- Date: generated by script",
        "",
        "| Engine | Batch Size | Input Length | Output Length | Time (s) | Generated Tokens | Throughput (tok/s) |",
        "|--------|------------|--------------|---------------|----------|------------------|--------------------|",
    ]
    for item in results:
        lines.append(
            f"| {item.engine} | {item.batch_size} | {item.input_length} | {item.output_length} | "
            f"{item.elapsed_seconds:.4f} | {item.generated_tokens} | {item.throughput_tokens_per_second:.2f} |"
        )
    return "\n".join(lines) + "\n"


def parse_int_list(raw: str) -> list[int]:
    """解析逗号分隔的整数列表。

    Args:
        raw: 逗号分隔的字符串

    Returns:
        整数列表
    """
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _run_nanovllm_in_subprocess(
    model_path: str,
    batch_sizes: list[int],
    input_lengths: list[int],
    output_length: int,
    max_model_len: int,
) -> list[BenchmarkResult]:
    """在子进程中运行 nano-vllm benchmark。

    nano-vllm 初始化 torch.distributed 后无法在同一进程中清理，
    所以将所有 nano-vllm 测试放在同一个子进程中执行。

    Args:
        model_path: 模型路径
        batch_sizes: 批大小列表
        input_lengths: 输入长度列表
        output_length: 生成 token 数
        max_model_len: 最大上下文长度

    Returns:
        基准测试结果列表
    """
    script = f'''
import sys, gc, json, os
sys.path.insert(0, "/app/minillm/nano-vllm")
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29599"
import torch
import torch.distributed as dist
if dist.is_initialized():
    dist.destroy_process_group()
from nanovllm import LLM
from nanovllm.sampling_params import SamplingParams
from transformers import AutoTokenizer

model_path = "{model_path}"
max_model_len = {max_model_len}
output_length = {output_length}
batch_sizes = {batch_sizes}
input_lengths = {input_lengths}

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

def build_prompt_token_ids(batch_size, input_length):
    base_token_ids = tokenizer.encode("hello benchmark sequence", add_special_tokens=False)
    prompts = []
    for i in range(batch_size):
        token_ids = [base_token_ids[(j + i) % len(base_token_ids)] for j in range(input_length)]
        token_ids[0] = base_token_ids[i % len(base_token_ids)]
        prompts.append(token_ids)
    return prompts

results = []
for batch_size in batch_sizes:
    for input_length in input_lengths:
        gc.collect()
        torch.cuda.empty_cache()
        prompt_token_ids = build_prompt_token_ids(batch_size, input_length)
        llm = LLM(model_path, enforce_eager=False, max_model_len=max_model_len)
        sp = [SamplingParams(temperature=0.001, max_tokens=output_length, ignore_eos=True) for _ in range(batch_size)]
        torch.cuda.synchronize()
        from time import perf_counter
        start = perf_counter()
        outputs = llm.generate(prompt_token_ids, sp, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed = perf_counter() - start
        generated_tokens = sum(len(o["token_ids"]) for o in outputs)
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        results.append({{
            "engine": "nano-vllm",
            "batch_size": batch_size,
            "input_length": input_length,
            "output_length": output_length,
            "elapsed_seconds": elapsed,
            "generated_tokens": generated_tokens,
            "throughput_tokens_per_second": generated_tokens / elapsed,
        }})
print(json.dumps(results))
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=600,
        env={**os.environ, "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29500"},
    )
    if result.returncode != 0:
        print(f"nano-vllm subprocess failed: {result.stderr}", file=sys.stderr)
        return []
    return [BenchmarkResult(**r) for r in json.loads(result.stdout.strip().split("\n")[-1])]


def main() -> None:
    """执行 benchmark。

    注意：nano-vllm 初始化 torch.distributed 后无法在同一进程中清理，
    因此将 nano-vllm 测试放在子进程中执行。
    """
    parser = argparse.ArgumentParser(description="Qwen3 模型 benchmark：minillm vs nano-vllm vs HuggingFace")
    parser.add_argument("--model", default="/app/models/Qwen3-0.6B", help="模型路径")
    parser.add_argument("--batch-sizes", default="1,2,4", help="逗号分隔的批大小列表")
    parser.add_argument("--input-lengths", default="128,512", help="逗号分隔的输入长度列表")
    parser.add_argument("--output-length", type=int, default=32, help="每条输出的生成 token 数")
    parser.add_argument("--hf-attn-implementation", default="sdpa", help="HuggingFace 注意力实现")
    parser.add_argument(
        "--output-json",
        default="/app/minillm/benchmark_results/qwen3_benchmark.json",
        help="JSON 输出路径",
    )
    parser.add_argument(
        "--output-md",
        default="/app/minillm/benchmark_results/qwen3_benchmark.md",
        help="Markdown 输出路径",
    )
    args = parser.parse_args()

    clear_cuda_cache()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    batch_sizes = parse_int_list(args.batch_sizes)
    input_lengths = parse_int_list(args.input_lengths)
    max_model_len = max(input_lengths) + args.output_length + 16
    results: list[BenchmarkResult] = []

    # --- nano-vllm（子进程执行，避免 torch.distributed 冲突）---
    nano_results = _run_nanovllm_in_subprocess(args.model, batch_sizes, input_lengths, args.output_length, max_model_len)
    results.extend(nano_results)

    # --- minillm ---
    for batch_size in batch_sizes:
        for input_length in input_lengths:
            prompt_token_ids = build_prompt_token_ids(tokenizer, batch_size, input_length)
            clear_cuda_cache()
            results.append(
                benchmark_minillm(args.model, prompt_token_ids, args.output_length, max_model_len)
            )

    # --- HuggingFace PyTorch ---
    clear_cuda_cache()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype="auto",
        attn_implementation=args.hf_attn_implementation,
    ).cuda().eval()

    for batch_size in batch_sizes:
        for input_length in input_lengths:
            prompt_token_ids = build_prompt_token_ids(tokenizer, batch_size, input_length)
            clear_cuda_cache()
            results.append(
                benchmark_hf(model, tokenizer, prompt_token_ids, args.output_length)
            )

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps([asdict(item) for item in results], indent=2))
    output_md.write_text(render_markdown(results, args.model, args.hf_attn_implementation))
    print(output_md.read_text())


if __name__ == "__main__":
    main()
