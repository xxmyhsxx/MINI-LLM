from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from minillm.engine.llm_engine import LLM
from minillm.sampling_params import SamplingParams


@dataclass
class BenchmarkResult:
    """单个文本基准测试结果。"""

    engine: str
    batch_size: int
    target_input_length: int
    actual_input_length: int
    output_length: int
    elapsed_seconds: float
    generated_tokens: int
    throughput_tokens_per_second: float


def clear_cuda_cache() -> None:
    """清理 GPU 侧缓存，尽量减少相邻测例干扰。"""
    gc.collect()
    torch.cuda.empty_cache()


def build_text_prompts(
    tokenizer: AutoTokenizer,
    batch_size: int,
    target_input_length: int,
) -> tuple[list[str], int]:
    """构造文本 prompt，并返回实际 token 长度。"""
    base_token_ids = tokenizer.encode(
        "hello benchmark sequence for minillm performance measurement",
        add_special_tokens=False,
    )
    assert base_token_ids
    prompts: list[str] = []
    actual_length = 0
    for i in range(batch_size):
        token_ids = [base_token_ids[(j + i) % len(base_token_ids)] for j in range(target_input_length)]
        text = tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        prompts.append(text)
        actual_length = len(tokenizer.encode(text, add_special_tokens=False))
    return prompts, actual_length


def benchmark_minillm(
    model_path: str,
    prompts: list[str],
    actual_input_length: int,
    output_length: int,
    max_model_len: int,
) -> BenchmarkResult:
    """测试 minillm 文本输入性能。"""
    batch_size = len(prompts)
    llm = LLM(model=model_path, max_num_seqs=batch_size, max_model_len=max_model_len)
    sampling_params = SamplingParams(temperature=0, max_tokens=output_length, ignore_eos=True)
    llm.generate(
        ["warmup prompt"] * batch_size,
        SamplingParams(temperature=0, max_tokens=1, ignore_eos=True),
        use_tqdm=False,
    )
    torch.cuda.synchronize()
    start = perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    torch.cuda.synchronize()
    elapsed = perf_counter() - start
    generated_tokens = sum(len(output["token_ids"]) for output in outputs)
    return BenchmarkResult(
        "minillm",
        batch_size,
        actual_input_length,
        actual_input_length,
        output_length,
        elapsed,
        generated_tokens,
        generated_tokens / elapsed,
    )


def benchmark_hf(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    target_input_length: int,
    output_length: int,
) -> BenchmarkResult:
    """测试 HuggingFace 文本输入性能。"""
    batch_size = len(prompts)
    warmup_inputs = tokenizer(["warmup prompt"] * batch_size, return_tensors="pt", padding=True).to("cuda")
    with torch.inference_mode():
        _ = model.generate(
            **warmup_inputs,
            do_sample=False,
            max_new_tokens=1,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=None,
        )
        torch.cuda.synchronize()
    start = perf_counter()
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
    actual_input_length = int(inputs["input_ids"].shape[1])
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=output_length,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=None,
        )
        torch.cuda.synchronize()
    elapsed = perf_counter() - start
    generated_tokens = int(outputs.shape[0] * (outputs.shape[1] - inputs["input_ids"].shape[1]))
    return BenchmarkResult(
        "huggingface_pytorch_text",
        batch_size,
        target_input_length,
        actual_input_length,
        output_length,
        elapsed,
        generated_tokens,
        generated_tokens / elapsed,
    )


def render_markdown(results: list[BenchmarkResult], model_path: str, hf_attn_implementation: str) -> str:
    """渲染 Markdown 结果。"""
    lines = [
        "# Text Benchmark Report",
        "",
        f"- Model: `{model_path}`",
        "- Input Mode: text prompt",
        f"- HuggingFace Attention: `{hf_attn_implementation}`",
        "",
        "| Engine | Batch Size | Target Input Len | Actual Input Len | Output Length | Time (s) | Generated Tokens | Throughput (tok/s) |",
        "|--------|------------|------------------|------------------|---------------|----------|------------------|--------------------|",
    ]
    for item in results:
        lines.append(
            f"| {item.engine} | {item.batch_size} | {item.target_input_length} | {item.actual_input_length} | {item.output_length} | "
            f"{item.elapsed_seconds:.4f} | {item.generated_tokens} | {item.throughput_tokens_per_second:.2f} |"
        )
    return "\n".join(lines) + "\n"


def parse_int_list(raw: str) -> list[int]:
    """解析整数列表。"""
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    """执行文本 benchmark。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/app/models/qwen2.5-1.5B-Instruct")
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--input-lengths", default="128,512,1024")
    parser.add_argument("--output-length", type=int, default=32)
    parser.add_argument("--hf-attn-implementation", default="sdpa")
    parser.add_argument("--output-json", default="/app/minillm/benchmark_results/qwen2_text_benchmark.json")
    parser.add_argument("--output-md", default="/app/minillm/benchmark_results/qwen2_text_benchmark.md")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    batch_sizes = parse_int_list(args.batch_sizes)
    input_lengths = parse_int_list(args.input_lengths)
    max_model_len = max(input_lengths) + args.output_length + 32
    results: list[BenchmarkResult] = []

    for batch_size in batch_sizes:
        for input_length in input_lengths:
            prompts, actual_input_length = build_text_prompts(tokenizer, batch_size, input_length)
            clear_cuda_cache()
            results.append(
                benchmark_minillm(
                    args.model,
                    prompts,
                    actual_input_length,
                    args.output_length,
                    max_model_len,
                )
            )

    clear_cuda_cache()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype="auto",
        attn_implementation=args.hf_attn_implementation,
    ).cuda().eval()
    for batch_size in batch_sizes:
        for input_length in input_lengths:
            prompts, _ = build_text_prompts(tokenizer, batch_size, input_length)
            clear_cuda_cache()
            results.append(benchmark_hf(model, tokenizer, prompts, input_length, args.output_length))

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.write_text(json.dumps([asdict(item) for item in results], indent=2))
    output_md.write_text(render_markdown(results, args.model, args.hf_attn_implementation))
    print(output_md.read_text())


if __name__ == "__main__":
    main()
