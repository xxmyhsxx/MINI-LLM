"""Qwen2.5-VL 与 HuggingFace 的多维度基准测试。

运行方式：
    source /opt/conda/bin/activate ramc && python /app/minillm/benchmarks/benchmark_vlm_qwen2_5.py

依赖：需要 GPU，需要下载好的 Qwen2.5-VL-3B-Instruct 模型权重。
"""

from __future__ import annotations

import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from minillm.engine.vlm_engine import VLMEngine
from minillm.sampling_params import SamplingParams

MODEL_PATH = "/app/models/Qwen2.5-VL-3B-Instruct"
IMAGE_PATH = "/app/minillm/img/image.png"
PROMPT = "描述这张图片"
OUTPUT_TOKENS = 32
BATCH_SIZES = (1, 2, 4)
OUT_JSON = "/app/minillm/benchmark_results/qwen2_5_vlm_benchmark.json"
OUT_MD = "/app/minillm/benchmark_results/qwen2_5_vlm_benchmark.md"


@dataclass
class BenchmarkResult:
    """单个多模态基准测试结果。"""

    engine: str
    batch_mode: str
    batch_size: int
    prompt_tokens_per_sample: int
    total_prompt_tokens: int
    preprocess_seconds: float | None
    vision_encode_seconds: float | None
    prefill_seconds: float
    prefill_tokens_per_second: float
    avg_ttft_seconds: float
    decode_seconds: float
    elapsed_seconds: float
    generated_tokens_per_sample: int
    total_generated_tokens: int
    overall_tokens_per_second: float
    decode_tokens_per_second: float
    model_bytes: int | None
    kv_cache_total_bytes: int | None
    cuda_max_memory_allocated_bytes: int
    cuda_max_memory_reserved_bytes: int
    text: str


def clear_cuda_cache() -> None:
    """清理 GPU 侧缓存。"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def load_image() -> Image.Image:
    """加载测试图像。"""
    return Image.open(IMAGE_PATH).convert("RGB")


def benchmark_minillm_batch(engine: VLMEngine, batch_size: int) -> BenchmarkResult:
    """测试 minillm 在当前 VLM 串行路径下的批量场景表现。"""
    sampling_params = SamplingParams(
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        max_tokens=OUTPUT_TOKENS,
        ignore_eos=True,
    )

    preprocess_seconds = 0.0
    vision_encode_seconds = 0.0
    prefill_seconds = 0.0
    decode_seconds = 0.0
    elapsed_seconds = 0.0
    total_prompt_tokens = 0
    total_generated_tokens = 0
    max_memory_allocated = 0
    max_memory_reserved = 0
    model_bytes = None
    kv_cache_total_bytes = None
    texts: list[str] = []
    prompt_tokens_per_sample = 0

    for _ in range(batch_size):
        image = load_image()

        preprocess_start = perf_counter()
        input_ids, pixel_values, image_grid_thw = engine._prepare_inputs(PROMPT, [image], True)
        preprocess_seconds += perf_counter() - preprocess_start

        if pixel_values is not None and image_grid_thw is not None:
            with torch.inference_mode():
                vision_start = perf_counter()
                _ = engine.model.visual(pixel_values, image_grid_thw)
                torch.cuda.synchronize()
                vision_encode_seconds += perf_counter() - vision_start

        final_metrics = None
        request_start = perf_counter()
        for event in engine.generate_stream(PROMPT, [image], sampling_params):
            if event["type"] == "metrics":
                final_metrics = event
        elapsed_seconds += perf_counter() - request_start
        assert final_metrics is not None

        prompt_tokens_per_sample = final_metrics["prompt_tokens"]
        total_prompt_tokens += final_metrics["prompt_tokens"]
        total_generated_tokens += final_metrics["generated_tokens"]
        prefill_seconds += final_metrics["ttft_seconds"]
        decode_seconds += max(final_metrics["total_time_seconds"] - final_metrics["ttft_seconds"], 0.0)
        max_memory_allocated = max(max_memory_allocated, final_metrics["cuda_max_memory_allocated_bytes"])
        max_memory_reserved = max(max_memory_reserved, final_metrics["cuda_max_memory_reserved_bytes"])
        model_bytes = final_metrics.get("model_bytes")
        kv_cache_total_bytes = final_metrics.get("kv_cache_total_bytes")
        texts.append(final_metrics["text"])

    avg_ttft_seconds = prefill_seconds / batch_size if batch_size > 0 else 0.0
    prefill_tokens_per_second = total_prompt_tokens / prefill_seconds if prefill_seconds > 0 else 0.0
    decode_token_count = max(total_generated_tokens - batch_size, 0)
    decode_tokens_per_second = decode_token_count / decode_seconds if decode_seconds > 0 else 0.0
    overall_tokens_per_second = total_generated_tokens / elapsed_seconds if elapsed_seconds > 0 else 0.0
    generated_tokens_per_sample = total_generated_tokens // batch_size if batch_size > 0 else 0

    return BenchmarkResult(
        engine="minillm_vlm",
        batch_mode="serial",
        batch_size=batch_size,
        prompt_tokens_per_sample=prompt_tokens_per_sample,
        total_prompt_tokens=total_prompt_tokens,
        preprocess_seconds=preprocess_seconds,
        vision_encode_seconds=vision_encode_seconds,
        prefill_seconds=prefill_seconds,
        prefill_tokens_per_second=prefill_tokens_per_second,
        avg_ttft_seconds=avg_ttft_seconds,
        decode_seconds=decode_seconds,
        elapsed_seconds=elapsed_seconds,
        generated_tokens_per_sample=generated_tokens_per_sample,
        total_generated_tokens=total_generated_tokens,
        overall_tokens_per_second=overall_tokens_per_second,
        decode_tokens_per_second=decode_tokens_per_second,
        model_bytes=model_bytes,
        kv_cache_total_bytes=kv_cache_total_bytes,
        cuda_max_memory_allocated_bytes=max_memory_allocated,
        cuda_max_memory_reserved_bytes=max_memory_reserved,
        text=texts[0] if texts else "",
    )


def benchmark_hf_batch(processor: AutoProcessor, model: Qwen2_5_VLForConditionalGeneration, batch_size: int) -> BenchmarkResult:
    """测试 HuggingFace 在原生批量路径下的多模态表现。"""
    images = [load_image() for _ in range(batch_size)]
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    preprocess_start = perf_counter()
    inputs = processor(text=[text] * batch_size, images=images, padding=True, return_tensors="pt").to("cuda")
    preprocess_seconds = perf_counter() - preprocess_start

    clear_cuda_cache()
    with torch.inference_mode():
        prefill_start = perf_counter()
        _ = model(**inputs, use_cache=True)
        torch.cuda.synchronize()
        prefill_seconds = perf_counter() - prefill_start

    clear_cuda_cache()
    with torch.inference_mode():
        total_start = perf_counter()
        outputs = model.generate(
            **inputs,
            max_new_tokens=OUTPUT_TOKENS,
            temperature=None,
            do_sample=False,
        )
        torch.cuda.synchronize()
        elapsed_seconds = perf_counter() - total_start

    generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
    decode_seconds = max(elapsed_seconds - prefill_seconds, 0.0)
    prompt_tokens_per_sample = int(inputs["input_ids"].shape[1])
    total_prompt_tokens = prompt_tokens_per_sample * batch_size
    total_generated_tokens = int(generated_ids.numel())
    prefill_tokens_per_second = total_prompt_tokens / prefill_seconds if prefill_seconds > 0 else 0.0
    decode_token_count = max(total_generated_tokens - batch_size, 0)
    decode_tokens_per_second = decode_token_count / decode_seconds if decode_seconds > 0 else 0.0
    overall_tokens_per_second = total_generated_tokens / elapsed_seconds if elapsed_seconds > 0 else 0.0
    avg_ttft_seconds = prefill_seconds

    return BenchmarkResult(
        engine="huggingface_vlm",
        batch_mode="native",
        batch_size=batch_size,
        prompt_tokens_per_sample=prompt_tokens_per_sample,
        total_prompt_tokens=total_prompt_tokens,
        preprocess_seconds=preprocess_seconds,
        vision_encode_seconds=None,
        prefill_seconds=prefill_seconds,
        prefill_tokens_per_second=prefill_tokens_per_second,
        avg_ttft_seconds=avg_ttft_seconds,
        decode_seconds=decode_seconds,
        elapsed_seconds=elapsed_seconds,
        generated_tokens_per_sample=int(generated_ids.shape[1]),
        total_generated_tokens=total_generated_tokens,
        overall_tokens_per_second=overall_tokens_per_second,
        decode_tokens_per_second=decode_tokens_per_second,
        model_bytes=None,
        kv_cache_total_bytes=None,
        cuda_max_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
        cuda_max_memory_reserved_bytes=torch.cuda.max_memory_reserved(),
        text=processor.batch_decode(generated_ids[:1], skip_special_tokens=True)[0],
    )


def render_markdown(results: list[BenchmarkResult]) -> str:
    """渲染 Markdown 结果。"""
    lines = [
        "# Qwen2.5-VL Benchmark Report",
        "",
        f"- Model: `{MODEL_PATH}`",
        f"- Image: `{IMAGE_PATH}`",
        f"- Prompt: `{PROMPT}`",
        f"- Output tokens per sample: `{OUTPUT_TOKENS}`",
        f"- Batch sizes: `{list(BATCH_SIZES)}`",
        "",
        "| Engine | Batch Mode | Batch | Prompt/Sample | Prompt Total | Preprocess (s) | Vision Encode (s) | Prefill (s) | Prefill tok/s | Avg TTFT (s) | Decode (s) | Decode tok/s | Total (s) | Generated/Sample | Generated Total | Overall tok/s | Peak Allocated (GiB) | Peak Reserved (GiB) |",
        "|--------|------------|-------|---------------|--------------|----------------|-------------------|-------------|----------------|--------------|------------|---------------|-----------|------------------|-----------------|---------------|----------------------|---------------------|",
    ]
    for item in results:
        alloc_gib = item.cuda_max_memory_allocated_bytes / (1024 ** 3)
        reserve_gib = item.cuda_max_memory_reserved_bytes / (1024 ** 3)
        preprocess = f"{item.preprocess_seconds:.4f}" if item.preprocess_seconds is not None else "-"
        vision = f"{item.vision_encode_seconds:.4f}" if item.vision_encode_seconds is not None else "-"
        lines.append(
            f"| {item.engine} | {item.batch_mode} | {item.batch_size} | {item.prompt_tokens_per_sample} | {item.total_prompt_tokens} | {preprocess} | {vision} | {item.prefill_seconds:.4f} | {item.prefill_tokens_per_second:.2f} | {item.avg_ttft_seconds:.4f} | {item.decode_seconds:.4f} | {item.decode_tokens_per_second:.2f} | {item.elapsed_seconds:.4f} | {item.generated_tokens_per_sample} | {item.total_generated_tokens} | {item.overall_tokens_per_second:.2f} | {alloc_gib:.2f} | {reserve_gib:.2f} |"
        )
    lines.append("")
    lines.append("## Example Outputs")
    lines.append("")
    for item in results:
        if item.batch_size != 1:
            continue
        lines.append(f"### {item.engine} (batch={item.batch_size})")
        lines.append("")
        lines.append(item.text)
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    """执行 VLM benchmark。"""
    results: list[BenchmarkResult] = []

    clear_cuda_cache()
    minillm_engine = VLMEngine(MODEL_PATH, device="cuda")
    for batch_size in BATCH_SIZES:
        clear_cuda_cache()
        results.append(benchmark_minillm_batch(minillm_engine, batch_size))
    del minillm_engine
    clear_cuda_cache()

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)
    hf_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    for batch_size in BATCH_SIZES:
        clear_cuda_cache()
        results.append(benchmark_hf_batch(processor, hf_model, batch_size))
    del hf_model
    clear_cuda_cache()

    out_json = Path(OUT_JSON)
    out_md = Path(OUT_MD)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps([asdict(item) for item in results], indent=2, ensure_ascii=False))
    out_md.write_text(render_markdown(results), encoding="utf-8")
    print(out_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
