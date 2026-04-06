"""Qwen2.5-VL 与 HuggingFace 的多维度基准测试。"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import torch
from PIL import Image, ImageOps
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from minillm.engine.vlm_engine import VLMEngine
from minillm.sampling_params import SamplingParams

MODEL_PATH = "/app/models/Qwen2.5-VL-3B-Instruct"
IMAGE_PATH = "/app/minillm/img/image.png"
ALT_IMAGE_PATH = "/app/minillm/img/test_img.jpg"
OUTPUT_TOKENS = 32
BATCH_SIZES = (1, 4, 8, 16)
CASE_MODES = ("repeat", "diverse")
OUT_JSON = "/app/minillm/benchmark_results/qwen2_5_vlm_benchmark.json"
OUT_MD = "/app/minillm/benchmark_results/qwen2_5_vlm_benchmark.md"


@dataclass(frozen=True)
class BenchmarkCase:
    """单个多模态 benchmark 样本。"""

    case_id: str
    image_path: str
    prompt: str
    transform: str = "original"


@dataclass
class BenchmarkResult:
    """单个多模态基准测试结果。"""

    engine: str
    sample_mode: str
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


def _build_repeat_case() -> BenchmarkCase:
    """构造重复样本模板。"""
    return BenchmarkCase(
        case_id="repeat_lake",
        image_path=IMAGE_PATH,
        prompt="描述这张图片",
    )


def _build_diverse_cases() -> list[BenchmarkCase]:
    """构造多样化样本集合。"""
    return [
        BenchmarkCase("lake_desc", IMAGE_PATH, "描述这张图片", "original"),
        BenchmarkCase("lake_colors", IMAGE_PATH, "请概括画面中的主要景物和颜色", "center_crop"),
        BenchmarkCase("lake_foreground", IMAGE_PATH, "请只描述画面前景最显眼的内容", "flip_lr"),
        BenchmarkCase("lake_scene", IMAGE_PATH, "这张图片更像什么场景？请给出依据", "grayscale"),
        BenchmarkCase("lake_elements", IMAGE_PATH, "列出图中最显眼的三个元素", "rotate_90"),
        BenchmarkCase("lake_summary", IMAGE_PATH, "请用一句话总结画面主体", "resize_512"),
        BenchmarkCase("lake_nature", IMAGE_PATH, "请说明画面是否以自然景物为主", "center_crop_wide"),
        BenchmarkCase("lake_depth", IMAGE_PATH, "请分别描述近处和远处的内容", "flip_tb"),
        BenchmarkCase("alt_desc", ALT_IMAGE_PATH, "描述这张图片中的主体和背景", "original"),
        BenchmarkCase("alt_object", ALT_IMAGE_PATH, "请概括画面中的主要物体", "flip_lr"),
        BenchmarkCase("alt_indoor", ALT_IMAGE_PATH, "这张图片更像室内还是室外？请说明依据", "grayscale"),
        BenchmarkCase("alt_colors", ALT_IMAGE_PATH, "请总结这张图片的主要颜色和结构", "center_crop"),
        BenchmarkCase("alt_scene", ALT_IMAGE_PATH, "这张图片可能拍摄于什么场景？", "rotate_90"),
        BenchmarkCase("alt_presence", ALT_IMAGE_PATH, "请判断图中是否包含人物、建筑或自然景物", "resize_384"),
        BenchmarkCase("alt_summary", ALT_IMAGE_PATH, "请用一句话总结这张图片", "flip_tb"),
        BenchmarkCase("alt_layers", ALT_IMAGE_PATH, "请描述主体、背景和整体氛围", "center_crop_wide"),
    ]


repeat_case = _build_repeat_case()
diverse_cases = _build_diverse_cases()


def clear_cuda_cache() -> None:
    """清理 GPU 侧缓存。"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def apply_transform(image: Image.Image, transform: str) -> Image.Image:
    """应用基准测试图像变换。"""
    image = image.convert("RGB")
    width, height = image.size
    if transform == "original":
        return image
    if transform == "center_crop":
        crop_w = max(int(width * 0.7), 64)
        crop_h = max(int(height * 0.7), 64)
        left = max((width - crop_w) // 2, 0)
        top = max((height - crop_h) // 2, 0)
        return image.crop((left, top, left + crop_w, top + crop_h))
    if transform == "center_crop_wide":
        crop_w = max(int(width * 0.85), 64)
        crop_h = max(int(height * 0.55), 64)
        left = max((width - crop_w) // 2, 0)
        top = max((height - crop_h) // 2, 0)
        return image.crop((left, top, left + crop_w, top + crop_h))
    if transform == "flip_lr":
        return ImageOps.mirror(image)
    if transform == "flip_tb":
        return ImageOps.flip(image)
    if transform == "grayscale":
        return ImageOps.grayscale(image).convert("RGB")
    if transform == "rotate_90":
        return image.rotate(90, expand=True)
    if transform == "resize_512":
        return image.resize((512, 512))
    if transform == "resize_384":
        return image.resize((384, 384))
    raise ValueError(f"未知图像变换: {transform}")


def load_case_image(case: BenchmarkCase) -> Image.Image:
    """加载并变换单个 benchmark 图像。"""
    image = Image.open(case.image_path)
    return apply_transform(image, case.transform)


def build_cases(sample_mode: str, batch_size: int) -> list[BenchmarkCase]:
    """构造指定模式下的测试样本。"""
    if sample_mode == "repeat":
        return [repeat_case] * batch_size
    assert sample_mode == "diverse", f"未知样本模式: {sample_mode}"
    assert batch_size <= len(diverse_cases), "diverse benchmark 样本数量不足"
    return diverse_cases[:batch_size]


def build_messages(prompt: str) -> list[dict]:
    """构造单图问答消息模板。"""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
    ]


def benchmark_minillm_batch(engine: VLMEngine, batch_size: int, sample_mode: str) -> BenchmarkResult:
    """测试 minillm 在指定样本模式下的 VLM 原生 batch 表现。"""
    sampling_params = SamplingParams(
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        max_tokens=OUTPUT_TOKENS,
        ignore_eos=True,
    )
    cases = build_cases(sample_mode, batch_size)
    core = engine._engine
    timings = {"prepare": 0.0, "prefill": 0.0, "decode": 0.0}
    orig_prepare_batch = core._prepare_batch
    orig_prefill_sequence = core._prefill_sequence
    orig_prefill_sequences = core._prefill_sequences
    orig_run = core.model_runner.run

    def timed_prepare_batch(requests, sampling_params=None, apply_chat_template=True):
        start = perf_counter()
        out = orig_prepare_batch(requests, sampling_params, apply_chat_template)
        torch.cuda.synchronize()
        timings["prepare"] += perf_counter() - start
        return out

    def timed_prefill_sequence(seq):
        start = perf_counter()
        out = orig_prefill_sequence(seq)
        torch.cuda.synchronize()
        timings["prefill"] += perf_counter() - start
        return out

    def timed_prefill_sequences(seqs):
        start = perf_counter()
        out = orig_prefill_sequences(seqs)
        torch.cuda.synchronize()
        timings["prefill"] += perf_counter() - start
        return out

    def timed_run(seqs, is_prefill):
        start = perf_counter()
        out = orig_run(seqs, is_prefill)
        torch.cuda.synchronize()
        if not is_prefill:
            timings["decode"] += perf_counter() - start
        return out

    core._prepare_batch = timed_prepare_batch
    core._prefill_sequence = timed_prefill_sequence
    core._prefill_sequences = timed_prefill_sequences
    core.model_runner.run = timed_run

    try:
        requests = [{"text": case.prompt, "images": [load_case_image(case)]} for case in cases]
        start = perf_counter()
        outputs = engine.batch_generate(requests, sampling_params)
        torch.cuda.synchronize()
        elapsed_seconds = perf_counter() - start
        profile = engine.get_memory_profile()
    finally:
        core._prepare_batch = orig_prepare_batch
        core._prefill_sequence = orig_prefill_sequence
        core._prefill_sequences = orig_prefill_sequences
        core.model_runner.run = orig_run

    total_prompt_tokens = 0
    for case in cases:
        prompt_ids, _, _, _ = core._prepare_inputs(case.prompt, [load_case_image(case)], True)
        total_prompt_tokens += len(prompt_ids)
    prompt_tokens_per_sample = round(total_prompt_tokens / batch_size) if batch_size > 0 else 0
    total_generated_tokens = sum(len(item["token_ids"]) for item in outputs)
    generated_tokens_per_sample = total_generated_tokens // batch_size if batch_size > 0 else 0
    prefill_seconds = timings["prefill"]
    decode_seconds = timings["decode"]
    preprocess_seconds = timings["prepare"]
    prefill_tokens_per_second = total_prompt_tokens / prefill_seconds if prefill_seconds > 0 else 0.0
    decode_tokens_per_second = total_generated_tokens / decode_seconds if decode_seconds > 0 else 0.0
    overall_tokens_per_second = total_generated_tokens / elapsed_seconds if elapsed_seconds > 0 else 0.0

    return BenchmarkResult(
        engine="minillm_vlm",
        sample_mode=sample_mode,
        batch_mode="native",
        batch_size=batch_size,
        prompt_tokens_per_sample=prompt_tokens_per_sample,
        total_prompt_tokens=total_prompt_tokens,
        preprocess_seconds=preprocess_seconds,
        vision_encode_seconds=None,
        prefill_seconds=prefill_seconds,
        prefill_tokens_per_second=prefill_tokens_per_second,
        avg_ttft_seconds=prefill_seconds,
        decode_seconds=decode_seconds,
        elapsed_seconds=elapsed_seconds,
        generated_tokens_per_sample=generated_tokens_per_sample,
        total_generated_tokens=total_generated_tokens,
        overall_tokens_per_second=overall_tokens_per_second,
        decode_tokens_per_second=decode_tokens_per_second,
        model_bytes=profile.get("model_bytes"),
        kv_cache_total_bytes=profile.get("kv_cache_total_bytes"),
        cuda_max_memory_allocated_bytes=profile["cuda_max_memory_allocated_bytes"],
        cuda_max_memory_reserved_bytes=profile["cuda_max_memory_reserved_bytes"],
        text=outputs[0]["text"] if outputs else "",
    )


def benchmark_hf_batch(
    processor: AutoProcessor,
    model: Qwen2_5_VLForConditionalGeneration,
    batch_size: int,
    sample_mode: str,
) -> BenchmarkResult:
    """测试 HuggingFace 在指定样本模式下的 VLM 原生 batch 表现。"""
    cases = build_cases(sample_mode, batch_size)
    images = [load_case_image(case) for case in cases]
    texts = [processor.apply_chat_template(build_messages(case.prompt), tokenize=False, add_generation_prompt=True) for case in cases]

    preprocess_start = perf_counter()
    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt").to("cuda")
    torch.cuda.synchronize()
    preprocess_seconds = perf_counter() - preprocess_start
    total_prompt_tokens = int(inputs["attention_mask"].sum().item())
    prompt_tokens_per_sample = round(total_prompt_tokens / batch_size) if batch_size > 0 else 0

    with torch.inference_mode():
        prefill_start = perf_counter()
        _ = model(**inputs, use_cache=True)
        torch.cuda.synchronize()
        prefill_seconds = perf_counter() - prefill_start

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
    total_generated_tokens = int(generated_ids.numel())
    generated_tokens_per_sample = int(generated_ids.shape[1]) if generated_ids.ndim == 2 else 0
    decode_seconds = max(elapsed_seconds - prefill_seconds, 0.0)
    prefill_tokens_per_second = total_prompt_tokens / prefill_seconds if prefill_seconds > 0 else 0.0
    decode_tokens_per_second = total_generated_tokens / decode_seconds if decode_seconds > 0 else 0.0
    overall_tokens_per_second = total_generated_tokens / elapsed_seconds if elapsed_seconds > 0 else 0.0

    return BenchmarkResult(
        engine="huggingface_vlm",
        sample_mode=sample_mode,
        batch_mode="native",
        batch_size=batch_size,
        prompt_tokens_per_sample=prompt_tokens_per_sample,
        total_prompt_tokens=total_prompt_tokens,
        preprocess_seconds=preprocess_seconds,
        vision_encode_seconds=None,
        prefill_seconds=prefill_seconds,
        prefill_tokens_per_second=prefill_tokens_per_second,
        avg_ttft_seconds=prefill_seconds,
        decode_seconds=decode_seconds,
        elapsed_seconds=elapsed_seconds,
        generated_tokens_per_sample=generated_tokens_per_sample,
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
        f"- Images: `{IMAGE_PATH}`, `{ALT_IMAGE_PATH}`",
        f"- Case modes: `{list(CASE_MODES)}`",
        f"- Output tokens per sample: `{OUTPUT_TOKENS}`",
        f"- Batch sizes: `{list(BATCH_SIZES)}`",
        "",
        "| Engine | Sample Mode | Batch Mode | Batch | Avg Prompt/Sample | Prompt Total | Preprocess (s) | Prefill (s) | Prefill tok/s | Avg TTFT (s) | Decode (s) | Decode tok/s | Total (s) | Generated/Sample | Generated Total | Overall tok/s | Peak Allocated (GiB) | Peak Reserved (GiB) |",
        "|--------|-------------|------------|-------|-------------------|--------------|----------------|-------------|----------------|--------------|------------|---------------|-----------|------------------|-----------------|---------------|----------------------|---------------------|",
    ]
    for item in results:
        alloc_gib = item.cuda_max_memory_allocated_bytes / (1024 ** 3)
        reserve_gib = item.cuda_max_memory_reserved_bytes / (1024 ** 3)
        preprocess = f"{item.preprocess_seconds:.4f}" if item.preprocess_seconds is not None else "-"
        lines.append(
            f"| {item.engine} | {item.sample_mode} | {item.batch_mode} | {item.batch_size} | {item.prompt_tokens_per_sample} | {item.total_prompt_tokens} | {preprocess} | {item.prefill_seconds:.4f} | {item.prefill_tokens_per_second:.2f} | {item.avg_ttft_seconds:.4f} | {item.decode_seconds:.4f} | {item.decode_tokens_per_second:.2f} | {item.elapsed_seconds:.4f} | {item.generated_tokens_per_sample} | {item.total_generated_tokens} | {item.overall_tokens_per_second:.2f} | {alloc_gib:.2f} | {reserve_gib:.2f} |"
        )
    lines.append("")
    lines.append("## Example Outputs")
    lines.append("")
    for item in results:
        if item.batch_size != 1:
            continue
        lines.append(f"### {item.engine} / {item.sample_mode}")
        lines.append("")
        lines.append(item.text)
        lines.append("")
    return "\n".join(lines) + "\n"


def parse_int_list(raw: str) -> tuple[int, ...]:
    """解析逗号分隔的整数列表。"""
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def parse_str_list(raw: str) -> tuple[str, ...]:
    """解析逗号分隔的字符串列表。"""
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def main() -> None:
    """执行 VLM benchmark。"""
    parser = argparse.ArgumentParser(description="Qwen2.5-VL benchmark：minillm vs HuggingFace")
    parser.add_argument("--model", default=MODEL_PATH, help="模型路径")
    parser.add_argument("--image-path", default=IMAGE_PATH, help="主图路径")
    parser.add_argument("--alt-image-path", default=ALT_IMAGE_PATH, help="第二张图路径")
    parser.add_argument("--output-tokens", type=int, default=OUTPUT_TOKENS, help="每条样本生成 token 数")
    parser.add_argument("--batch-sizes", default=",".join(str(item) for item in BATCH_SIZES), help="逗号分隔的 batch 列表")
    parser.add_argument("--case-modes", default=",".join(CASE_MODES), help="逗号分隔的样本模式列表")
    parser.add_argument("--hf-attn-implementation", default="sdpa", help="HuggingFace 注意力实现")
    parser.add_argument("--output-json", default=OUT_JSON, help="JSON 输出路径")
    parser.add_argument("--output-md", default=OUT_MD, help="Markdown 输出路径")
    args = parser.parse_args()

    globals()["MODEL_PATH"] = args.model
    globals()["IMAGE_PATH"] = args.image_path
    globals()["ALT_IMAGE_PATH"] = args.alt_image_path
    globals()["OUTPUT_TOKENS"] = args.output_tokens
    globals()["BATCH_SIZES"] = parse_int_list(args.batch_sizes)
    globals()["CASE_MODES"] = parse_str_list(args.case_modes)
    globals()["OUT_JSON"] = args.output_json
    globals()["OUT_MD"] = args.output_md
    globals()["repeat_case"] = _build_repeat_case()
    globals()["diverse_cases"] = _build_diverse_cases()

    results: list[BenchmarkResult] = []

    clear_cuda_cache()
    minillm_engine = VLMEngine(MODEL_PATH, device="cuda", enforce_eager=True, cache_size_tokens=131072)
    for sample_mode in CASE_MODES:
        for batch_size in BATCH_SIZES:
            clear_cuda_cache()
            results.append(benchmark_minillm_batch(minillm_engine, batch_size, sample_mode))
    del minillm_engine
    clear_cuda_cache()

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    if hasattr(processor, "padding_side"):
        processor.padding_side = "left"
    hf_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation=args.hf_attn_implementation,
    )
    for sample_mode in CASE_MODES:
        for batch_size in BATCH_SIZES:
            clear_cuda_cache()
            result = benchmark_hf_batch(processor, hf_model, batch_size, sample_mode)
            if args.hf_attn_implementation != "sdpa":
                result.engine = f"huggingface_vlm_{args.hf_attn_implementation}"
            results.append(result)
    del hf_model
    clear_cuda_cache()

    out_json = Path(OUT_JSON)
    out_md = Path(OUT_MD)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps([asdict(item) for item in results], indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(render_markdown(results), encoding="utf-8")
    print(out_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
