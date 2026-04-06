"""命令行推理入口。"""

import argparse
import json

from transformers import AutoTokenizer

from minillm.engine.llm_engine import LLM
from minillm.sampling_params import SamplingParams


def bytes_to_gib(num_bytes: int) -> float:
    """将字节转换为 GiB。"""
    return num_bytes / (1024 ** 3)


def summarize_request_metrics(metrics: dict) -> dict:
    """将单次请求 metrics 压缩为精简摘要。"""
    return {
        "ttft_seconds": metrics["ttft_seconds"],
        "total_time_seconds": metrics["total_time_seconds"],
        "prompt_tokens": metrics["prompt_tokens"],
        "generated_tokens": metrics["generated_tokens"],
        "overall_tokens_per_second": metrics["overall_tokens_per_second"],
        "decode_tokens_per_second": metrics["decode_tokens_per_second"],
        "model_gib": bytes_to_gib(metrics["model_bytes"]),
        "kv_cache_total_gib": bytes_to_gib(metrics["kv_cache_total_bytes"]),
        "kv_cache_peak_gib": bytes_to_gib(metrics["kv_cache_peak_used_bytes"]),
        "kv_cache_peak_blocks": metrics["kv_cache_peak_used_blocks"],
        "kv_cache_total_blocks": metrics["kv_cache_total_blocks"],
        "cuda_peak_allocated_gib": bytes_to_gib(metrics["cuda_max_memory_allocated_bytes"]),
        "cuda_peak_reserved_gib": bytes_to_gib(metrics["cuda_max_memory_reserved_bytes"]),
    }


def summarize_live_memory_profile(profile: dict) -> dict:
    """将实时显存画像压缩为精简摘要。"""
    return {
        "model_gib": bytes_to_gib(profile["model_bytes"]),
        "kv_cache_total_gib": bytes_to_gib(profile["kv_cache_total_bytes"]),
        "kv_cache_used_gib": bytes_to_gib(profile["kv_cache_used_bytes"]),
        "kv_cache_peak_gib": bytes_to_gib(profile["kv_cache_peak_used_bytes"]),
        "kv_cache_used_blocks": profile["kv_cache_used_blocks"],
        "kv_cache_peak_blocks": profile["kv_cache_peak_used_blocks"],
        "kv_cache_total_blocks": profile["kv_cache_total_blocks"],
        "cuda_allocated_gib": bytes_to_gib(profile["cuda_memory_allocated_bytes"]),
        "cuda_peak_allocated_gib": bytes_to_gib(profile["cuda_max_memory_allocated_bytes"]),
        "cuda_reserved_gib": bytes_to_gib(profile["cuda_memory_reserved_bytes"]),
        "cuda_peak_reserved_gib": bytes_to_gib(profile["cuda_max_memory_reserved_bytes"]),
    }


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。

    Returns:
        参数解析器
    """
    parser = argparse.ArgumentParser(description="运行 Qwen2.5 单卡推理")
    parser.add_argument("--model", default="/app/models/qwen2.5-1.5B-Instruct", help="模型目录")
    parser.add_argument("--prompt", default=None, help="输入文本 prompt（单条推理）")
    parser.add_argument("--prompts-file", default=None, help="批量 prompt 文件（每行一条 prompt，或 JSON 数组）")
    parser.add_argument("--max-tokens", "--max-new-tokens", dest="max_tokens", type=int, default=64, help="最大生成新 token 数")
    parser.add_argument("--temperature", type=float, default=0.0, help="温度，0 表示 greedy")
    parser.add_argument("--top-k", type=int, default=0, help="top-k，0 表示关闭")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p，1.0 表示关闭")
    parser.add_argument("--ignore-eos", action="store_true", help="是否忽略 EOS")
    parser.add_argument("--max-num-seqs", type=int, default=1, help="最大并发序列数")
    parser.add_argument("--max-model-len", type=int, default=None, help="最大上下文长度，默认自动按 prompt+max_tokens 推导")
    parser.add_argument("--cache-size-mb", type=int, default=None, help="按 MB 指定 KV Cache 大小")
    parser.add_argument("--cache-size-tokens", type=int, default=None, help="按 token 容量指定 KV Cache 大小")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="自动分配 KV Cache 时的显存利用率")
    parser.add_argument("--enforce-eager", action="store_true", help="禁用 decode CUDA Graph")
    parser.add_argument("--print-token-ids", action="store_true", help="额外输出生成 token id")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    parser.add_argument("--no-chat-template", action="store_true", help="禁用自动对话模板拼接")
    parser.add_argument("--no-tqdm", action="store_true", help="关闭进度条")
    parser.add_argument("--stream", action="store_true", help="启用流式输出")
    parser.add_argument("--profile", action="store_true", help="输出性能分析信息（TTFT、速度、显存）")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """校验命令行参数。

    Args:
        args: 解析后的参数
    """
    # 仅当参数中包含 prompt 字段时才校验（CLI 需要，Server 不需要）
    if hasattr(args, 'prompt'):
        assert args.prompt is not None or args.prompts_file is not None, (
            "--prompt 和 --prompts-file 必须指定其中一个"
        )
        assert not (args.prompt is not None and args.prompts_file is not None), (
            "--prompt 和 --prompts-file 不能同时指定"
        )
    assert not (args.cache_size_mb is not None and args.cache_size_tokens is not None), (
        "cache_size_mb 与 cache_size_tokens 只能二选一"
    )


def load_prompts_from_file(path: str) -> list[str]:
    """从文件加载批量 prompt。

    支持两种格式：
    1. 每行一条 prompt（纯文本文件）
    2. JSON 数组 ["prompt1", "prompt2", ...]

    Args:
        path: 文件路径

    Returns:
        prompt 列表
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    # 尝试解析为 JSON 数组
    if content.startswith("["):
        prompts = json.loads(content)
        assert isinstance(prompts, list), "JSON 格式必须是数组"
        assert all(isinstance(p, str) for p in prompts), "JSON 数组元素必须是字符串"
        return prompts
    # 否则按每行一条 prompt 处理
    prompts = [line.strip() for line in content.split("\n") if line.strip()]
    assert len(prompts) > 0, "prompts 文件不能为空"
    return prompts


def build_engine_kwargs(args: argparse.Namespace) -> dict:
    """构建引擎初始化参数。

    Args:
        args: 解析后的参数

    Returns:
        传给 LLM 的关键字参数
    """
    kwargs = {
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
    }
    if args.max_model_len is not None:
        kwargs["max_model_len"] = args.max_model_len
    if args.cache_size_mb is not None:
        kwargs["cache_size_mb"] = args.cache_size_mb
    if args.cache_size_tokens is not None:
        kwargs["cache_size_tokens"] = args.cache_size_tokens
    return kwargs


def build_sampling_params(args: argparse.Namespace) -> SamplingParams:
    """构建采样参数。

    Args:
        args: 解析后的参数

    Returns:
        采样参数对象
    """
    return SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        ignore_eos=args.ignore_eos,
    )


def resolve_max_model_len(args: argparse.Namespace, tokenizer: AutoTokenizer) -> int:
    """推导最终使用的 max_model_len。

    Args:
        args: 解析后的参数
        tokenizer: 分词器

    Returns:
        最终上下文长度
    """
    if args.max_model_len is not None:
        return args.max_model_len
    if args.prompt is not None:
        prompt_len = len(tokenizer.encode(args.prompt))
        return prompt_len + args.max_tokens + 64
    # 批量模式：计算最长 prompt
    prompts = load_prompts_from_file(args.prompts_file)
    max_prompt_len = max(len(tokenizer.encode(p)) for p in prompts)
    return max_prompt_len + args.max_tokens + 64


def format_metrics(metrics: dict) -> str:
    """格式化性能与显存指标。

    Args:
        metrics: 性能指标事件

    Returns:
        可读字符串
    """
    summary = summarize_request_metrics(metrics)
    lines = [
        f"Latency: TTFT {summary['ttft_seconds']:.4f}s | Total {summary['total_time_seconds']:.4f}s",
        f"Tokens: Prompt {summary['prompt_tokens']} | Generated {summary['generated_tokens']}",
        (
            f"Throughput: Overall {summary['overall_tokens_per_second']:.2f} tok/s | "
            f"Decode {summary['decode_tokens_per_second']:.2f} tok/s"
        ),
        (
            f"Memory: Model {summary['model_gib']:.2f} GiB | "
            f"KV Peak {summary['kv_cache_peak_gib']:.2f}/{summary['kv_cache_total_gib']:.2f} GiB "
            f"({summary['kv_cache_peak_blocks']}/{summary['kv_cache_total_blocks']} blocks) | "
            f"CUDA Peak {summary['cuda_peak_allocated_gib']:.2f}/{summary['cuda_peak_reserved_gib']:.2f} GiB "
            f"(alloc/reserved)"
        ),
    ]
    return "\n".join(lines)


def run_stream(llm: LLM, args: argparse.Namespace) -> None:
    """执行流式推理。

    Args:
        llm: 推理引擎
        args: 命令行参数
    """
    llm.reset_peak_memory_stats()
    metrics = None
    streamed_any_token = False
    for event in llm.generate_stream(args.prompt, build_sampling_params(args)):
        if event["type"] == "token":
            if args.stream:
                streamed_any_token = True
                if args.json:
                    print(json.dumps(event, ensure_ascii=False), flush=True)
                else:
                    print(event["text"], end="", flush=True)
            continue
        metrics = event

    assert metrics is not None, "流式推理必须返回 metrics 事件"

    if args.stream and not args.json and streamed_any_token:
        print()

    if args.json:
        payload = metrics.copy()
        if not args.print_token_ids:
            payload.pop("token_ids", None)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if not args.stream:
        print(metrics["text"])
        if args.print_token_ids:
            print(metrics["token_ids"])

    if args.profile:
        print(format_metrics(metrics))


def main() -> None:
    """执行命令行推理。"""
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    args.max_model_len = resolve_max_model_len(args, tokenizer)
    llm = LLM(args.model, **build_engine_kwargs(args))
    sp = build_sampling_params(args)

    # 批量推理模式
    if args.prompts_file is not None:
        prompts = load_prompts_from_file(args.prompts_file)
        outputs = llm.generate(prompts, sp, use_tqdm=not args.no_tqdm, apply_chat_template=not args.no_chat_template)
        for i, output in enumerate(outputs):
            if args.json:
                payload = {"index": i, "text": output["text"]}
                if args.print_token_ids:
                    payload["token_ids"] = output["token_ids"]
                print(json.dumps(payload, ensure_ascii=False))
            else:
                print(f"[{i}] {output['text']}")
                if args.print_token_ids:
                    print(f"    token_ids: {output['token_ids']}")
        return

    # 单条推理模式
    if args.stream or args.profile:
        run_stream(llm, args)
        return

    outputs = llm.generate([args.prompt], sp, use_tqdm=not args.no_tqdm, apply_chat_template=not args.no_chat_template)
    output = outputs[0]

    if args.json:
        payload = {"text": output["text"]}
        if args.print_token_ids:
            payload["token_ids"] = output["token_ids"]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(output["text"])
    if args.print_token_ids:
        print(output["token_ids"])


if __name__ == "__main__":
    main()
