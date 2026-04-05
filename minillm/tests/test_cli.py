"""命令行入口测试。"""

import pytest
from types import SimpleNamespace

from minillm.cli import bytes_to_gib, build_engine_kwargs, build_parser, build_sampling_params, format_metrics, resolve_max_model_len, validate_args


def test_validate_args_rejects_both_kv_sizes():
    """测试 KV Cache 两种大小参数不能同时指定。"""
    parser = build_parser()
    args = parser.parse_args([
        "--prompt", "hello",
        "--cache-size-mb", "512",
        "--cache-size-tokens", "4096",
    ])
    with pytest.raises(AssertionError, match="只能二选一"):
        validate_args(args)


def test_build_engine_kwargs_prefers_explicit_kv_setting():
    """测试引擎参数会正确带上 KV Cache 配置。"""
    parser = build_parser()
    args = parser.parse_args([
        "--prompt", "hello",
        "--cache-size-tokens", "4096",
        "--max-model-len", "1024",
        "--max-num-seqs", "2",
    ])
    kwargs = build_engine_kwargs(args)
    assert kwargs["cache_size_tokens"] == 4096
    assert kwargs["max_model_len"] == 1024
    assert kwargs["max_num_seqs"] == 2


def test_build_sampling_params_from_args():
    """测试命令行参数会正确转换为采样参数。"""
    parser = build_parser()
    args = parser.parse_args([
        "--prompt", "hello",
        "--temperature", "0.7",
        "--top-k", "20",
        "--top-p", "0.9",
        "--max-new-tokens", "16",
        "--ignore-eos",
    ])
    sampling_params = build_sampling_params(args)
    assert sampling_params.temperature == 0.7
    assert sampling_params.top_k == 20
    assert sampling_params.top_p == 0.9
    assert sampling_params.max_tokens == 16
    assert sampling_params.ignore_eos is True


def test_resolve_max_model_len_auto_expands_for_generation():
    """测试会根据 prompt 和 max_tokens 自动推导 max_model_len。"""
    parser = build_parser()
    args = parser.parse_args(["--prompt", "hello", "--max-tokens", "1024"])
    tokenizer = SimpleNamespace(encode=lambda text: [1, 2, 3, 4])
    assert resolve_max_model_len(args, tokenizer) == 1092  # 4 + 1024 + 64


def test_parser_accepts_stream_and_profile_flags():
    """测试解析器支持流式和性能分析参数。"""
    parser = build_parser()
    args = parser.parse_args(["--prompt", "hello", "--stream", "--profile"])
    assert args.stream is True
    assert args.profile is True


def test_format_metrics_contains_memory_breakdown():
    """测试性能指标格式化输出包含显存拆分。"""
    text = format_metrics({
        "ttft_seconds": 0.12,
        "total_time_seconds": 0.56,
        "prompt_tokens": 8,
        "generated_tokens": 16,
        "overall_tokens_per_second": 28.5,
        "decode_tokens_per_second": 40.2,
        "model_bytes": 1024,
        "kv_cache_total_bytes": 2048,
        "kv_cache_used_bytes": 1024,
        "kv_cache_peak_used_bytes": 1536,
        "kv_cache_total_blocks": 8,
        "kv_cache_used_blocks": 4,
        "kv_cache_peak_used_blocks": 6,
        "cuda_memory_allocated_bytes": 4096,
        "cuda_max_memory_allocated_bytes": 8192,
        "cuda_memory_reserved_bytes": 16384,
        "cuda_max_memory_reserved_bytes": 32768,
    })
    assert "TTFT" in text
    assert "Model Memory" in text
    assert "KV Cache Total" in text
    assert "KV Cache Used Current" in text
    assert "KV Cache Used Peak" in text
    assert "CUDA Peak Reserved" in text


def test_bytes_to_gib():
    """测试字节到 GiB 转换。"""
    assert bytes_to_gib(1024 ** 3) == 1.0
