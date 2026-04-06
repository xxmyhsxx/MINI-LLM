"""常驻推理服务测试。"""

import json
import threading
import urllib.request


def open_local_request(request: urllib.request.Request):
    """绕过环境代理，直接请求本地测试服务。"""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(request)

from http.server import ThreadingHTTPServer

import pytest

from minillm.sampling_params import SamplingParams
from minillm.server import InferenceService, build_parser, build_sampling_params_from_payload, create_handler, normalize_prompt, sanitize_metrics


class FakeLLM:
    """用于服务测试的伪造引擎。"""

    def __init__(self):
        self.reset_calls = 0

    def reset_peak_memory_stats(self) -> None:
        """记录峰值显存重置次数。"""
        self.reset_calls += 1

    def generate_stream(self, prompt, sampling_params, apply_chat_template=True):
        """返回固定的流式事件。"""
        assert prompt == "hello"
        assert sampling_params.max_tokens == 2
        yield {"type": "token", "seq_id": 1, "token_id": 10, "text": "A", "generated_tokens": 1}
        yield {
            "type": "metrics",
            "seq_id": 1,
            "prompt_tokens": 3,
            "generated_tokens": 2,
            "ttft_seconds": 0.1,
            "total_time_seconds": 0.2,
            "overall_tokens_per_second": 10.0,
            "decode_tokens_per_second": 20.0,
            "text": "AB",
            "token_ids": [10, 11],
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
        }

    def get_memory_profile(self) -> dict:
        """返回固定显存画像。"""
        return {
            "model_bytes": 1024,
            "kv_cache_total_bytes": 2048,
            "kv_cache_used_bytes": 512,
            "kv_cache_peak_used_bytes": 1024,
            "kv_cache_total_blocks": 8,
            "kv_cache_used_blocks": 2,
            "kv_cache_peak_used_blocks": 4,
            "cuda_memory_allocated_bytes": 4096,
            "cuda_max_memory_allocated_bytes": 8192,
            "cuda_memory_reserved_bytes": 16384,
            "cuda_max_memory_reserved_bytes": 32768,
        }


def test_build_parser_accepts_service_flags():
    """测试服务解析器支持 host 与 port 参数。"""
    parser = build_parser()
    args = parser.parse_args(["--host", "127.0.0.1", "--port", "9000"])
    assert args.host == "127.0.0.1"
    assert args.port == 9000


def test_build_sampling_params_from_nested_payload():
    """测试服务请求体支持嵌套 sampling_params。"""
    sampling_params = build_sampling_params_from_payload({
        "prompt": "hello",
        "sampling_params": {
            "temperature": 0.7,
            "top_k": 8,
            "top_p": 0.9,
            "max_new_tokens": 16,
            "ignore_eos": True,
        },
    })
    assert sampling_params.temperature == 0.7
    assert sampling_params.top_k == 8
    assert sampling_params.top_p == 0.9
    assert sampling_params.max_tokens == 16
    assert sampling_params.ignore_eos is True


def test_normalize_prompt_rejects_non_int_tokens():
    """测试 token prompt 必须全部为 int。"""
    with pytest.raises(AssertionError, match="必须全部为 int"):
        normalize_prompt([1, "2"])


def test_sanitize_metrics_can_include_profile_and_token_ids():
    """测试最终响应可选择性包含 token 与精简 profile。"""
    payload = sanitize_metrics({
        "type": "metrics",
        "seq_id": 1,
        "text": "AB",
        "token_ids": [10, 11],
        "prompt_tokens": 3,
        "generated_tokens": 2,
        "ttft_seconds": 0.1,
        "total_time_seconds": 0.2,
        "overall_tokens_per_second": 10.0,
        "decode_tokens_per_second": 20.0,
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
    }, True, True)
    assert payload["text"] == "AB"
    assert payload["token_ids"] == [10, 11]
    assert payload["metrics"]["kv_cache_peak_gib"] > 0
    assert payload["metrics"]["kv_cache_peak_blocks"] == 6
    assert "model_bytes" not in payload["metrics"]


def test_inference_service_generate_resets_peak_stats():
    """测试服务在生成前会重置峰值显存统计。"""
    llm = FakeLLM()
    service = InferenceService(llm, {"model": "fake", "max_model_len": 512, "max_num_seqs": 1, "cache_size_mb": None, "cache_size_tokens": 1024, "gpu_memory_utilization": 0.9, "enforce_eager": False})
    metrics = service.generate("hello", SamplingParams(max_tokens=2))
    assert metrics["text"] == "AB"
    assert llm.reset_calls == 1


def test_inference_service_memory_includes_gib_fields():
    """测试显存画像接口会补充 GiB 字段。"""
    llm = FakeLLM()
    service = InferenceService(llm, {"model": "fake", "max_model_len": 512, "max_num_seqs": 1, "cache_size_mb": None, "cache_size_tokens": 1024, "gpu_memory_utilization": 0.9, "enforce_eager": False})
    profile = service.memory()
    assert profile["kv_cache_used_gib"] > 0
    assert profile["kv_cache_peak_gib"] > 0


def test_http_generate_endpoint_returns_json_profile():
    """测试 HTTP 非流式接口返回 JSON 结果。"""
    service = InferenceService(FakeLLM(), {"model": "fake", "max_model_len": 512, "max_num_seqs": 1, "cache_size_mb": None, "cache_size_tokens": 1024, "gpu_memory_utilization": 0.9, "enforce_eager": False})
    server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(service))
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/v1/generate"
        request = urllib.request.Request(
            url,
            data=json.dumps({"prompt": "hello", "max_tokens": 2, "profile": True, "return_token_ids": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with open_local_request(request) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["text"] == "AB"
        assert payload["token_ids"] == [10, 11]
        assert payload["metrics"]["generated_tokens"] == 2
        assert "model_bytes" not in payload["metrics"]
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_http_generate_stream_endpoint_emits_sse():
    """测试 HTTP 流式接口返回 SSE 事件。"""
    service = InferenceService(FakeLLM(), {"model": "fake", "max_model_len": 512, "max_num_seqs": 1, "cache_size_mb": None, "cache_size_tokens": 1024, "gpu_memory_utilization": 0.9, "enforce_eager": False})
    server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(service))
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/v1/generate_stream"
        request = urllib.request.Request(
            url,
            data=json.dumps({"prompt": "hello", "max_tokens": 2, "profile": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with open_local_request(request) as response:
            body = response.read().decode("utf-8")
        assert "event: token" in body
        assert "event: metrics" in body
        assert '"text": "AB"' in body
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
