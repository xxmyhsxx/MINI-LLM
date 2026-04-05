"""LLM Engine 测试。"""

from types import SimpleNamespace

import pytest

from minillm.engine.llm_engine import LLMEngine
from minillm.models.registry import list_registered
from minillm.sampling_params import SamplingParams


def test_registry_has_qwen2():
    """测试 Qwen2 模型已注册。"""
    registered = list_registered()
    assert "qwen2" in registered


def test_generate_rejects_mismatched_sampling_params():
    """测试 generate 会拒绝长度不匹配的采样参数列表。"""
    engine = object.__new__(LLMEngine)
    engine.add_request = lambda prompt, sp: 0
    engine.is_finished = lambda: True
    engine.step = lambda: ([], 0)
    engine.tokenizer = SimpleNamespace(decode=lambda token_ids: "")

    prompts = ["a", "b"]
    sampling_params = [SamplingParams(max_tokens=1)]

    with pytest.raises(AssertionError, match="sampling_params 数量必须与 prompts 数量一致"):
        engine.generate(prompts, sampling_params, use_tqdm=False)


def test_add_request_rejects_length_overflow():
    """测试请求长度超过 max_model_len 时会明确报错。"""
    engine = object.__new__(LLMEngine)
    engine.max_model_len = 16
    engine.tokenizer = SimpleNamespace(encode=lambda text: [1, 2, 3, 4])
    engine.scheduler = SimpleNamespace(add=lambda seq: None)

    with pytest.raises(AssertionError, match="请求长度超过 max_model_len"):
        engine.add_request("hello", SamplingParams(max_tokens=32))


def test_generate_stream_reports_metrics():
    """测试流式生成会输出 token 事件和 metrics 事件。"""
    engine = object.__new__(LLMEngine)
    engine._step_index = 0
    engine.tokenizer = SimpleNamespace(
        encode=lambda text: [1, 2, 3],
        decode=lambda token_ids, skip_special_tokens=False: "".join(str(token_id) for token_id in token_ids),
    )
    engine.add_request = lambda prompt, sp, apply_chat_template=True: 7
    engine.get_memory_profile = lambda: {
        "model_bytes": 1,
        "kv_cache_total_bytes": 2,
        "kv_cache_used_bytes": 1,
        "kv_cache_peak_used_bytes": 2,
        "kv_cache_total_blocks": 8,
        "kv_cache_used_blocks": 4,
        "kv_cache_peak_used_blocks": 8,
        "cuda_memory_allocated_bytes": 3,
        "cuda_max_memory_allocated_bytes": 4,
        "cuda_memory_reserved_bytes": 5,
        "cuda_max_memory_reserved_bytes": 6,
    }

    def step_with_details():
        engine._step_index += 1
        if engine._step_index == 1:
            return [], 3, [(7, 10)]
        return [(7, [10, 11])], -1, [(7, 11)]

    engine.step_with_details = step_with_details
    engine.is_finished = lambda: engine._step_index >= 2

    events = list(engine.generate_stream("hello", SamplingParams(max_tokens=2)))
    assert events[0]["type"] == "token"
    assert events[1]["type"] == "token"
    assert events[2]["type"] == "metrics"
    assert events[2]["generated_tokens"] == 2
    assert events[2]["prompt_tokens"] == 3
    assert events[2]["model_bytes"] == 1


def test_get_memory_profile_uses_block_manager_usage():
    """测试显存画像会带上当前与峰值 KV block 信息。"""
    engine = object.__new__(LLMEngine)
    engine._peak_kv_cache_used_blocks = 5
    engine.model_runner = SimpleNamespace(
        get_memory_profile=lambda used_blocks, peak_used_blocks: {
            "kv_cache_used_blocks": used_blocks,
            "kv_cache_peak_used_blocks": peak_used_blocks,
        }
    )
    engine.scheduler = SimpleNamespace(block_manager=SimpleNamespace(used_block_ids={1, 2, 3}))
    profile = engine.get_memory_profile()
    assert profile["kv_cache_used_blocks"] == 3
    assert profile["kv_cache_peak_used_blocks"] == 5
