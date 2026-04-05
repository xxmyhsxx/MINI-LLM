"""注意力后端抽象层测试（需要 GPU）。

测试 AttentionBackend 抽象接口、FlashAttentionBackend 实现、
以及 Attention 类接受自定义后端的能力。

运行方式：
    pytest -v minillm/tests/test_attention_backend.py

依赖：需要 GPU。
"""

import pytest
import torch


@pytest.mark.gpu
def test_attention_default_backend_is_flash():
    """测试 Attention 不传 backend 时默认使用 FlashAttentionBackend。"""
    from minillm.layers.attention import Attention
    from minillm.layers.flash_attention_backend import FlashAttentionBackend

    attn = Attention(2, 16, 0.125, 2)
    assert isinstance(attn.backend, FlashAttentionBackend)
    del attn


@pytest.mark.gpu
def test_attention_accepts_custom_backend():
    """测试 Attention 类接受自定义 backend。"""
    from minillm.layers.attention import Attention
    from minillm.layers.attention_backend import AttentionBackend

    class DummyBackend(AttentionBackend):
        """测试用后端，返回全零。"""

        def prefill(self, q, k, v, context, scale, k_cache, v_cache):
            return torch.zeros_like(q)

        def decode(self, q, k, v, context, scale, k_cache, v_cache):
            return torch.zeros_like(q)

    attn = Attention(2, 16, 0.125, 2, backend=DummyBackend())
    assert isinstance(attn.backend, DummyBackend)
    del attn


@pytest.mark.gpu
def test_flash_attention_backend_store_kvcache():
    """测试 FlashAttentionBackend 的 store_kvcache 正确调用 Triton kernel。"""
    from minillm.layers.flash_attention_backend import FlashAttentionBackend

    backend = FlashAttentionBackend()
    N, num_heads, head_dim = 4, 2, 16
    D = num_heads * head_dim
    key = torch.randn(N, num_heads, head_dim, device="cuda")
    value = torch.randn(N, num_heads, head_dim, device="cuda")
    # KV cache 使用与 ModelRunner 一致的形状 (num_blocks, block_size, num_heads, head_dim)
    num_blocks, block_size = 10, 16
    k_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim, device="cuda")
    v_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim, device="cuda")
    slot_mapping = torch.arange(N, device="cuda")

    backend.store_kvcache(key, value, k_cache, v_cache, slot_mapping)

    # 验证数据被写入指定槽位
    for i in range(N):
        slot = slot_mapping[i].item()
        block_id = slot // block_size
        block_offset = slot % block_size
        expected_k = key[i]
        actual_k = k_cache[block_id, block_offset]
        assert torch.allclose(expected_k, actual_k), f"slot {slot} key 不一致"
    del backend, key, value, k_cache, v_cache
