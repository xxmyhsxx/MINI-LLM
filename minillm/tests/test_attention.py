"""Attention 层单元测试（需要 GPU）。"""

import pytest
import torch

from minillm.layers.attention import store_kvcache
from minillm.layers.activation import SiluAndMul
from minillm.layers.layernorm import RMSNorm
from minillm.layers.sampler import Sampler
from minillm.layers.rotary_embedding import RotaryEmbedding


@pytest.mark.gpu
def test_store_kvcache():
    """测试 Triton KV Cache 写入内核。"""
    N, num_heads, head_dim = 4, 2, 64
    D = num_heads * head_dim
    block_size = 8
    num_blocks = 4

    key = torch.randn(N, num_heads, head_dim, device="cuda").contiguous()
    value = torch.randn(N, num_heads, head_dim, device="cuda").contiguous()
    k_cache = torch.zeros(num_blocks, block_size, D, device="cuda")
    v_cache = torch.zeros(num_blocks, block_size, D, device="cuda")
    slot_mapping = torch.arange(N, device="cuda", dtype=torch.int32)

    store_kvcache(key, value, k_cache, v_cache, slot_mapping)

    # 验证 KV Cache 中的值
    for i in range(N):
        slot = slot_mapping[i].item()
        block_idx = slot // block_size
        slot_idx = slot % block_size
        cached_k = k_cache[block_idx, slot_idx].view(num_heads, head_dim)
        assert torch.allclose(cached_k, key[i], atol=1e-5)


@pytest.mark.gpu
def test_silu_and_mul():
    """测试 SiLU + 门控乘法。"""
    model = SiluAndMul()
    x = torch.randn(4, 256, device="cuda")  # intermediate_size = 128
    out = model(x)
    assert out.shape == (4, 128)


@pytest.mark.gpu
def test_rms_norm():
    """测试 RMSNorm。"""
    model = RMSNorm(128)
    model = model.cuda()
    x = torch.randn(4, 128, device="cuda")
    out = model(x)
    assert out.shape == (4, 128)


@pytest.mark.gpu
def test_rms_norm_with_residual():
    """测试 RMSNorm + 残差。"""
    model = RMSNorm(128)
    model = model.cuda()
    x = torch.randn(4, 128, device="cuda")
    residual = torch.randn(4, 128, device="cuda")
    out, new_residual = model(x, residual)
    assert out.shape == (4, 128)
    assert new_residual.shape == (4, 128)


@pytest.mark.gpu
def test_sampler():
    """测试采样器（temperature=1.0, 无 top-k/top-p）。"""
    sampler = Sampler()
    logits = torch.randn(4, 1000, device="cuda")
    temperatures = torch.ones(4, device="cuda")
    top_ks = torch.zeros(4, dtype=torch.int32, device="cuda")
    top_ps = torch.ones(4, device="cuda")
    tokens = sampler(logits, temperatures, top_ks, top_ps)
    assert tokens.shape == (4,)
    assert (tokens >= 0).all()
    assert (tokens < 1000).all()


@pytest.mark.gpu
def test_sampler_greedy():
    """测试 greedy 采样（temperature=0）。"""
    sampler = Sampler()
    # 构造 logits 使得第 0 个样本的最大值在索引 42
    logits = torch.randn(2, 100, device="cuda")
    logits[0, 42] = 100.0
    logits[1, 77] = 100.0
    temperatures = torch.zeros(2, device="cuda")  # greedy
    top_ks = torch.zeros(2, dtype=torch.int32, device="cuda")
    top_ps = torch.ones(2, device="cuda")
    tokens = sampler(logits, temperatures, top_ks, top_ps)
    assert tokens[0].item() == 42
    assert tokens[1].item() == 77


@pytest.mark.gpu
def test_sampler_top_k():
    """测试 top-k 采样。"""
    sampler = Sampler()
    logits = torch.zeros(1, 10, device="cuda")
    logits[0, 0] = 10.0  # 最高
    logits[0, 1] = 5.0   # 第二
    logits[0, 2] = 3.0   # 第三
    # 其余为 0
    temperatures = torch.zeros(1, device="cuda")  # greedy
    top_ks = torch.tensor([2], dtype=torch.int32, device="cuda")
    top_ps = torch.ones(1, device="cuda")
    tokens = sampler(logits, temperatures, top_ks, top_ps)
    # greedy + top_k=2 应该从 top 2 中选最大值
    assert tokens[0].item() == 0  # 最高分


@pytest.mark.gpu
def test_sampler_top_p():
    """测试 top-p 采样。"""
    sampler = Sampler()
    logits = torch.zeros(1, 10, device="cuda")
    logits[0, 0] = 10.0  # p ≈ 0.999
    logits[0, 1] = 0.0   # p ≈ 0.0001
    temperatures = torch.zeros(1, device="cuda")  # greedy
    top_ks = torch.zeros(1, dtype=torch.int32, device="cuda")
    top_ps = torch.tensor([0.5], device="cuda")  # 只保留 p ≥ 0.5 的 token
    tokens = sampler(logits, temperatures, top_ks, top_ps)
    assert tokens[0].item() == 0  # 只有 token 0 满足 top_p


@pytest.mark.gpu
def test_attention_prefill_varlen_isolation():
    """测试 prefill 阶段不会发生批内串扰。"""
    from flash_attn import flash_attn_varlen_func

    from minillm.layers.attention import Attention
    from minillm.utils.context import reset_context, set_context

    num_heads = 2
    head_dim = 16
    num_kv_heads = 2
    q = torch.randn(5, num_heads, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(5, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(5, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32, device="cuda")

    attn = Attention(num_heads, head_dim, head_dim ** -0.5, num_kv_heads).cuda()
    set_context(
        True,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=3,
        max_seqlen_k=3,
        slot_mapping=torch.empty(0, dtype=torch.int32, device="cuda"),
    )
    out = attn(q, k, v)
    expected = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=3,
        max_seqlen_k=3,
        softmax_scale=head_dim ** -0.5,
        causal=True,
    )
    reset_context()

    assert torch.allclose(out, expected, atol=1e-3, rtol=1e-3)

@pytest.mark.gpu
def test_rotary_embedding():
    """测试旋转位置编码。"""
    head_size = 64
    max_pos = 2048
    rope = RotaryEmbedding(head_size, head_size, max_pos, base=10000.0)
    rope = rope.cuda()

    positions = torch.arange(4, device="cuda")
    q = torch.randn(4, 1, head_size, device="cuda")
    k = torch.randn(4, 1, head_size, device="cuda")
    q_out, k_out = rope(positions, q, k)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


@pytest.mark.gpu
def test_rotary_embedding_preserves_norm():
    """测试 RoPE 保持向量范数。"""
    head_size = 64
    rope = RotaryEmbedding(head_size, head_size, 2048, base=10000.0)
    rope = rope.cuda()

    positions = torch.zeros(1, device="cuda", dtype=torch.long)
    q = torch.randn(1, 1, head_size, device="cuda")
    k = torch.randn(1, 1, head_size, device="cuda")
    q_norm_before = q.norm()
    k_norm_before = k.norm()
    q_out, k_out = rope(positions, q, k)
    # RoPE 是旋转操作，范数应保持不变（近似）
    assert torch.allclose(q_out.norm(), q_norm_before, atol=1e-4)
    assert torch.allclose(k_out.norm(), k_norm_before, atol=1e-4)
