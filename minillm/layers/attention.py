"""PageAttention 注意力层。

使用 AttentionBackend 抽象接口，支持通过配置切换注意力计算实现。
默认使用 FlashAttentionBackend。
Triton KV Cache 内核保留在本文件中。
"""

import torch
from torch import nn
import triton
import triton.language as tl

from minillm.layers.attention_backend import AttentionBackend
from minillm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    """Triton KV Cache 写入内核。

    将 key/value 写入 paged KV cache 的指定槽位。

    Args:
        key_ptr: key 张量指针
        key_stride: key 张量步长
        value_ptr: value 张量指针
        value_stride: value 张量步长
        k_cache_ptr: key cache 指针
        v_cache_ptr: value cache 指针
        slot_mapping_ptr: 槽位映射指针
        D: key/value 的总维度（num_heads * head_dim）
    """
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """将 key/value 写入 paged KV cache。

    Args:
        key: key 张量，形状 (N, num_heads, head_dim)
        value: value 张量，形状 (N, num_heads, head_dim)
        k_cache: key cache
        v_cache: value cache
        slot_mapping: 槽位映射
    """
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](
        key, key.stride(0), value, value.stride(0),
        k_cache, v_cache, slot_mapping, D,
    )


class Attention(nn.Module):
    """PageAttention 注意力层。

    通过 AttentionBackend 抽象接口委托注意力计算，支持通过配置切换后端。
    Triton KV Cache 内核保留在本模块中。

    Attributes:
        num_heads: 注意力头数
        head_dim: 每个头的维度
        scale: 缩放因子
        num_kv_heads: KV 头数（GQA 支持）
        backend: 注意力计算后端
        k_cache: key cache（由 ModelRunner 注入）
        v_cache: value cache（由 ModelRunner 注入）
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
        backend: AttentionBackend | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        # 默认使用 FlashAttentionBackend
        if backend is None:
            from minillm.layers.flash_attention_backend import FlashAttentionBackend
            backend = FlashAttentionBackend()
        self.backend = backend

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """注意力前向传播。

        Args:
            q: query 张量，形状 (N, num_heads, head_dim)
            k: key 张量，形状 (N, num_kv_heads, head_dim)
            v: value 张量，形状 (N, num_kv_heads, head_dim)

        Returns:
            注意力输出，形状 (N, num_heads, head_dim)
        """
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # 确保 k_cache 和 v_cache 与 q 的 dtype 一致
        if k_cache.numel() and k_cache.dtype != q.dtype:
            k_cache = k_cache.to(q.dtype)
            v_cache = v_cache.to(q.dtype)
            self.k_cache, self.v_cache = k_cache, v_cache

        # 只在有 KV cache 且有 slot_mapping 时才存储
        if k_cache.numel() and v_cache.numel() and context.slot_mapping is not None:
            self.backend.store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:
                # 前缀缓存模式
                k, v = k_cache, v_cache
            o = self.backend.prefill(q, k, v, context, self.scale, k_cache, v_cache)
        else:
            o = self.backend.decode(q, k, v, context, self.scale, k_cache, v_cache)
        return o
