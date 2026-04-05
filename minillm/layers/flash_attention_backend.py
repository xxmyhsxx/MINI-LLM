"""Flash Attention 后端实现。

使用 flash-attn 库的 varlen 和 kvcache 函数。
KV Cache 写入使用 Triton kernel。
"""

import torch
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from minillm.layers.attention_backend import AttentionBackend
from minillm.layers.attention import store_kvcache


class FlashAttentionBackend(AttentionBackend):
    """基于 Flash Attention 的注意力后端。

    使用 flash-attn 库实现高效注意力计算。
    """

    def store_kvcache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """使用 Triton kernel 写入 KV Cache。

        Args:
            key: key 张量
            value: value 张量
            k_cache: key cache
            v_cache: value cache
            slot_mapping: 槽位映射
        """
        store_kvcache(key, value, k_cache, v_cache, slot_mapping)

    def prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context,
        scale: float,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill 阶段注意力计算（变长序列）。

        Args:
            q: query 张量
            k: key 张量
            v: value 张量
            context: 推理上下文
            scale: softmax 缩放因子
            k_cache: key cache（前缀缓存模式下使用）
            v_cache: value cache（前缀缓存模式下使用）

        Returns:
            注意力输出
        """
        if context.block_tables is not None:
            # 前缀缓存模式：从 cache 中读取完整序列
            k, v = k_cache, v_cache
        return flash_attn_varlen_func(
            q, k, v,
            max_seqlen_q=context.max_seqlen_q,
            cu_seqlens_q=context.cu_seqlens_q,
            max_seqlen_k=context.max_seqlen_k,
            cu_seqlens_k=context.cu_seqlens_k,
            softmax_scale=scale,
            causal=True,
            block_table=context.block_tables,
        )

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context,
        scale: float,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        """Decode 阶段注意力计算（单 token）。

        Args:
            q: query 张量
            k: key 张量（未使用，从 cache 读取）
            v: value 张量（未使用，从 cache 读取）
            context: 推理上下文
            scale: softmax 缩放因子
            k_cache: key cache
            v_cache: value cache

        Returns:
            注意力输出
        """
        return flash_attn_with_kvcache(
            q.unsqueeze(1), k_cache, v_cache,
            cache_seqlens=context.context_lens,
            block_table=context.block_tables,
            softmax_scale=scale,
            causal=True,
        )
