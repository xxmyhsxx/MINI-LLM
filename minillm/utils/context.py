from dataclasses import dataclass

import torch


@dataclass
class Context:
    """全局推理上下文，存储当前推理步骤的状态。

    Attributes:
        is_prefill: 是否为 prefill 阶段
        cu_seqlens_q: query 累积序列长度
        cu_seqlens_k: key 累积序列长度
        max_seqlen_q: 最大 query 序列长度
        max_seqlen_k: 最大 key 序列长度
        slot_mapping: KV Cache 槽位映射
        context_lens: 上下文长度（decode 阶段）
        block_tables: 物理块表
    """

    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None


_CONTEXT = Context()


def get_context() -> Context:
    """获取全局推理上下文。"""
    return _CONTEXT


def set_context(
    is_prefill: bool,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    slot_mapping: torch.Tensor | None = None,
    context_lens: torch.Tensor | None = None,
    block_tables: torch.Tensor | None = None,
):
    """设置全局推理上下文。"""
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        slot_mapping, context_lens, block_tables,
    )


def reset_context():
    """重置全局推理上下文。"""
    global _CONTEXT
    _CONTEXT = Context()
