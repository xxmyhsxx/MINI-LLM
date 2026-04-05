"""注意力后端抽象层。

定义 AttentionBackend 接口，支持通过配置切换不同的注意力计算实现。
当前提供 FlashAttentionBackend 作为默认实现。
"""

from abc import ABC, abstractmethod

import torch


class AttentionBackend(ABC):
    """注意力计算后端基类。

    所有注意力后端必须实现 prefill 和 decode 两个方法。
    KV Cache 的写入也由后端负责（某些后端可能将写入融合到计算中）。
    """

    @abstractmethod
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
        """Prefill 阶段注意力计算。

        Args:
            q: query，形状 (N, num_heads, head_dim)
            k: key，形状 (N, num_kv_heads, head_dim)
            v: value，形状 (N, num_kv_heads, head_dim)
            context: 全局推理上下文
            scale: softmax 缩放因子
            k_cache: key cache
            v_cache: value cache

        Returns:
            注意力输出，形状 (N, num_heads, head_dim)
        """
        ...

    @abstractmethod
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
            q: query，形状 (N, num_heads, head_dim)
            k: key，形状 (N, num_kv_heads, head_dim)
            v: value，形状 (N, num_kv_heads, head_dim)
            context: 全局推理上下文
            scale: softmax 缩放因子
            k_cache: key cache
            v_cache: value cache

        Returns:
            注意力输出，形状 (N, num_heads, head_dim)
        """
        ...

    def store_kvcache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """将 key/value 写入 KV Cache。

        默认空实现（某些后端可能不需要显式写入）。

        Args:
            key: key 张量
            value: value 张量
            k_cache: key cache
            v_cache: value cache
            slot_mapping: 槽位映射
        """
        pass
