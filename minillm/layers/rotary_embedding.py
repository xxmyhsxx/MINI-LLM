from functools import lru_cache

import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """对最后一维做半维旋转。"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """应用旋转位置编码。"""
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """旋转位置编码（RoPE）。"""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        mrope_section: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.mrope_section = mrope_section
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _apply_multimodal_rotary(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """对 Qwen2.5-VL 的 3D mRoPE 应用旋转。"""
        assert self.mrope_section is not None
        sections = list(self.mrope_section)
        cos = torch.cat([chunk[i % 3] for i, chunk in enumerate(cos.split(sections, dim=-1))], dim=-1)
        sin = torch.cat([chunk[i % 3] for i, chunk in enumerate(sin.split(sections, dim=-1))], dim=-1)
        return apply_rotary_emb(x, cos, sin)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """应用 RoPE 到 query 和 key。"""
        if positions.ndim == 1:
            cos_sin = self.cos_sin_cache[positions]
            cos, sin = cos_sin.chunk(2, dim=-1)
            query = apply_rotary_emb(query, cos, sin)
            key = apply_rotary_emb(key, cos, sin)
            return query, key

        assert positions.ndim == 2 and positions.shape[0] == 3, "多模态位置编码必须为 (3, N)"
        assert self.mrope_section is not None, "多模态位置编码需要 mrope_section"
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = self._apply_multimodal_rotary(query, cos, sin)
        key = self._apply_multimodal_rotary(key, cos, sin)
        return query, key


@lru_cache(32)
def _get_rope_cached(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    mrope_section: tuple[int, ...] | None,
) -> RotaryEmbedding:
    """获取可缓存的 RoPE 实例。"""
    return RotaryEmbedding(head_size, rotary_dim, max_position, base, mrope_section=mrope_section)


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
) -> RotaryEmbedding:
    """获取 RoPE 实例（带缓存）。"""
    mrope_section = None
    if rope_scaling is not None:
        # 仅当 rope_scaling 存在且类型为 mrope 时才处理
        if rope_scaling.get("type") == "mrope":
            mrope_section = tuple(rope_scaling["mrope_section"])
            base = rope_scaling.get("rope_theta", base)
        # 其他类型的 rope_scaling 暂不支持，使用默认 RoPE
    return _get_rope_cached(head_size, rotary_dim, max_position, base, mrope_section)
