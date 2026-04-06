"""Vision encoder for Qwen2.5-VL models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_func
from typing import Optional


class VisionRotaryEmbedding(nn.Module):
    """Rotary position embeddings for vision tokens."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, max_grid_size: int):
        """Generate rotary embeddings for grid positions."""
        seq = torch.arange(max_grid_size, device=self.inv_freq.device)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class PatchEmbed(nn.Module):
    """Embed image patches."""

    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 3584,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # 3D 卷积：(in_channels, temporal_patch_size, patch_size, patch_size) -> embed_dim
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            stride=(temporal_patch_size, patch_size, patch_size),
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project patches to embeddings.

        Args:
            x: (seq_len, patch_dim) flattened patches

        Returns:
            (seq_len, embed_dim) patch embeddings
        """
        seq_len, patch_dim = x.shape
        # reshape to (seq_len, C, T, H, W) to use Conv3d as Linear
        x = x.view(seq_len, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        x = self.proj(x)  # (seq_len, embed_dim, 1, 1, 1)
        return x.flatten(1)


class VisionAttention(nn.Module):
    """Multi-head attention for vision encoder."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.qkv = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=True)
        self.proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Optional[tuple] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Apply attention with position embeddings.

        Args:
            hidden_states: (total_seq_len, hidden_size)
            cu_seqlens: Cumulative sequence lengths for variable-length attention
            position_embeddings: (cos, sin) tuple for rotary embeddings
        """
        seq_len, _ = hidden_states.shape

        qkv = self.qkv(hidden_states)
        qkv = qkv.reshape(seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(1)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            # 应用 RoPE: 使用 rotate_half 方法 (与 HF 对齐)
            # cos/sin shape: (seq_len, rotary_dim) -> (seq_len, 1, rotary_dim) for broadcasting
            cos = cos.unsqueeze(1)  # (seq_len, 1, rotary_dim)
            sin = sin.unsqueeze(1)  # (seq_len, 1, rotary_dim)

            # rotate_half: 将后半部分移到前面并取负
            def rotate_half(x):
                x1 = x[..., : x.shape[-1] // 2]
                x2 = x[..., x.shape[-1] // 2 :]
                return torch.cat((-x2, x1), dim=-1)

            # 应用 RoPE: q_embed = q * cos + rotate_half(q) * sin
            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
        attn_output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=1.0 / (self.head_dim ** 0.5),
            causal=False,
        )
        attn_output = attn_output.reshape(seq_len, self.hidden_size)
        return self.proj(attn_output)


class VisionMLP(nn.Module):
    """MLP for vision encoder."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class VisionBlock(nn.Module):
    """Transformer block for vision encoder."""

    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6, bias=False)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6, bias=False)
        self.attn = VisionAttention(config)
        self.mlp = VisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Optional[tuple] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Attention with residual
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        # MLP with residual
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class PatchMerger(nn.Module):
    """Merge spatial patches to reduce sequence length."""

    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2):
        super().__init__()
        self.hidden_size = context_dim
        self.spatial_merge_size = spatial_merge_size

        # 计算中间维度：context_dim * spatial_merge_size^2
        # 对于 Qwen2.5-VL: 1280 * 2^2 = 5120
        hidden_dim = context_dim * (spatial_merge_size ** 2)

        self.ln_q = nn.LayerNorm(context_dim, eps=1e-6, bias=False)
        self.mlp = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=True),  # 5120 -> 5120
            nn.GELU(),
            nn.Linear(hidden_dim, dim, bias=True),  # 5120 -> 2048
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Merge patches spatially.

        Args:
            x: (seq_len, hidden_size) where seq_len = grid_t * grid_h * grid_w

        Returns:
            (merged_seq_len, dim) where merged_seq_len accounts for spatial merging
        """
        x = self.ln_q(x)

        # Reshape to merge spatial patches
        # Assuming x is already arranged spatially
        seq_len, hidden_size = x.shape
        merge_factor = self.spatial_merge_size ** 2

        # Group every merge_factor patches
        merged_seq_len = seq_len // merge_factor
        x = x.reshape(merged_seq_len, merge_factor * hidden_size)

        # Apply MLP
        for layer in self.mlp:
            x = layer(x)

        return x


class Qwen2_5_VisionTransformer(nn.Module):
    """Vision transformer for Qwen2.5-VL."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.window_size = config.window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        # Patch embedding
        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        # Rotary position embeddings
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            VisionBlock(config) for _ in range(config.depth)
        ])

        # Patch merger
        self.merger = PatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
        )
        self._grid_cache: dict[
            tuple[tuple[int, int, int], ...],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}

    def rot_pos_emb(self, grid_thw: torch.Tensor):
        """Generate rotary position embeddings for grid."""
        pos_ids = []
        for t, h, w in grid_thw.tolist():
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3).flatten()

            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))

        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_window_index(self, grid_thw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """计算视觉窗口重排索引。"""
        window_index = []
        cu_window_seqlens = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size

        # 处理 window_size 为 0 的情况
        if vit_merger_window_size == 0:
            vit_merger_window_size = 1

        for grid_t, grid_h, grid_w in grid_thw.tolist():
            llm_grid_h = grid_h // self.spatial_merge_size
            llm_grid_w = grid_w // self.spatial_merge_size
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += grid_t * llm_grid_h * llm_grid_w

        return torch.cat(window_index, dim=0), torch.tensor(cu_window_seqlens, dtype=torch.int32)

    def _get_cached_grid_state(
        self,
        grid_thw: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """缓存与 grid_thw 相关的视觉索引与位置编码。"""
        cache_key = tuple(tuple(int(x) for x in row) for row in grid_thw.tolist())
        if cache_key not in self._grid_cache:
            rotary_pos_emb = self.rot_pos_emb(grid_thw)
            window_index, cu_window_seqlens = self.get_window_index(grid_thw)
            reverse_indices = torch.argsort(window_index)
            seq_len = rotary_pos_emb.shape[0]
            rotary_pos_emb_windowed = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
            rotary_pos_emb_windowed = rotary_pos_emb_windowed[window_index, :, :]
            rotary_pos_emb_windowed = rotary_pos_emb_windowed.reshape(seq_len, -1)
            emb = torch.cat((rotary_pos_emb_windowed, rotary_pos_emb_windowed), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            self._grid_cache[cache_key] = (
                rotary_pos_emb.cpu(),
                window_index.cpu(),
                cu_window_seqlens.cpu(),
                reverse_indices.cpu(),
                torch.stack([cos.cpu(), sin.cpu()], dim=0),
            )
        rotary_pos_emb, window_index, cu_window_seqlens, reverse_indices, cos_sin = self._grid_cache[cache_key]
        return (
            rotary_pos_emb.to(device=device),
            window_index.to(device=device),
            cu_window_seqlens.to(device=device),
            reverse_indices.to(device=device),
            cos_sin.to(device=device),
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Encode image patches.

        Args:
            pixel_values: (seq_len, patch_dim) flattened image patches
            grid_thw: (num_images, 3) grid dimensions [t, h, w]

        Returns:
            (merged_seq_len, hidden_size) encoded vision features
        """
        hidden_states = self.patch_embed(pixel_values)

        rotary_pos_emb, window_index, cu_window_seqlens, reverse_indices, cos_sin = self._get_cached_grid_state(
            grid_thw,
            hidden_states.device,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        position_embeddings = (cos_sin[0], cos_sin[1])

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32).to(device=hidden_states.device)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for layer_num, block in enumerate(self.blocks):
            cu_seqlens_now = cu_seqlens if layer_num in self.fullatt_block_indexes else cu_window_seqlens
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                position_embeddings=position_embeddings,
            )

        merged_hidden_states = self.merger(hidden_states)
        merged_hidden_states = merged_hidden_states[reverse_indices, :]
        return merged_hidden_states
