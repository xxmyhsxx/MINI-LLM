"""Qwen2 模型实现（单卡版本，无 TP）。

支持 Qwen2.5 系列模型。
"""

import torch
from torch import nn
from transformers import Qwen2Config

from minillm.layers.activation import SiluAndMul
from minillm.layers.attention import Attention
from minillm.layers.layernorm import RMSNorm
from minillm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from minillm.layers.rotary_embedding import get_rope
from minillm.layers.embed_head import Embedding, LMHead


class Qwen2Attention(nn.Module):
    """Qwen2 注意力层。

    Qwen2 使用 GQA（Grouped Query Attention），无 Q/K RMSNorm。

    Attributes:
        qkv_proj: 合并的 QKV 投影层
        o_proj: 输出投影层
        rotary_emb: 旋转位置编码
        attn: Flash Attention 注意力计算
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = True,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size, self.head_dim,
            num_heads, num_kv_heads, bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            num_heads * self.head_dim, hidden_size, bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads, self.head_dim, self.scaling, self.num_kv_heads,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """注意力前向传播。

        Args:
            positions: 位置索引
            hidden_states: 输入隐藏状态

        Returns:
            注意力输出
        """
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen2MLP(nn.Module):
    """Qwen2 MLP 层（SwiGLU 激活）。

    Attributes:
        gate_up_proj: 合并的 gate/up 投影
        down_proj: 下投影
        act_fn: SiLU 激活函数
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2, bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """MLP 前向传播。

        Args:
            x: 输入张量

        Returns:
            输出张量
        """
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen2DecoderLayer(nn.Module):
    """Qwen2 解码器层。

    Attributes:
        self_attn: 自注意力层
        mlp: MLP 层
        input_layernorm: 输入归一化
        post_attention_layernorm: 注意力后归一化
    """

    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.self_attn = Qwen2Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=getattr(config, "head_dim", None),
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", True),
            rope_theta=getattr(config, "rope_theta", 10000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """解码器层前向传播。

        Args:
            positions: 位置索引
            hidden_states: 输入隐藏状态
            residual: 残差张量

        Returns:
            (输出隐藏状态, 新残差)
        """
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen2Model(nn.Module):
    """Qwen2 主模型。

    Attributes:
        embed_tokens: 词嵌入层
        layers: 解码器层列表
        norm: 最终归一化层
    """

    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen2DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """模型前向传播。

        Args:
            input_ids: 输入 token id
            positions: 位置索引

        Returns:
            最终隐藏状态
        """
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def forward_embeds(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """从 embedding 开始的前向传播（用于 VLM）。

        Args:
            hidden_states: 输入的 embedding 张量
            positions: 位置索引

        Returns:
            最终隐藏状态
        """
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen2ForCausalLM(nn.Module):
    """Qwen2 因果语言模型。

    Attributes:
        packed_modules_mapping: HuggingFace 权重打包映射
        model: 主模型
        lm_head: 语言模型输出头
    """

    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.hf_config = config
        self.model = Qwen2Model(config)
        self.lm_head = LMHead(config.vocab_size, config.hidden_size)
        if getattr(config, 'tie_word_embeddings', False):
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播，返回隐藏状态。

        Args:
            input_ids: 输入 token id
            positions: 位置索引

        Returns:
            隐藏状态
        """
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """计算 logits。

        Args:
            hidden_states: 隐藏状态

        Returns:
            logits
        """
        return self.lm_head(hidden_states)
