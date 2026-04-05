import torch
from torch import nn
import torch.nn.functional as F

from minillm.utils.context import get_context


class Embedding(nn.Module):
    """词嵌入层（单卡版本）。

    Attributes:
        weight: 嵌入权重
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """权重加载器。

        Args:
            param: 目标参数
            loaded_weight: 加载的权重
        """
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: token id 张量

        Returns:
            嵌入向量
        """
        return F.embedding(x, self.weight)


class LMHead(nn.Module):
    """语言模型输出头（单卡版本）。

    将隐藏状态映射到词表空间。

    Attributes:
        weight: 输出头权重
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        super().__init__()
        assert not bias, "暂不支持 bias"
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """权重加载器。

        Args:
            param: 目标参数
            loaded_weight: 加载的权重
        """
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 隐藏状态，形状 (seq_len, hidden_size)

        Returns:
            logits，形状 (batch_size, vocab_size)
        """
        context = get_context()
        if context.is_prefill:
            # prefill 阶段只取最后一个 token 的 hidden state
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight)
        return logits
