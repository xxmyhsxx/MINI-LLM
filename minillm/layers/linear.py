import torch
from torch import nn
import torch.nn.functional as F


class LinearBase(nn.Module):
    """线性层基类（单卡版本，无 TP 依赖）。

    支持自定义 weight_loader，用于加载 HuggingFace 权重。

    Attributes:
        weight: 权重参数
        bias: 偏置参数（可选）
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """默认权重加载器，直接拷贝。

        Args:
            param: 目标参数
            loaded_weight: 加载的权重
        """
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。"""
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """直接线性层，无并行。

    用于不需要分片的场景。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量

        Returns:
            输出张量
        """
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """列并行线性层（单卡版本，加载完整权重）。

    保留 packed_modules_mapping 兼容性，权重加载时按 shard 分片。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。"""
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """合并列并行线性层（单卡版本）。

    用于 gate_proj + up_proj 的合并加载。

    Attributes:
        output_sizes: 各分片的输出大小列表
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        """按分片 ID 加载权重。

        Args:
            param: 目标参数
            loaded_weight: 加载的权重
            loaded_shard_id: 分片 ID（0 或 1）
        """
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id])
        shard_size = self.output_sizes[loaded_shard_id]
        param_data = param_data.narrow(0, shard_offset, shard_size)
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    """QKV 并行线性层（单卡版本）。

    将 q_proj、k_proj、v_proj 合并加载到一个权重矩阵。

    Attributes:
        head_size: 每个头的维度
        num_heads: 注意力头数
        num_kv_heads: KV 头数
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = total_num_heads
        self.num_kv_heads = total_num_kv_heads
        output_size = (total_num_heads + 2 * total_num_kv_heads) * head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        """按 Q/K/V 分片加载权重。

        Args:
            param: 目标参数
            loaded_weight: 加载的权重
            loaded_shard_id: 分片标识（"q"、"k"、"v"）
        """
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(0, shard_offset, shard_size)
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    """行并行线性层（单卡版本）。

    权重直接加载，无 all_reduce。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。"""
        return F.linear(x, self.weight, self.bias)
