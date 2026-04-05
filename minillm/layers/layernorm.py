import torch
from torch import nn


class RMSNorm(nn.Module):
    """RMSNorm 归一化层。

    支持两种前向模式：
    - 仅归一化（rms_forward）
    - 归一化 + 残差加法（add_rms_forward）

    Attributes:
        eps: 数值稳定性常量
        weight: 可学习的缩放参数
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """仅归一化前向传播。

        Args:
            x: 输入张量

        Returns:
            归一化后的张量
        """
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """归一化 + 残差加法前向传播。

        Args:
            x: 输入张量
            residual: 残差张量

        Returns:
            (归一化后张量, 新残差张量)
        """
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """前向传播。

        Args:
            x: 输入张量
            residual: 可选的残差张量

        Returns:
            归一化后的张量，或 (归一化张量, 残差) 元组
        """
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
