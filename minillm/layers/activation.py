import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    """SiLU 激活函数 + 门控乘法（SwiGLU）。"""

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量，最后一维为 2 * intermediate_size

        Returns:
            激活后的张量，最后一维为 intermediate_size
        """
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
