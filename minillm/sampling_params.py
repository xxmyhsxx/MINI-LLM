from dataclasses import dataclass


@dataclass
class SamplingParams:
    """采样参数。

    Attributes:
        temperature: 温度参数。0 表示 greedy（argmax），>0 表示采样。
        top_k: Top-K 采样。0 表示不启用。
        top_p: Top-P（nucleus）采样。1.0 表示不启用。
        max_tokens: 最大生成 token 数
        ignore_eos: 是否忽略 EOS token
    """

    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        """校验采样参数。"""
        assert self.temperature >= 0, "温度不能为负数"
        assert self.top_k >= 0, "top_k 不能为负数"
        assert 0 < self.top_p <= 1.0, "top_p 必须在 (0, 1] 范围内"
