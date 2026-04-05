import torch
from torch import nn


class Sampler(nn.Module):
    """采样器，兼容 HuggingFace transformers 采样管线。

    支持模式：
    - greedy（temperature=0）：直接 argmax
    - temperature：scores / temperature
    - top-k：保留 top_k 个最高分 token
    - top-p：nucleus 采样
    - 最终：softmax → multinomial

    采样顺序（与 HF 一致）：
    温度缩放 → Top-K 掩码 → Top-P 掩码 → softmax → multinomial
    """

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
    ) -> torch.Tensor:
        """采样前向传播。

        Args:
            logits: 未归一化的 logits，形状 (batch_size, vocab_size)
            temperatures: 每个样本的温度，形状 (batch_size,)
            top_ks: 每个样本的 top_k，形状 (batch_size,)
            top_ps: 每个样本的 top_p，形状 (batch_size,)

        Returns:
            采样的 token id，形状 (batch_size,)
        """
        logits = logits.float()
        greedy_mask = temperatures == 0

        # 温度缩放（greedy 样本设为 1.0 避免除零）
        safe_temps = torch.where(greedy_mask, torch.ones_like(temperatures), temperatures)
        logits = logits / safe_temps.unsqueeze(1)

        # Top-K 掩码
        if (top_ks > 0).any():
            logits = self._apply_top_k(logits, top_ks)

        # Top-P 掩码
        if (top_ps < 1.0).any():
            logits = self._apply_top_p(logits, top_ps)

        # Greedy：argmax
        greedy_tokens = logits.argmax(dim=-1)

        # Sampling：softmax → multinomial
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        # 合并结果
        tokens = torch.where(greedy_mask, greedy_tokens, sample_tokens)
        return tokens

    def _apply_top_k(self, logits: torch.Tensor, top_ks: torch.Tensor) -> torch.Tensor:
        """应用 Top-K 掩码。

        Args:
            logits: logits 张量，形状 (batch_size, vocab_size)
            top_ks: top_k 值，形状 (batch_size,)

        Returns:
            掩码后的 logits
        """
        # 找到每个样本的第 k 大值
        # 对于 top_k=0 的样本，不进行掩码
        max_k = int(top_ks.max().item())
        if max_k <= 0:
            return logits

        # 取 top_k 值
        top_k_values = torch.topk(logits, max_k, dim=-1).values[..., -1, None]
        # 对 top_k=0 的样本，设阈值为 -inf（不掩码任何值）
        threshold = torch.where(
            top_ks.unsqueeze(1) > 0,
            top_k_values,
            torch.full_like(top_k_values, float("-inf")),
        )
        # 掩码掉低于阈值的 logits
        indices_to_remove = logits < threshold
        logits = logits.masked_fill(indices_to_remove, float("-inf"))
        return logits

    def _apply_top_p(self, logits: torch.Tensor, top_ps: torch.Tensor) -> torch.Tensor:
        """应用 Top-P（nucleus）掩码。

        Args:
            logits: logits 张量，形状 (batch_size, vocab_size)
            top_ps: top_p 值，形状 (batch_size,)

        Returns:
            掩码后的 logits
        """
        # 降序排序
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        # 计算累积概率
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

        # 找到需要移除的 token（累积概率 > top_p）
        # 移除规则：将累积概率右移一位，判断哪些 token 应该保留
        sorted_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_ps.unsqueeze(1)
        # 对 top_p=1.0 的样本，不掩码任何值
        sorted_mask = sorted_mask & (top_ps < 1.0).unsqueeze(1)
        # 始终保留至少 1 个 token
        sorted_mask[:, 0] = False

        # 还原原始顺序
        indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
        logits = logits.masked_fill(indices_to_remove, float("-inf"))
        return logits
