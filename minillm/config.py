import os
from dataclasses import dataclass

from transformers import AutoConfig


@dataclass
class Config:
    """推理引擎配置。

    Attributes:
        model: 模型路径（HuggingFace 格式目录）
        max_num_batched_tokens: 单步最大 token 数
        max_num_seqs: 最大并发序列数
        max_model_len: 最大模型长度
        gpu_memory_utilization: GPU 显存利用率
        enforce_eager: 是否禁用 CUDA Graph
        kvcache_block_size: KV Cache 物理块大小（tokens）
        num_kvcache_blocks: KV Cache 物理块数量（-1 表示自动计算）
        cache_size_mb: 用户指定 KV Cache 显存大小（MB），优先于 num_kvcache_blocks
        cache_size_tokens: 用户指定 KV Cache Token 容量，优先于 cache_size_mb
        hf_config: HuggingFace 模型配置（自动加载）
        eos: EOS token id
    """

    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    enforce_eager: bool = False
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    cache_size_mb: int | None = None
    cache_size_tokens: int | None = None
    hf_config: AutoConfig | None = None
    eos: int = -1

    def __post_init__(self):
        """校验配置参数并加载 HuggingFace 模型配置。"""
        assert os.path.isdir(self.model), f"模型路径不存在: {self.model}"
        assert self.kvcache_block_size % 256 == 0, "kvcache_block_size 必须是 256 的倍数"
        self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        max_position_embeddings = getattr(self.hf_config, "max_position_embeddings", None)
        if max_position_embeddings is None and hasattr(self.hf_config, "text_config"):
            max_position_embeddings = getattr(self.hf_config.text_config, "max_position_embeddings", None)
        if max_position_embeddings is not None:
            self.max_model_len = min(self.max_model_len, max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
