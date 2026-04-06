"""模型运行器（单卡版本）。

负责模型加载、KV Cache 分配、推理执行。
"""

import torch

from minillm.config import Config
from minillm.engine.sequence import Sequence
from minillm.layers.sampler import Sampler
from minillm.models.registry import create
from minillm.utils.context import get_context, reset_context, set_context
from minillm.utils.loader import load_model


class ModelRunner:
    """单卡模型运行器。

    简化自 nano-vllm 的 ModelRunner，移除 TP，保留单卡 decode CUDA Graph 支持。

    Attributes:
        config: 推理配置
        model: 推理模型
        sampler: 采样器
        kv_cache: KV Cache 张量
        block_size: 物理块大小
    """

    def __init__(self, config: Config):
        """初始化模型运行器。

        Args:
            config: 推理配置
        """
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager

        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = create(hf_config.model_type, hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()

        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()

        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

    def warmup_model(self):
        """预热模型，测量峰值显存占用。"""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens = self.config.max_num_batched_tokens
        max_model_len = self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """分配 KV Cache 显存。

        支持三种配置方式：
        1. cache_size_tokens — 按 token 容量分配
        2. cache_size_mb — 按显存大小分配
        3. 自动计算（默认）
        """
        config = self.config
        hf_config = config.hf_config

        # 处理 VLM 配置（text_config 嵌套）
        if hasattr(hf_config, 'text_config'):
            text_config = hf_config.text_config
            num_kv_heads = text_config.num_key_value_heads
            head_dim = getattr(text_config, "head_dim", text_config.hidden_size // text_config.num_attention_heads)
            num_hidden_layers = text_config.num_hidden_layers
            torch_dtype = text_config.torch_dtype
        else:
            num_kv_heads = hf_config.num_key_value_heads
            head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
            num_hidden_layers = hf_config.num_hidden_layers
            torch_dtype = hf_config.torch_dtype
        block_bytes = (
            2 * num_hidden_layers * self.block_size
            * num_kv_heads * head_dim * torch_dtype.itemsize
        )

        if config.cache_size_tokens is not None:
            config.num_kvcache_blocks = config.cache_size_tokens // self.block_size
        elif config.cache_size_mb is not None:
            config.num_kvcache_blocks = (config.cache_size_mb * 1024 * 1024) // block_bytes
        else:
            free, total = torch.cuda.mem_get_info()
            used = total - free
            peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
            current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
            config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes

        assert config.num_kvcache_blocks > 0, f"KV Cache 块数必须大于 0: {config.num_kvcache_blocks}"
        self.kv_cache = torch.empty(
            2,
            num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
            device='cuda',
        )

        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def get_kv_cache_bytes(self) -> int:
        """返回 KV Cache 总字节数。"""
        return self.kv_cache.numel() * self.kv_cache.element_size()

    def get_model_bytes(self) -> int:
        """返回模型参数与缓冲区总字节数。"""
        parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in self.model.parameters())
        buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in self.model.buffers())
        return parameter_bytes + buffer_bytes

    def get_memory_profile(
        self,
        kv_cache_used_blocks: int = 0,
        kv_cache_peak_used_blocks: int = 0,
    ) -> dict:
        """返回结构化显存画像。

        Args:
            kv_cache_used_blocks: 当前已使用的 KV block 数量
            kv_cache_peak_used_blocks: 本次请求峰值已使用的 KV block 数量

        Returns:
            显存画像字典
        """
        kv_cache_total_bytes = self.get_kv_cache_bytes()
        kv_cache_total_blocks = self.config.num_kvcache_blocks
        kv_cache_used_bytes = 0
        kv_cache_peak_used_bytes = 0
        if kv_cache_total_blocks > 0:
            kv_cache_used_bytes = kv_cache_total_bytes * kv_cache_used_blocks // kv_cache_total_blocks
            kv_cache_peak_used_bytes = kv_cache_total_bytes * kv_cache_peak_used_blocks // kv_cache_total_blocks
        return {
            "model_bytes": self.get_model_bytes(),
            "kv_cache_total_bytes": kv_cache_total_bytes,
            "kv_cache_used_bytes": kv_cache_used_bytes,
            "kv_cache_peak_used_bytes": kv_cache_peak_used_bytes,
            "kv_cache_total_blocks": kv_cache_total_blocks,
            "kv_cache_used_blocks": kv_cache_used_blocks,
            "kv_cache_peak_used_blocks": kv_cache_peak_used_blocks,
            "cuda_memory_allocated_bytes": torch.cuda.memory_allocated(),
            "cuda_max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_memory_reserved_bytes": torch.cuda.memory_reserved(),
            "cuda_max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        }

    def reset_peak_memory_stats(self) -> None:
        """重置 CUDA 峰值显存统计。"""
        torch.cuda.reset_peak_memory_stats()

    def prepare_block_tables(self, seqs: list[Sequence]) -> torch.Tensor:
        """准备 block table 张量。

        Args:
            seqs: 序列列表

        Returns:
            block table 张量，形状 (batch_size, max_blocks)
        """
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        return torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

    def prepare_prefill(self, seqs: list[Sequence]) -> tuple[torch.Tensor, torch.Tensor]:
        """准备 prefill 阶段的输入。

        Args:
            seqs: 序列列表

        Returns:
            (input_ids, positions) 元组
        """
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
        )
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]) -> tuple[torch.Tensor, torch.Tensor]:
        """准备 decode 阶段的输入。

        Args:
            seqs: 序列列表

        Returns:
            (input_ids, positions) 元组
        """
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """准备采样参数张量。

        Args:
            seqs: 序列列表

        Returns:
            (temperatures, top_ks, top_ps) 元组
        """
        temperatures = [seq.temperature for seq in seqs]
        top_ks = [seq.top_k for seq in seqs]
        top_ps = [seq.top_p for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        top_ks = torch.tensor(top_ks, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        top_ps = torch.tensor(top_ps, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures, top_ks, top_ps

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        """执行模型前向传播。

        Args:
            input_ids: 输入 token id
            positions: 位置索引
            is_prefill: 是否为 prefill 阶段

        Returns:
            logits
        """
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))

        bs = input_ids.size(0)
        context = get_context()
        graph_bs = next(graph_bs for graph_bs in self.graph_bs if graph_bs >= bs)
        graph = self.graphs[graph_bs]
        graph_vars = self.graph_vars
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"].fill_(-1)
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """执行一轮推理。

        Args:
            seqs: 序列列表
            is_prefill: 是否为 prefill 阶段

        Returns:
            生成的 token id 列表
        """
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures, top_ks, top_ps = self.prepare_sample(seqs)
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures, top_ks, top_ps).tolist()
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        """捕获 decode 阶段的 CUDA Graph。"""
        config = self.config
        hf_config = config.hf_config

        # 处理 VLM 配置（text_config 嵌套）
        if hasattr(hf_config, 'text_config'):
            hidden_size = hf_config.text_config.hidden_size
        else:
            hidden_size = hf_config.hidden_size

        max_bs = min(config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64, device='cuda')
        positions = torch.zeros(max_bs, dtype=torch.int64, device='cuda')
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32, device='cuda')
        context_lens = torch.zeros(max_bs, dtype=torch.int32, device='cuda')
        block_tables = torch.full((max_bs, max_num_blocks), -1, dtype=torch.int32, device='cuda')
        outputs = torch.zeros(max_bs, hidden_size, device='cuda')
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = {
            "input_ids": input_ids,
            "positions": positions,
            "slot_mapping": slot_mapping,
            "context_lens": context_lens,
            "block_tables": block_tables,
            "outputs": outputs,
        }
