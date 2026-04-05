from collections import deque

from minillm.config import Config
from minillm.engine.sequence import Sequence, SequenceStatus
from minillm.engine.block_manager import BlockManager


class Scheduler:
    """推理调度器，管理 prefill 和 decode 阶段的请求调度。

    支持 prefill 批处理、decode 调度和 preemption（抢占）。

    Attributes:
        max_num_seqs: 最大并发序列数
        max_num_batched_tokens: 单步最大 token 数
        eos: EOS token id
        block_manager: 物理块管理器
        waiting: 等待队列
        running: 运行队列
    """

    def __init__(self, config: Config):
        """初始化调度器。

        Args:
            config: 推理配置
        """
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self) -> bool:
        """检查所有序列是否处理完毕。"""
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """添加一个新序列到等待队列。

        Args:
            seq: 新序列
        """
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """执行一次调度，返回待处理的序列列表和是否为 prefill 阶段。

        Returns:
            (scheduled_seqs, is_prefill) 元组
        """
        # prefill 阶段
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode 阶段
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """抢占一个序列（释放其资源，放回等待队列）。

        Args:
            seq: 被抢占的序列
        """
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        """后处理推理结果，追加 token 并检查完成状态。

        Args:
            seqs: 推理序列列表
            token_ids: 生成的 token id 列表
        """
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            is_eos = False
            if not seq.ignore_eos:
                if isinstance(self.eos, list) or isinstance(self.eos, set) or isinstance(self.eos, tuple):
                    is_eos = token_id in self.eos
                else:
                    is_eos = token_id == self.eos
            if is_eos or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
