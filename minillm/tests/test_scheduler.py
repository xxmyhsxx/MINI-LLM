"""Scheduler 单元测试。"""

from minillm.config import Config
from minillm.engine.scheduler import Scheduler
from minillm.engine.sequence import Sequence, SequenceStatus


def _make_scheduler(max_num_seqs=4, max_num_batched_tokens=4096, num_kvcache_blocks=100) -> Scheduler:
    """创建一个不依赖模型路径的调度器（用于测试）。"""
    sched = Scheduler.__new__(Scheduler)
    from minillm.engine.block_manager import BlockManager
    sched.max_num_seqs = max_num_seqs
    sched.max_num_batched_tokens = max_num_batched_tokens
    sched.eos = -1
    sched.block_manager = BlockManager(num_kvcache_blocks, 256)
    from collections import deque
    sched.waiting = deque()
    sched.running = deque()
    return sched


def test_scheduler_init():
    """测试调度器初始化。"""
    sched = _make_scheduler()
    assert sched.is_finished()


def test_scheduler_add():
    """测试添加序列。"""
    sched = _make_scheduler()
    seq = Sequence([1, 2, 3])
    sched.add(seq)
    assert not sched.is_finished()
    assert len(sched.waiting) == 1


def test_scheduler_prefill_schedule():
    """测试 prefill 调度。"""
    sched = _make_scheduler()
    seq1 = Sequence([1, 2, 3])
    seq2 = Sequence([4, 5, 6])
    sched.add(seq1)
    sched.add(seq2)

    seqs, is_prefill = sched.schedule()
    assert is_prefill
    assert len(seqs) == 2
    assert seq1.status == SequenceStatus.RUNNING
    assert seq2.status == SequenceStatus.RUNNING
    assert len(sched.waiting) == 0
    assert len(sched.running) == 2


def test_scheduler_max_num_seqs():
    """测试最大并发数限制。"""
    sched = _make_scheduler(max_num_seqs=2)
    for i in range(4):
        sched.add(Sequence([i, i + 1, i + 2]))

    seqs, is_prefill = sched.schedule()
    assert is_prefill
    assert len(seqs) == 2
    assert len(sched.waiting) == 2


def test_scheduler_postprocess():
    """测试后处理。"""
    sched = _make_scheduler()
    seq = Sequence([1, 2, 3])
    sp = Sequence.__new__(Sequence)
    sp.max_tokens = 2
    sp.ignore_eos = True
    sp.temperature = 1.0

    seq.max_tokens = 2
    seq.ignore_eos = True
    sched.add(seq)
    sched.schedule()
    sched.postprocess([seq], [10])

    assert seq.num_tokens == 4
    assert seq.last_token == 10


def test_scheduler_preempt():
    """测试抢占。"""
    sched = _make_scheduler(num_kvcache_blocks=4)  # 只有 4 个块
    seq = Sequence([0] * 300)  # 需要 2 个块
    sched.add(seq)
    sched.schedule()
    assert seq.status == SequenceStatus.RUNNING

    sched.preempt(seq)
    assert seq.status == SequenceStatus.WAITING
    assert len(sched.waiting) == 1
