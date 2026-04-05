"""Sequence 单元测试。"""

from minillm.engine.sequence import Sequence, SequenceStatus
from minillm.sampling_params import SamplingParams


def test_sequence_init():
    """测试序列初始化。"""
    seq = Sequence([1, 2, 3, 4, 5])
    assert seq.num_tokens == 5
    assert seq.num_prompt_tokens == 5
    assert seq.num_completion_tokens == 0
    assert seq.status == SequenceStatus.WAITING
    assert seq.last_token == 5
    assert len(seq) == 5


def test_sequence_append_token():
    """测试追加 token。"""
    seq = Sequence([1, 2, 3])
    seq.append_token(4)
    assert seq.num_tokens == 4
    assert seq.num_completion_tokens == 1
    assert seq.last_token == 4
    assert seq.completion_token_ids == [4]


def test_sequence_block_calculation():
    """测试物理块计算。"""
    # block_size = 256（默认值）
    seq = Sequence([0] * 300)
    assert seq.num_blocks == 2  # ceil(300/256) = 2
    assert seq.last_block_num_tokens == 44  # 300 - 256
    assert len(seq.block(0)) == 256
    assert len(seq.block(1)) == 44


def test_sequence_block_boundary():
    """测试物理块边界。"""
    seq = Sequence([0] * 256)
    assert seq.num_blocks == 1
    assert seq.last_block_num_tokens == 256


def test_sequence_getitem():
    """测试 token 访问。"""
    seq = Sequence([10, 20, 30])
    assert seq[0] == 10
    assert seq[1] == 20
    assert seq[2] == 30
    assert seq[-1] == 30


def test_sequence_is_finished():
    """测试完成状态。"""
    seq = Sequence([1, 2])
    assert not seq.is_finished
    seq.status = SequenceStatus.FINISHED
    assert seq.is_finished


def test_sequence_getstate_setstate():
    """测试序列化和反序列化。"""
    seq = Sequence([1, 2, 3])
    state = seq.__getstate__()
    new_seq = Sequence.__new__(Sequence)
    new_seq.__setstate__(state)
    assert new_seq.num_tokens == 3
    assert new_seq.num_prompt_tokens == 3


def test_sequence_with_sampling_params():
    """测试自定义采样参数。"""
    sp = SamplingParams(temperature=0.5, max_tokens=100)
    seq = Sequence([1, 2, 3], sp)
    assert seq.temperature == 0.5
    assert seq.max_tokens == 100


def test_sequence_prompt_completion_split():
    """测试 prompt 和 completion 分割。"""
    seq = Sequence([10, 20, 30])
    seq.append_token(40)
    seq.append_token(50)
    assert seq.prompt_token_ids == [10, 20, 30]
    assert seq.completion_token_ids == [40, 50]
    assert seq.num_prompt_tokens == 3
    assert seq.num_completion_tokens == 2
