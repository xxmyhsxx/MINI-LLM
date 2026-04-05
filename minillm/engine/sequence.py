from copy import copy
from enum import Enum, auto
from itertools import count

from minillm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """序列状态枚举。"""

    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    """表示一个推理请求序列。

    跟踪 token 序列、状态、block_table 等信息。
    支持序列化（__getstate__/__setstate__）用于状态保存。

    Attributes:
        block_size: 共享的 KV Cache 块大小
        seq_id: 序列唯一 ID
        status: 序列状态
        token_ids: token id 列表
        last_token: 最后一个 token id
        num_tokens: 总 token 数
        num_prompt_tokens: prompt token 数
        num_cached_tokens: 已缓存的 token 数
        block_table: 物理块索引列表
        temperature: 采样温度
        max_tokens: 最大生成 token 数
        ignore_eos: 是否忽略 EOS
    """

    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams = SamplingParams()):
        """初始化序列。

        Args:
            token_ids: prompt token id 列表
            sampling_params: 采样参数
        """
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.top_k = sampling_params.top_k
        self.top_p = sampling_params.top_p
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self) -> int:
        """返回序列总 token 数。"""
        return self.num_tokens

    def __getitem__(self, key):
        """按索引访问 token。"""
        return self.token_ids[key]

    @property
    def is_finished(self) -> bool:
        """检查序列是否已完成。"""
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self) -> int:
        """返回已生成的 completion token 数。"""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self) -> list[int]:
        """返回 prompt token 列表。"""
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self) -> list[int]:
        """返回 completion token 列表。"""
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self) -> int:
        """返回已缓存的物理块数。"""
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self) -> int:
        """返回序列所需的总物理块数。"""
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self) -> int:
        """返回最后一个物理块中的 token 数。"""
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i: int) -> list[int]:
        """获取第 i 个物理块对应的 token 列表。

        Args:
            i: 物理块索引

        Returns:
            对应的 token id 列表
        """
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size: (i + 1) * self.block_size]

    def append_token(self, token_id: int):
        """追加一个 token 到序列末尾。

        Args:
            token_id: token id
        """
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        """序列化状态。"""
        return (
            self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
            self.block_table,
            self.token_ids if self.num_completion_tokens == 0 else self.last_token,
        )

    def __setstate__(self, state):
        """反序列化状态。"""
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
