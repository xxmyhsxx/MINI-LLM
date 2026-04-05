from collections import deque

import xxhash
import numpy as np

from minillm.engine.sequence import Sequence


class Block:
    """KV Cache 物理块。

    Attributes:
        block_id: 物理块唯一标识
        ref_count: 引用计数
        hash: 块内容哈希（用于前缀缓存）
        token_ids: 块内 token 列表
    """

    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        """更新块的哈希和 token 内容。

        Args:
            hash: 新哈希值
            token_ids: 新 token 列表
        """
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """重置块状态为已分配。"""
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """KV Cache 物理块管理器。

    管理固定数量的物理块，支持分配、释放、前缀缓存。

    Attributes:
        block_size: 每个物理块的 token 容量
        blocks: 所有物理块列表
        hash_to_block_id: 哈希到物理块索引的映射（前缀缓存）
        free_block_ids: 空闲物理块队列
        used_block_ids: 已使用物理块集合
    """

    def __init__(self, num_blocks: int, block_size: int):
        """初始化块管理器。

        Args:
            num_blocks: 物理块总数
            block_size: 每个物理块的 token 容量
        """
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1) -> int:
        """计算 token 序列的哈希值（用于前缀缓存）。

        Args:
            token_ids: token 列表
            prefix: 前一个块的哈希（用于链式哈希）

        Returns:
            哈希值
        """
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        """分配一个空闲物理块。

        Args:
            block_id: 目标物理块索引

        Returns:
            分配的物理块
        """
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int):
        """释放一个物理块。

        Args:
            block_id: 目标物理块索引
        """
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        """检查是否可以为序列分配足够的物理块。

        Args:
            seq: 目标序列

        Returns:
            是否可以分配
        """
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """为序列分配物理块，支持前缀缓存。

        Args:
            seq: 目标序列
        """
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        """释放序列占用的所有物理块。

        Args:
            seq: 目标序列
        """
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """检查是否可以为序列追加 token（可能需要新块）。

        Args:
            seq: 目标序列

        Returns:
            是否可以追加
        """
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """为序列追加 token 做准备（更新块哈希或分配新块）。

        Args:
            seq: 目标序列
        """
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks - 1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1

    def free_all(self):
        """释放所有已使用的块。"""
        for block_id in list(self.used_block_ids):
            block = self.blocks[block_id]
            block.ref_count = 0
            self._deallocate_block(block_id)
        self.hash_to_block_id.clear()
