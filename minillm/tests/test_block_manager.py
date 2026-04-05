"""BlockManager 单元测试。"""

from minillm.engine.block_manager import BlockManager, Block
from minillm.engine.sequence import Sequence


def test_block_init():
    """测试物理块初始化。"""
    block = Block(42)
    assert block.block_id == 42
    assert block.ref_count == 0
    assert block.hash == -1
    assert block.token_ids == []


def test_block_reset():
    """测试物理块重置。"""
    block = Block(0)
    block.update(123, [1, 2, 3])
    block.reset()
    assert block.ref_count == 1
    assert block.hash == -1
    assert block.token_ids == []


def test_block_manager_init():
    """测试块管理器初始化。"""
    bm = BlockManager(num_blocks=10, block_size=256)
    assert len(bm.free_block_ids) == 10
    assert len(bm.used_block_ids) == 0
    assert len(bm.blocks) == 10


def test_block_manager_can_allocate():
    """测试分配检查。"""
    bm = BlockManager(num_blocks=4, block_size=256)
    seq = Sequence([0] * 300)  # 需要 2 个块
    assert bm.can_allocate(seq)

    seq2 = Sequence([0] * 800)  # 需要 4 个块（ceil(800/256) = 4）
    assert bm.can_allocate(seq2)

    seq3 = Sequence([0] * 1000)  # 需要 4 个块
    assert bm.can_allocate(seq3)


def test_block_manager_allocate():
    """测试分配。"""
    bm = BlockManager(num_blocks=10, block_size=256)
    seq = Sequence([0] * 300)
    bm.allocate(seq)
    assert len(seq.block_table) == 2
    assert len(bm.free_block_ids) == 8
    assert len(bm.used_block_ids) == 2


def test_block_manager_deallocate():
    """测试释放。"""
    bm = BlockManager(num_blocks=10, block_size=256)
    seq = Sequence([0] * 300)
    bm.allocate(seq)
    bm.deallocate(seq)
    assert len(seq.block_table) == 0
    assert seq.num_cached_tokens == 0
    assert len(bm.free_block_ids) == 10
    assert len(bm.used_block_ids) == 0


def test_block_manager_can_append():
    """测试追加检查。"""
    bm = BlockManager(num_blocks=10, block_size=256)
    seq = Sequence([0] * 256)
    bm.allocate(seq)
    assert len(seq.block_table) == 1

    # 长度 256，追加后为 257，需要新块
    seq.append_token(0)  # 长度变为 257
    assert bm.can_append(seq)


def test_block_manager_hash():
    """测试哈希计算。"""
    h1 = BlockManager.compute_hash([1, 2, 3])
    h2 = BlockManager.compute_hash([1, 2, 3])
    h3 = BlockManager.compute_hash([1, 2, 4])
    assert h1 == h2
    assert h1 != h3


def test_block_manager_hash_with_prefix():
    """测试带前缀的哈希计算。"""
    h1 = BlockManager.compute_hash([1, 2, 3], prefix=100)
    h2 = BlockManager.compute_hash([1, 2, 3], prefix=200)
    h3 = BlockManager.compute_hash([1, 2, 3])
    assert h1 != h2  # 不同前缀导致不同哈希
    assert h1 != h3


def test_block_manager_multiple_sequences():
    """测试多序列分配和释放。"""
    bm = BlockManager(num_blocks=10, block_size=256)
    seq1 = Sequence([1] * 300)  # 2 个块
    seq2 = Sequence([2] * 256)  # 1 个块，不同的 token 避免前缀缓存
    bm.allocate(seq1)
    bm.allocate(seq2)
    assert len(bm.used_block_ids) == 3
    assert len(bm.free_block_ids) == 7

    bm.deallocate(seq1)
    assert len(bm.used_block_ids) == 1
    assert len(bm.free_block_ids) == 9


def test_block_manager_prefix_caching():
    """测试前缀缓存。"""
    bm = BlockManager(num_blocks=10, block_size=4)  # 小块便于测试
    token_ids = [1, 2, 3, 4]  # 刚好一个块
    seq1 = Sequence(token_ids)
    bm.allocate(seq1)
    assert len(seq1.block_table) == 1

    # 第二个序列使用相同 token，应该命中缓存
    seq2 = Sequence(token_ids)
    bm.allocate(seq2)
    assert seq2.num_cached_tokens == 4  # 命中一个块的缓存
