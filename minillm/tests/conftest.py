"""pytest 配置文件。"""

import pytest


def pytest_collection_modifyitems(config, items):
    """自动标记需要 GPU 的测试。"""
    for item in items:
        if "gpu" in item.nodeid.lower() or any(
            mark.name in ("gpu", "slow") for mark in item.iter_markers()
        ):
            item.add_marker(pytest.mark.gpu)
