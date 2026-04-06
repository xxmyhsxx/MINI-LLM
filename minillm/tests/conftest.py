"""pytest 配置文件。"""

import gc

import pytest
import torch


_priority_test = "tests/test_qwen2_5_vl.py::test_vlm_engine_output_is_image_grounded"


def _cleanup_cuda() -> None:
    """清理 CUDA 缓存，降低测试之间的显存串扰。"""
    if not torch.cuda.is_available():
        return
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(autouse=True)
def cleanup_cuda_for_gpu_tests(request):
    """在 GPU 测试前后清理 CUDA 缓存。"""
    if "gpu" in request.keywords:
        _cleanup_cuda()
    yield
    if "gpu" in request.keywords:
        _cleanup_cuda()


def pytest_collection_modifyitems(config, items):
    """自动标记 GPU 测试，并将超重显存用例提前执行。"""
    del config
    for item in items:
        if "gpu" in item.nodeid.lower() or any(
            mark.name in ("gpu", "slow") for mark in item.iter_markers()
        ):
            item.add_marker(pytest.mark.gpu)
    items.sort(key=lambda item: (item.nodeid != _priority_test, item.nodeid))
