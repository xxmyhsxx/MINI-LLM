"""测试 VLM Engine 端到端推理。"""

import gc

import numpy as np
import pytest
import torch
from PIL import Image

from minillm.engine.vlm_engine import VLMEngine
from minillm.sampling_params import SamplingParams

MODEL_PATH = "/app/models/Qwen2.5-VL-3B-Instruct"


def _cleanup_cuda() -> None:
    """清理测试残留的 CUDA 显存。"""
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def vlm_engine() -> VLMEngine:
    """复用单个 VLM 引擎，避免完整测试集重复加载大模型。"""
    engine = VLMEngine(MODEL_PATH)
    yield engine
    del engine
    _cleanup_cuda()


@pytest.fixture(autouse=True)
def cleanup_cuda_between_tests():
    """在每个测试后清理缓存，降低全量回归时的显存压力。"""
    yield
    _cleanup_cuda()


@pytest.mark.gpu
def test_vlm_engine_text_only(vlm_engine: VLMEngine):
    """测试纯文本推理（无图像）。"""
    result = vlm_engine.generate(
        prompt="你好，请介绍一下你自己。",
        images=None,
        sampling_params=SamplingParams(max_tokens=32, temperature=0.0),
    )

    assert "text" in result
    assert "token_ids" in result
    assert isinstance(result["text"], str)
    assert isinstance(result["token_ids"], list)
    assert len(result["text"]) > 0
    assert len(result["token_ids"]) > 0


@pytest.mark.gpu
def test_vlm_engine_with_image(vlm_engine: VLMEngine):
    """测试图像+文本推理。"""
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

    result = vlm_engine.generate(
        prompt="描述这张图片。",
        images=[img],
        sampling_params=SamplingParams(max_tokens=64, temperature=0.7),
    )

    assert "text" in result
    assert "token_ids" in result
    assert isinstance(result["text"], str)
    assert isinstance(result["token_ids"], list)
    assert len(result["text"]) > 0
    assert len(result["token_ids"]) > 0


@pytest.mark.gpu
def test_vlm_engine_multiple_images(vlm_engine: VLMEngine):
    """测试多图像推理。"""
    img1 = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    img2 = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

    result = vlm_engine.generate(
        prompt="比较这两张图片的差异。",
        images=[img1, img2],
        sampling_params=SamplingParams(max_tokens=64, temperature=0.7),
    )

    assert "text" in result
    assert "token_ids" in result
    assert len(result["text"]) > 0
    assert len(result["token_ids"]) > 0
