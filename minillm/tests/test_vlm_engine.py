"""测试 VLM Engine 端到端推理。

运行方式：
    pytest -v minillm/tests/test_vlm_engine.py

依赖：需要 GPU，需要下载好的 Qwen2.5-VL-3B-Instruct 模型权重。
"""

import numpy as np
import pytest
import torch
from PIL import Image

from minillm.engine.vlm_engine import VLMEngine
from minillm.sampling_params import SamplingParams


@pytest.mark.gpu
def test_vlm_engine_text_only():
    """测试纯文本推理（无图像）。"""
    engine = VLMEngine("/app/models/Qwen2.5-VL-3B-Instruct")

    result = engine.generate(
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

    print(f"Generated text: {result['text']}")


@pytest.mark.gpu
def test_vlm_engine_with_image():
    """测试图像+文本推理。"""
    engine = VLMEngine("/app/models/Qwen2.5-VL-3B-Instruct")

    # 创建测试图像
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

    result = engine.generate(
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

    print(f"Generated text: {result['text']}")


@pytest.mark.gpu
def test_vlm_engine_multiple_images():
    """测试多图像推理。"""
    engine = VLMEngine("/app/models/Qwen2.5-VL-3B-Instruct")

    # 创建两张测试图像
    img1 = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    img2 = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

    result = engine.generate(
        prompt="比较这两张图片的差异。",
        images=[img1, img2],
        sampling_params=SamplingParams(max_tokens=64, temperature=0.7),
    )

    assert "text" in result
    assert "token_ids" in result
    assert len(result["text"]) > 0

    print(f"Generated text: {result['text']}")


if __name__ == "__main__":
    print("Testing VLM Engine...")
    test_vlm_engine_text_only()
    test_vlm_engine_with_image()
    test_vlm_engine_multiple_images()
    print("\nAll VLM Engine tests passed!")
