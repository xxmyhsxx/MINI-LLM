"""测试 Qwen2.5-VL 图像预处理、视觉编码器以及与 HuggingFace 的语义对齐。"""

import gc

import numpy as np
import pytest
import torch
from PIL import Image

from minillm.engine.vlm_engine import VLMEngine
from minillm.vision.image_processor import Qwen2VLImageProcessor

MODEL_PATH = "/app/models/Qwen2.5-VL-3B-Instruct"
IMAGE_PATH = "/app/minillm/img/image.png"


def _cleanup_cuda() -> None:
    """清理测试残留的 CUDA 显存。"""
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(autouse=True)
def cleanup_cuda_between_tests():
    """在每个测试后清理缓存，避免语义对齐测试累计占满显存。"""
    yield
    _cleanup_cuda()


def test_image_processor():
    """验证图像预处理输出与 Transformers 对齐。"""
    from transformers import Qwen2VLImageProcessorFast

    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

    our_processor = Qwen2VLImageProcessor()
    our_result = our_processor([img])

    hf_processor = Qwen2VLImageProcessorFast.from_pretrained(MODEL_PATH)
    hf_result = hf_processor(images=[img], return_tensors="pt")

    assert our_result["pixel_values"].shape == hf_result["pixel_values"].shape
    assert our_result["image_grid_thw"].shape == hf_result["image_grid_thw"].shape
    assert torch.allclose(our_result["pixel_values"], hf_result["pixel_values"], atol=1e-4)
    assert torch.equal(our_result["image_grid_thw"], hf_result["image_grid_thw"])


@pytest.mark.gpu
def test_vision_encoder():
    """验证视觉编码器输出形状正确。"""
    from minillm.vision.vision_encoder import Qwen2_5_VisionTransformer

    class VisionConfig:
        hidden_size = 1536
        num_heads = 16
        intermediate_size = 8960
        depth = 32
        patch_size = 14
        temporal_patch_size = 2
        in_channels = 3
        spatial_merge_size = 2
        out_hidden_size = 3584
        fullatt_block_indexes = [0, 4, 8, 12, 16, 20, 24, 28]
        window_size = 14

    encoder = None
    pixel_values = None
    image_grid_thw = None
    try:
        encoder = Qwen2_5_VisionTransformer(VisionConfig()).to("cuda").to(torch.bfloat16)
        pixel_values = torch.randn(256, 1176, dtype=torch.bfloat16, device="cuda")
        image_grid_thw = torch.tensor([[1, 16, 16]], device="cuda")
        output = encoder(pixel_values, image_grid_thw)
        assert output.shape == (64, VisionConfig.out_hidden_size)
    finally:
        del encoder, pixel_values, image_grid_thw


@pytest.mark.gpu
def test_vlm_engine_output_is_image_grounded():
    """验证 minillm 输出与图片语义相关，而不是泛化风景描述。"""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration as HFQwen2_5_VL

    prompt = "描述这张图片"
    image = Image.open(IMAGE_PATH).convert("RGB")
    hf_model = None
    processor = None
    hf_inputs = None
    hf_output_ids = None
    hf_new_tokens = None
    engine = None
    try:
        hf_model = HFQwen2_5_VL.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        hf_inputs = processor(text=[text], images=[image], return_tensors="pt")
        hf_inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in hf_inputs.items()}
        hf_output_ids = hf_model.generate(**hf_inputs, max_new_tokens=64)
        hf_new_tokens = hf_output_ids[0, hf_inputs["input_ids"].shape[1]:]
        hf_text = processor.decode(hf_new_tokens, skip_special_tokens=True)

        # 先释放 HF 参考模型，再加载 minillm，避免双模型同时驻留导致显存溢出。
        hf_model = None
        processor = None
        hf_inputs = None
        hf_output_ids = None
        hf_new_tokens = None
        _cleanup_cuda()

        engine = VLMEngine(MODEL_PATH, device="cuda")
        result = engine.generate(prompt, [image])
        minillm_text = result["text"]

        assert minillm_text.strip() != ""
        assert result["token_ids"] and result["token_ids"][0] != 151645

        keywords = ["湖", "水", "岩", "石", "山", "雪", "天空", "蓝"]
        hf_hits = {kw for kw in keywords if kw in hf_text}
        minillm_hits = {kw for kw in keywords if kw in minillm_text}
        assert hf_hits, f"HuggingFace 参考输出未命中关键词，输出为: {hf_text}"
        assert len(hf_hits & minillm_hits) >= 2, (
            f"minillm 输出与图片语义关联不足。HF={hf_text!r}, minillm={minillm_text!r}"
        )
    finally:
        del engine, hf_model, processor, hf_inputs, hf_output_ids, hf_new_tokens
        _cleanup_cuda()
