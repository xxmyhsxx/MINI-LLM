"""测试 Qwen2.5-VL 图像预处理、视觉编码器以及与 HuggingFace 的语义对齐。

运行方式：
    pytest -v minillm/tests/test_qwen2_5_vl.py

依赖：需要 GPU，需要 `/app/models/Qwen2.5-VL-3B-Instruct` 和 `/app/minillm/img/image.png`。
"""

import torch
import numpy as np
from PIL import Image

from minillm.engine.vlm_engine import VLMEngine
from minillm.vision.image_processor import Qwen2VLImageProcessor
from minillm.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration


def test_image_processor():
    """Test image processor output matches transformers."""
    from transformers import Qwen2VLImageProcessorFast

    # Create test image
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

    # Our implementation
    our_processor = Qwen2VLImageProcessor()
    our_result = our_processor([img])

    # HuggingFace implementation
    hf_processor = Qwen2VLImageProcessorFast.from_pretrained('/app/models/Qwen2.5-VL-3B-Instruct')
    hf_result = hf_processor(images=[img], return_tensors='pt')

    # Compare shapes
    assert our_result['pixel_values'].shape == hf_result['pixel_values'].shape, \
        f"pixel_values shape mismatch: {our_result['pixel_values'].shape} vs {hf_result['pixel_values'].shape}"

    assert our_result['image_grid_thw'].shape == hf_result['image_grid_thw'].shape, \
        f"image_grid_thw shape mismatch: {our_result['image_grid_thw'].shape} vs {hf_result['image_grid_thw'].shape}"

    # Compare values (should be very close due to same preprocessing)
    assert torch.allclose(our_result['pixel_values'], hf_result['pixel_values'], atol=1e-4), \
        "pixel_values values don't match"

    assert torch.equal(our_result['image_grid_thw'], hf_result['image_grid_thw']), \
        "image_grid_thw values don't match"

    print("✓ Image processor test passed")


def test_vision_encoder():
    """Test vision encoder output shape."""
    from minillm.vision.vision_encoder import Qwen2_5_VisionTransformer

    # Create mock config
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

    config = VisionConfig()
    encoder = Qwen2_5_VisionTransformer(config).to("cuda").to(torch.bfloat16)

    # Test input
    pixel_values = torch.randn(256, 1176, dtype=torch.bfloat16, device="cuda")  # 16x16 grid
    image_grid_thw = torch.tensor([[1, 16, 16]], device="cuda")

    # Forward pass
    output = encoder(pixel_values, image_grid_thw)

    # Check output shape
    # After spatial merge (2x2), 16x16 -> 8x8 = 64 patches
    expected_seq_len = 64
    assert output.shape == (expected_seq_len, config.out_hidden_size), \
        f"Output shape mismatch: {output.shape} vs ({expected_seq_len}, {config.out_hidden_size})"

    print("✓ Vision encoder test passed")


def test_vlm_engine_output_is_image_grounded():
    """测试 VLM 输出与图片语义相关，而不是泛化风景描述。"""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration as HFQwen2_5_VL

    model_path = "/app/models/Qwen2.5-VL-3B-Instruct"
    image_path = "/app/minillm/img/image.png"
    prompt = "描述这张图片"

    image = Image.open(image_path).convert("RGB")

    # HuggingFace 参考输出
    hf_model = HFQwen2_5_VL.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
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

    # minillm 输出
    engine = VLMEngine(model_path, device="cuda")
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

    del engine, hf_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print("Testing Qwen2.5-VL implementation...")
    test_image_processor()
    test_vision_encoder()
    print("\nAll tests passed!")
