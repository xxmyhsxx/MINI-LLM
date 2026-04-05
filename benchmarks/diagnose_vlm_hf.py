"""VLM 与 HuggingFace 中间状态对比诊断脚本。

运行方式：
    source /opt/conda/bin/activate ramc && python /app/minillm/benchmarks/diagnose_vlm_hf.py

依赖：需要 GPU，需要下载好的 Qwen2.5-VL-3B-Instruct 模型权重。
"""

from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from minillm.engine.vlm_engine import VLMEngine
from minillm.sampling_params import SamplingParams

MODEL_PATH = "/app/models/Qwen2.5-VL-3B-Instruct"
IMAGE_PATH = "/app/minillm/img/image.png"
PROMPT = "描述这张图片"


def load_image() -> Image.Image:
    """加载测试图像。"""
    return Image.open(IMAGE_PATH).convert("RGB")


def main() -> None:
    """对比 minillm 与 HuggingFace 的中间状态。"""
    image = load_image()

    # ================================================================
    # 1. 图像预处理对比
    # ================================================================
    print("=" * 80)
    print("1. 图像预处理对比")
    print("=" * 80)

    # minillm 预处理
    minillm_engine = VLMEngine(MODEL_PATH, device="cuda")
    minillm_input_ids, minillm_pixel_values, minillm_image_grid_thw = minillm_engine._prepare_inputs(PROMPT, [image], True)

    # HF 预处理
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    hf_inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to("cuda")

    # 对比 pixel_values
    print(f"\nminillm pixel_values shape: {minillm_pixel_values.shape}")
    print(f"HF pixel_values shape:      {hf_inputs['pixel_values'].shape}")
    if minillm_pixel_values.shape == hf_inputs["pixel_values"].shape:
        pixel_diff = (minillm_pixel_values.float() - hf_inputs["pixel_values"].float()).abs()
        print(f"pixel_values 最大绝对差: {pixel_diff.max().item():.6f}")
        print(f"pixel_values 平均绝对差: {pixel_diff.mean().item():.6f}")
    else:
        print("pixel_values 形状不同，跳过逐元素对比")

    # 对比 image_grid_thw
    print(f"\nminillm image_grid_thw: {minillm_image_grid_thw.tolist()}")
    if "image_grid_thw" in hf_inputs:
        print(f"HF image_grid_thw:     {hf_inputs['image_grid_thw'].tolist()}")
    else:
        print("HF inputs 中无 image_grid_thw")

    # 对比 input_ids
    print(f"\nminillm input_ids shape: {minillm_input_ids.shape}")
    print(f"HF input_ids shape:     {hf_inputs['input_ids'].shape}")
    minillm_id_list = minillm_input_ids[0].tolist()
    hf_id_list = hf_inputs["input_ids"][0].tolist()
    print(f"minillm 前 20 tokens: {minillm_id_list[:20]}")
    print(f"HF 前 20 tokens:     {hf_id_list[:20]}")
    print(f"minillm 尾 20 tokens: {minillm_id_list[-20:]}")
    print(f"HF 尾 20 tokens:     {hf_id_list[-20:]}")

    # 统计 image token 数
    image_token_id = 151655
    minillm_img_count = sum(1 for t in minillm_id_list if t == image_token_id)
    hf_img_count = sum(1 for t in hf_id_list if t == image_token_id)
    print(f"\nminillm image token 数: {minillm_img_count}")
    print(f"HF image token 数:     {hf_img_count}")

    # ================================================================
    # 2. 视觉编码器输出对比
    # ================================================================
    print("\n" + "=" * 80)
    print("2. 视觉编码器输出对比")
    print("=" * 80)

    with torch.inference_mode():
        minillm_vision_embeds = minillm_engine.model.visual(
            minillm_pixel_values, minillm_image_grid_thw
        )

    print(f"minillm vision_embeds shape: {minillm_vision_embeds.shape}")
    print(f"minillm vision_embeds mean:  {minillm_vision_embeds.float().mean().item():.6f}")
    print(f"minillm vision_embeds std:   {minillm_vision_embeds.float().std().item():.6f}")

    # 3. mRoPE position_ids / rope_deltas 对比
    print("\n" + "=" * 80)
    print("3. mRoPE 位置编码对比")
    print("=" * 80)

    minillm_pos_ids, minillm_deltas = minillm_engine.model.get_rope_index(
        minillm_input_ids, minillm_image_grid_thw
    )
    print(f"minillm position_ids shape: {minillm_pos_ids.shape}")
    print(f"minillm rope_deltas:        {minillm_deltas.item()}")
    print(f"minillm position_ids[0,0,:10]: {minillm_pos_ids[0, 0, :10].tolist()}")
    print(f"minillm position_ids[1,0,:10]: {minillm_pos_ids[1, 0, :10].tolist()}")
    print(f"minillm position_ids[2,0,:10]: {minillm_pos_ids[2, 0, :10].tolist()}")

    # 4. Prefill logits 对比（使用 minillm 自己的 generate 来做 1-step）
    print("\n" + "=" * 80)
    print("4. 首 token logits / 生成结果对比")
    print("=" * 80)

    # minillm: 温度 0 生成
    result = minillm_engine.generate(
        PROMPT, [image],
        SamplingParams(temperature=0.0, top_k=0, top_p=1.0, max_tokens=32),
    )
    print(f"minillm 生成结果: {result['text']}")

    # HF: 温度 0 生成
    hf_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="cuda",
    )
    with torch.inference_mode():
        hf_outputs = hf_model.generate(**hf_inputs, max_new_tokens=32, temperature=None, do_sample=False)
    hf_text = processor.decode(hf_outputs[0][hf_inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"HF 生成结果:     {hf_text}")

    del minillm_engine
    del hf_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
