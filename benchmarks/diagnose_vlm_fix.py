"""VLM cos/sin 修正后逐层对比诊断脚本（v7）。

使用 HF 的 inv_freq 重建 minillm 的 VisionRotaryEmbedding，检查是否因为 inv_freq 的 bfloat16 精度损失导致级联放大。

运行方式：
    source /opt/conda/bin/activate ramc && python /app/minillm/benchmarks/diagnose_vlm_fix.py

依赖：需要 GPU，需要下载好的 Qwen2.5-VL-3B-Instruct 模型权重。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from minillm.engine.vlm_engine import VLMEngine
from minillm.vision.vision_encoder import VisionRotaryEmbedding
from minillm.sampling_params import SamplingParams

MODEL_PATH = "/app/models/Qwen2.5-VL-3B-Instruct"
IMAGE_PATH = "/app/minillm/img/image.png"
PROMPT = "描述这张图片"


def load_image() -> Image.Image:
    return Image.open(IMAGE_PATH).convert("RGB")


def main() -> None:
    image = load_image()

    print("初始化 minillm ...")
    engine = VLMEngine(MODEL_PATH, device="cuda")
    input_ids, pixel_values, image_grid_thw = engine._prepare_inputs(PROMPT, [image], True)

    print("初始化 HF ...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    hf_inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to("cuda")

    hf_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="cuda",
    )

    smu = engine.model.visual.spatial_merge_unit
    seq_len_px = pixel_values.shape[0]

    # 1. 用 HF 的 inv_freq 创建 minillm 的 VisionRotaryEmbedding
    print("\n" + "=" * 80)
    print("1. 用 HF inv_freq 替换 minillm 的 rot_pos_emb")
    print("=" * 80)

    hf_inv_freq = hf_model.model.visual.rotary_pos_emb.inv_freq.clone()
    head_dim = engine.model.visual.config.hidden_size // engine.model.visual.config.num_heads
    patched_rope = VisionRotaryEmbedding(head_dim // 2, theta=10000.0)
    patched_rope.inv_freq = hf_inv_freq.to(patched_rope.inv_freq.device)

    with torch.inference_mode():
        minillm_rope = engine.model.visual.rot_pos_emb(image_grid_thw)
        hf_rope = hf_model.model.visual.rot_pos_emb(hf_inputs["image_grid_thw"])
        patched_rope_out = patched_rope(int(image_grid_thw[:, 1:].max()))

    print(f"minillm inv_freq first 5: {engine.model.visual.rotary_pos_emb.inv_freq[:5].tolist()}")
    print(f"HF inv_freq first 5:     {hf_inv_freq[:5].tolist()}")
    print(f"patched vs HF rot_pos_emb match: {(patched_rope_out == hf_model.model.visual.rotary_pos_emb(int(image_grid_thw[:, 1:].max()))).all().item()}")

    # 2. 用 patched rot_pos_emb 做逐层对比
    print("\n" + "=" * 80)
    print("2. 逐层对比：使用 patched inv_freq")
    print("=" * 80)

    with torch.inference_mode():
        window_index, cu_window_seqlens = engine.model.visual.get_window_index(image_grid_thw)
        cu_window_seqlens_unique = torch.unique_consecutive(cu_window_seqlens.to(device="cuda"))

        # 用 patched rot_pos_emb 构建 cos/sin
        patched_rope_seq = patched_rope_out.reshape(seq_len_px // smu, smu, -1)
        patched_rope_w = patched_rope_seq[window_index, :, :].reshape(seq_len_px, -1)
        patched_emb = torch.cat((patched_rope_w, patched_rope_w), dim=-1)
        patched_cos = patched_emb.cos()
        patched_sin = patched_emb.sin()

        # HF 的 cos/sin
        hf_cos = hf_rope.reshape(seq_len_px // smu, smu, -1)
        hf_cos_w = hf_cos[window_index, :, :].reshape(seq_len_px, -1)
        hf_emb = torch.cat((hf_cos_w, hf_cos_w), dim=-1)
        hf_cos_actual = hf_emb.cos()
        hf_sin_actual = hf_emb.sin()

        print(f"patched cos vs HF cos max_diff: {(patched_cos.float() - hf_cos_actual.float()).abs().max().item():.10f}")
        print(f"patched sin vs HF sin max_diff: {(patched_sin.float() - hf_sin_actual.float()).abs().max().item():.10f}")

        # 共同起点
        minillm_h = engine.model.visual.patch_embed(pixel_values)
        minillm_h = minillm_h.reshape(seq_len_px // smu, smu, -1)
        minillm_h = minillm_h[window_index, :, :]
        minillm_h = minillm_h.reshape(seq_len_px, -1)

        hf_h = hf_model.model.visual.patch_embed(hf_inputs["pixel_values"].to(torch.bfloat16))
        hf_h = hf_h.reshape(seq_len_px // smu, smu, -1)
        hf_h = hf_h[window_index, :, :]
        hf_h = hf_h.reshape(seq_len_px, -1)

        cu_seqlens_v = torch.repeat_interleave(
            image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32).to(device="cuda")
        cu_seqlens_v = F.pad(cu_seqlens_v, (1, 0), value=0)

        for layer_num in range(len(engine.model.visual.blocks)):
            cu_now = cu_seqlens_v if layer_num in engine.model.visual.fullatt_block_indexes else cu_window_seqlens_unique

            # minillm blocks + patched cos/sin (should now match HF)
            minillm_h = engine.model.visual.blocks[layer_num](
                minillm_h.cuda(), cu_seqlens=cu_now,
                position_embeddings=(patched_cos.cuda(), patched_sin.cuda()),
            )

            hf_h = hf_model.model.visual.blocks[layer_num](
                hf_h.cuda(), cu_seqlens=cu_now,
                position_embeddings=(hf_cos_actual.cuda(), hf_sin_actual.cuda()),
            )

            if layer_num % 8 == 0 or layer_num == len(engine.model.visual.blocks) - 1:
                diff = (minillm_h.float() - hf_h.float()).abs()
                print(f"  L{layer_num:2d}: minillm mean={minillm_h.float().mean():.6f} HF mean={hf_h.float().mean():.6f} max_diff={diff.max().item():.4f} mean_diff={diff.mean().item():.6f}")

    # 3. merger
    print("\n" + "=" * 80)
    print("3. Merger 输出对比（patched inv_freq）")
    print("=" * 80)

    with torch.inference_mode():
        reverse_indices = torch.argsort(window_index.to("cuda"))
        minillm_merged = engine.model.visual.merger(minillm_h)
        minillm_merged = minillm_merged[reverse_indices, :]

        hf_merged = hf_model.model.visual.merger(hf_h)
        hf_merged = hf_merged[reverse_indices, :]

    m_diff = (minillm_merged.float() - hf_merged.float()).abs()
    print(f"merged 最大绝对差: {m_diff.max().item():.6f}")
    print(f"merged 平均绝对差: {m_diff.mean().item():.6f}")

    # 4. 完整 forward 对比
    print("\n" + "=" * 80)
    print("4. 完整 VLM forward logits 对比（需要修复 inv_freq 后再跑）")
    print("=" * 80)

    print("当前 minillm 还使用旧 inv_freq, 以上结果说明:")
    print("inv_freq 的 bf16 精度差异是 cos/sin 差异的根源")
    print("修复方法: 在 VisionRotaryEmbedding 初始化时用 float32 计算 inv_freq")

    del engine
    del hf_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
