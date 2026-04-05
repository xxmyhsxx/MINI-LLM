"""VLM rotary position embedding 与 cos/sin 对比诊断脚本（v6）。

运行方式：
    source /opt/conda/bin/activate ramc && python /app/minillm/benchmarks/diagnose_vlm_rope.py

依赖：需要 GPU，需要下载好的 Qwen2.5-VL-3B-Instruct 模型权重。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from minillm.engine.vlm_engine import VLMEngine
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

    # 1. inv_freq 对比
    print("\n" + "=" * 80)
    print("1. VisionRotaryEmbedding.inv_freq 对比")
    print("=" * 80)

    minillm_inv = engine.model.visual.rotary_pos_emb.inv_freq
    hf_inv = hf_model.model.visual.rotary_pos_emb.inv_freq
    print(f"minillm inv_freq first 5: {minillm_inv[:5].tolist()}")
    print(f"HF inv_freq first 5:     {hf_inv[:5].tolist()}")
    inv_diff = (minillm_inv.float() - hf_inv.float()).abs()
    print(f"inv_freq 最大绝对差: {inv_diff.max().item():.12f}")

    # 2. rot_pos_emb 对比
    print("\n" + "=" * 80)
    print("2. rot_pos_emb 输出对比")
    print("=" * 80)

    with torch.inference_mode():
        minillm_rope = engine.model.visual.rot_pos_emb(image_grid_thw)
        hf_rope = hf_model.model.visual.rot_pos_emb(hf_inputs["image_grid_thw"])

    rope_diff = (minillm_rope.float() - hf_rope.float()).abs()
    print(f"rot_pos_emb 最大绝对差: {rope_diff.max().item():.6f}")
    print(f"rot_pos_emb 平均绝对差: {rope_diff.mean().item():.6f}")

    # 3. cos/sin 对比（windowed 排序后）
    print("\n" + "=" * 80)
    print("3. cos/sin 对比（windowed）")
    print("=" * 80)

    with torch.inference_mode():
        window_index, cu_window_seqlens = engine.model.visual.get_window_index(image_grid_thw)

        # minillm
        minillm_rope_seq = minillm_rope.reshape(seq_len_px // smu, smu, -1)
        minillm_rope_w = minillm_rope_seq[window_index, :, :].reshape(seq_len_px, -1)
        minillm_emb = torch.cat((minillm_rope_w, minillm_rope_w), dim=-1)
        minillm_cos = minillm_emb.cos()
        minillm_sin = minillm_emb.sin()

        # HF
        hf_rope_seq = hf_rope.reshape(seq_len_px // smu, smu, -1)
        hf_rope_w = hf_rope_seq[window_index, :, :].reshape(seq_len_px, -1)
        hf_emb = torch.cat((hf_rope_w, hf_rope_w), dim=-1)
        hf_cos = hf_emb.cos()
        hf_sin = hf_emb.sin()

    cos_diff = (minillm_cos.float() - hf_cos.float()).abs()
    sin_diff = (minillm_sin.float() - hf_sin.float()).abs()
    print(f"cos 最大绝对差: {cos_diff.max().item():.6f}")
    print(f"sin 最大绝对差: {sin_diff.max().item():.6f}")
    print(f"cos 平均绝对差: {cos_diff.mean().item():.6f}")
    print(f"sin 平均绝对差: {sin_diff.mean().item():.6f}")

    # 4. 逐层用 HF 的 cos/sin 跑 minillm blocks
    print("\n" + "=" * 80)
    print("4. 逐层对比：minillm blocks + minillm cos/sin vs HF blocks + HF cos/sin")
    print("=" * 80)

    with torch.inference_mode():
        # 共同起点: patch_embed 后 window 排序
        minillm_h = engine.model.visual.patch_embed(pixel_values)
        minillm_h = minillm_h.reshape(seq_len_px // smu, smu, -1)
        minillm_h = minillm_h[window_index, :, :]
        minillm_h = minillm_h.reshape(seq_len_px, -1)

        hf_h = hf_model.model.visual.patch_embed(hf_inputs["pixel_values"].to(torch.bfloat16))
        hf_h = hf_h.reshape(seq_len_px // smu, smu, -1)
        hf_h = hf_h[window_index, :, :]
        hf_h = hf_h.reshape(seq_len_px, -1)

        pe_diff = (minillm_h.float() - hf_h.float()).abs()
        print(f"After patch_embed+window: max_diff={pe_diff.max().item():.8f}")

        cu_seqlens_v = torch.repeat_interleave(
            image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32).to(device="cuda")
        cu_seqlens_v = F.pad(cu_seqlens_v, (1, 0), value=0)
        cu_window_seqlens_unique = torch.unique_consecutive(cu_window_seqlens.to(device="cuda"))

        for layer_num in range(len(engine.model.visual.blocks)):
            cu_now = cu_seqlens_v if layer_num in engine.model.visual.fullatt_block_indexes else cu_window_seqlens_unique

            minillm_h = engine.model.visual.blocks[layer_num](
                minillm_h.cuda(), cu_seqlens=cu_now,
                position_embeddings=(minillm_cos.cuda(), minillm_sin.cuda()),
            )

            cu_now_hf = cu_seqlens_v if layer_num in hf_model.model.visual.fullatt_block_indexes else cu_window_seqlens_unique
            hf_h = hf_model.model.visual.blocks[layer_num](
                hf_h.cuda(), cu_seqlens=cu_now_hf,
                position_embeddings=(hf_cos.cuda(), hf_sin.cuda()),
            )

            if layer_num % 8 == 0 or layer_num == len(engine.model.visual.blocks) - 1:
                diff = (minillm_h.float() - hf_h.float()).abs()
                print(f"  L{layer_num:2d}: minillm mean={minillm_h.float().mean():.6f} HF mean={hf_h.float().mean():.6f} max_diff={diff.max().item():.4f} mean_diff={diff.mean().item():.4f}")

    # 5. merger
    print("\n" + "=" * 80)
    print("5. Merger 输出对比")
    print("=" * 80)

    with torch.inference_mode():
        reverse_indices = torch.argsort(window_index.to("cuda"))
        hf_rev = torch.argsort(window_index.to("cuda"))

        minillm_merged = engine.model.visual.merger(minillm_h)
        minillm_merged = minillm_merged[reverse_indices, :]

        hf_merged = hf_model.model.visual.merger(hf_h)
        hf_merged = hf_merged[hf_rev, :]

    m_diff = (minillm_merged.float() - hf_merged.float()).abs()
    print(f"merged 最大绝对差: {m_diff.max().item():.6f}")
    print(f"merged 平均绝对差: {m_diff.mean().item():.6f}")
    print(f"merged cosine sim: {F.cosine_similarity(minillm_merged.float()[0:1], hf_merged.float()[0:1]).item():.8f}")

    del engine
    del hf_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
