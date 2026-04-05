"""VLM 与 HuggingFace 首 token logits 深度对比诊断脚本（v4）。

运行方式：
    source /opt/conda/bin/activate ramc && python /app/minillm/benchmarks/diagnose_vlm_logits.py

依赖：需要 GPU，需要下载好的 Qwen2.5-VL-3B-Instruct 模型权重。
"""

from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from minillm.engine.vlm_engine import VLMEngine
from minillm.sampling_params import SamplingParams
from minillm.utils.context import set_context, reset_context

MODEL_PATH = "/app/models/Qwen2.5-VL-3B-Instruct"
IMAGE_PATH = "/app/minillm/img/image.png"
PROMPT = "描述这张图片"


def load_image() -> Image.Image:
    return Image.open(IMAGE_PATH).convert("RGB")


def hf_get_vision_embeds(model, pixel_values, grid_thw):
    """获取 HF 视觉编码器输出的合并 tensor。"""
    out = model.model.visual(pixel_values.to(torch.bfloat16), grid_thw=grid_thw, return_dict=True)
    po = out.pooler_output
    if isinstance(po, torch.Tensor):
        return po
    if isinstance(po, (list, tuple)):
        return torch.cat([t for t in po], dim=0)
    raise TypeError(f"Unexpected pooler_output type: {type(po)}")


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

    # 1. 视觉编码器对比
    print("\n" + "=" * 80)
    print("1. 视觉编码器输出对比")
    print("=" * 80)

    with torch.inference_mode():
        minillm_vision = engine.model.visual(pixel_values, image_grid_thw)
        hf_vision = hf_get_vision_embeds(hf_model, hf_inputs["pixel_values"], hf_inputs["image_grid_thw"])

    print(f"minillm vision: shape={minillm_vision.shape}, mean={minillm_vision.float().mean():.6f}, std={minillm_vision.float().std():.6f}")
    print(f"HF vision:      shape={hf_vision.shape}, mean={hf_vision.float().mean():.6f}, std={hf_vision.float().std():.6f}")

    if minillm_vision.shape == hf_vision.shape:
        vision_diff = (minillm_vision.float() - hf_vision.float()).abs()
        print(f"vision 最大绝对差: {vision_diff.max().item():.6f}")
        print(f"vision 平均绝对差: {vision_diff.mean().item():.6f}")
        # cosine similarity
        cos = torch.nn.functional.cosine_similarity(
            minillm_vision.float().flatten().unsqueeze(0),
            hf_vision.float().flatten().unsqueeze(0),
        )
        print(f"vision cosine similarity: {cos.item():.8f}")
    else:
        print(f"vision 形状不同!")

    # 2. patch_embed 对比
    print("\n" + "=" * 80)
    print("2. patch_embed 输出对比")
    print("=" * 80)

    with torch.inference_mode():
        minillm_pe = engine.model.visual.patch_embed(pixel_values)
        hf_pe = hf_model.model.visual.patch_embed(hf_inputs["pixel_values"].to(torch.bfloat16))

    print(f"minillm PE: shape={minillm_pe.shape}, mean={minillm_pe.float().mean():.6f}")
    print(f"HF PE:      shape={hf_pe.shape}, mean={hf_pe.float().mean():.6f}")

    if minillm_pe.shape == hf_pe.shape:
        pe_diff = (minillm_pe.float() - hf_pe.float()).abs()
        print(f"PE 最大绝对差: {pe_diff.max().item():.6f}")
        print(f"PE 平均绝对差: {pe_diff.mean().item():.6f}")
    else:
        print(f"PE 形状不同!")

    # 3. rot_pos_emb / window_index
    print("\n" + "=" * 80)
    print("3. rot_pos_emb / window_index")
    print("=" * 80)

    with torch.inference_mode():
        minillm_rope = engine.model.visual.rot_pos_emb(image_grid_thw)
        hf_rope = hf_model.model.visual.rot_pos_emb(hf_inputs["image_grid_thw"])

    if minillm_rope.shape == hf_rope.shape:
        rope_diff = (minillm_rope.float() - hf_rope.float()).abs()
        print(f"rot_pos_emb 最大绝对差: {rope_diff.max().item():.8f}")
    else:
        print(f"rot_pos_emb 形状不同: {minillm_rope.shape} vs {hf_rope.shape}")

    with torch.inference_mode():
        minillm_wi, _ = engine.model.visual.get_window_index(image_grid_thw)
        hf_wi, _ = hf_model.model.visual.get_window_index(hf_inputs["image_grid_thw"])

    if minillm_wi.shape == hf_wi.shape:
        wi_same = (minillm_wi == hf_wi).all().item()
        print(f"window_index 完全相同: {wi_same}")
        if not wi_same:
            print(f"  不同数: {(minillm_wi != hf_wi).sum().item()}")
    else:
        print(f"window_index 形状不同: {minillm_wi.shape} vs {hf_wi.shape}")

    # 4. 配置对比
    print("\n" + "=" * 80)
    print("4. 视觉编码器配置对比")
    print("=" * 80)
    print(f"minillm fullatt_block_indexes: {engine.model.visual.fullatt_block_indexes}")
    print(f"HF fullatt_block_indexes:      {hf_model.model.visual.fullatt_block_indexes}")
    print(f"minillm window_size: {engine.model.visual.window_size}")
    print(f"HF window_size:      {hf_model.model.visual.window_size}")

    # 5. prefill logits 对比
    print("\n" + "=" * 80)
    print("5. 全链路 prefill logits 对比")
    print("=" * 80)

    engine.model._reset_cache()
    seq_len = input_ids.shape[1]
    positions = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=input_ids.device)
    seq = engine._build_vlm_sequence(input_ids, SamplingParams(temperature=0.0, max_tokens=1))
    _, _, slot_mapping = engine._prepare_prefill_context(seq, input_ids)
    set_context(
        is_prefill=True,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=seq_len,
        max_seqlen_k=seq_len,
        slot_mapping=slot_mapping,
    )
    try:
        with torch.inference_mode():
            minillm_logits = engine.model.forward(
                input_ids=input_ids,
                positions=positions,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                use_cache=False,
            )
    finally:
        reset_context()

    with torch.inference_mode():
        hf_outputs = hf_model(**hf_inputs)
    hf_logits = hf_outputs.logits

    minillm_top5 = torch.topk(minillm_logits[:, -1, :], 5)
    hf_top5 = torch.topk(hf_logits[:, -1, :], 5)
    print(f"minillm 首 token: {minillm_top5.indices[0,0].item()} ({processor.decode([minillm_top5.indices[0,0].item()])})")
    print(f"HF 首 token:     {hf_top5.indices[0,0].item()} ({processor.decode([hf_top5.indices[0,0].item()])})")
    print(f"minillm top5: ids={minillm_top5.indices[0].tolist()}")
    print(f"HF top5:     ids={hf_top5.indices[0].tolist()}")

    # 6. embed_tokens 对比
    print("\n" + "=" * 80)
    print("6. embed_tokens 对比")
    print("=" * 80)

    minillm_emb = engine.model.language_model.model.embed_tokens(input_ids)
    with torch.inference_mode():
        hf_emb = hf_model.model.get_input_embeddings()(hf_inputs["input_ids"])

    if minillm_emb.shape == hf_emb.shape:
        emb_diff = (minillm_emb.float() - hf_emb.float()).abs()
        print(f"embed 最大绝对差: {emb_diff.max().item():.8f}")
        print(f"embed 平均绝对差: {emb_diff.mean().item():.8f}")
    else:
        print(f"embed 形状不同!")

    # 7. 对比 get_rope_index 调用方式差异
    print("\n" + "=" * 80)
    print("7. HF get_rope_index 需要 mm_token_type_ids")
    print("=" * 80)

    if "mm_token_type_ids" in hf_inputs:
        print(f"HF mm_token_type_ids shape: {hf_inputs['mm_token_type_ids'].shape}")
        print(f"HF mm_token_type_ids[0,:20]: {hf_inputs['mm_token_type_ids'][0,:20].tolist()}")
    else:
        print("HF inputs 中无 mm_token_type_ids")

    del engine
    del hf_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
