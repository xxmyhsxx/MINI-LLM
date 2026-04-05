"""Qwen3 模型单元测试（需要 GPU）。

测试 Qwen3 模型创建、Q/K RMSNorm、权重加载和端到端推理。
与 Qwen2 的主要差异：可选的 Q/K RMSNorm、rope_theta=1000000、qkv_bias=False。

运行方式：
    pytest -v minillm/tests/test_qwen3.py

依赖：需要 GPU，需要 Qwen3-0.6B 模型权重（/app/models/Qwen3-0.6B）。
"""

import gc

import pytest
import torch
from transformers import AutoConfig

from minillm.sampling_params import SamplingParams


QWEN3_MODEL_PATH = "/app/models/Qwen3-0.6B"


@pytest.mark.gpu
def test_qwen3_model_creation():
    """测试 Qwen3 模型从 config 正确构造。"""
    from minillm.models.qwen3 import Qwen3ForCausalLM

    config = AutoConfig.from_pretrained(QWEN3_MODEL_PATH, trust_remote_code=True)
    model = Qwen3ForCausalLM(config).cuda()
    assert hasattr(model, "model")
    assert hasattr(model, "lm_head")
    assert len(model.model.layers) == config.num_hidden_layers
    del model
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.gpu
def test_qwen3_qknorm_applied_when_no_bias():
    """测试 Qwen3 的 Q/K RMSNorm 在 qkv_bias=False 时生效。"""
    from minillm.models.qwen3 import Qwen3Attention

    config = AutoConfig.from_pretrained(QWEN3_MODEL_PATH, trust_remote_code=True)
    # Qwen3 默认 attention_bias=False，应启用 Q/K RMSNorm
    assert getattr(config, "attention_bias", True) is False
    attn = Qwen3Attention(
        hidden_size=config.hidden_size,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_dim=getattr(config, "head_dim", None),
        qkv_bias=False,
        rms_norm_eps=config.rms_norm_eps,
    ).cuda()
    assert hasattr(attn, "q_norm"), "qkv_bias=False 时应有 q_norm"
    assert hasattr(attn, "k_norm"), "qkv_bias=False 时应有 k_norm"
    del attn
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.gpu
def test_qwen3_no_qknorm_when_bias():
    """测试 Qwen3 在 qkv_bias=True 时不使用 Q/K RMSNorm。"""
    from minillm.models.qwen3 import Qwen3Attention

    config = AutoConfig.from_pretrained(QWEN3_MODEL_PATH, trust_remote_code=True)
    attn = Qwen3Attention(
        hidden_size=config.hidden_size,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_dim=getattr(config, "head_dim", None),
        qkv_bias=True,
        rms_norm_eps=config.rms_norm_eps,
    ).cuda()
    assert not hasattr(attn, "q_norm"), "qkv_bias=True 时不应有 q_norm"
    assert not hasattr(attn, "k_norm"), "qkv_bias=True 时不应有 k_norm"
    del attn
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.gpu
def test_qwen3_weight_loading():
    """测试 Qwen3 权重加载正确。"""
    from minillm.models.qwen3 import Qwen3ForCausalLM
    from minillm.utils.loader import load_model

    config = AutoConfig.from_pretrained(QWEN3_MODEL_PATH, trust_remote_code=True)
    model = Qwen3ForCausalLM(config).cuda()
    load_model(model, QWEN3_MODEL_PATH)
    # 验证关键层的权重非零
    assert model.model.embed_tokens.weight.data.abs().sum() > 0, "embedding 权重不应全零"
    assert model.lm_head.weight.data.abs().sum() > 0, "lm_head 权重不应全零"
    # 验证 Q/K RMSNorm 权重（Qwen3 qkv_bias=False 时有）
    attn = model.model.layers[0].self_attn
    assert attn.q_norm.weight.data.abs().sum() > 0, "q_norm 权重不应全零"
    assert attn.k_norm.weight.data.abs().sum() > 0, "k_norm 权重不应全零"
    del model
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.gpu
def test_qwen3_end_to_end():
    """端到端测试：Qwen3 使用 LLMEngine 生成文本。"""
    gc.collect()
    torch.cuda.empty_cache()

    from minillm.engine.llm_engine import LLM

    llm = LLM(
        model=QWEN3_MODEL_PATH,
        max_num_seqs=4,
        max_model_len=512,
    )
    sp = SamplingParams(temperature=0, max_tokens=16)
    outputs = llm.generate(["Hello, my name is"], sp, use_tqdm=False)
    assert len(outputs) == 1
    assert len(outputs[0]["token_ids"]) > 0
    assert isinstance(outputs[0]["text"], str)
    assert len(outputs[0]["text"]) > 0
    del llm
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.gpu
def test_qwen3_batch_generation():
    """批量生成测试：Qwen3 多条 prompt 同时推理。"""
    gc.collect()
    torch.cuda.empty_cache()

    from minillm.engine.llm_engine import LLM

    llm = LLM(
        model=QWEN3_MODEL_PATH,
        max_num_seqs=4,
        max_model_len=512,
    )
    sp = SamplingParams(temperature=0, max_tokens=8)
    prompts = ["Hello", "The capital of France is", "1 + 1 ="]
    outputs = llm.generate(prompts, sp, use_tqdm=False)
    assert len(outputs) == len(prompts)
    for out in outputs:
        assert len(out["token_ids"]) > 0
        assert isinstance(out["text"], str)
    del llm
    gc.collect()
    torch.cuda.empty_cache()
