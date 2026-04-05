"""Qwen2 模型单元测试（需要 GPU）。"""

import pytest
import torch


@pytest.mark.gpu
def test_qwen2_model_creation():
    """测试 Qwen2 模型创建。"""
    from transformers import AutoConfig
    from minillm.models.qwen2 import Qwen2ForCausalLM

    config = AutoConfig.from_pretrained("/app/models/qwen2.5-1.5B-Instruct", trust_remote_code=True)
    model = Qwen2ForCausalLM(config)
    model = model.cuda()

    # 检查模型结构
    assert hasattr(model, "model")
    assert hasattr(model, "lm_head")
    assert len(model.model.layers) == config.num_hidden_layers
    assert model.model.embed_tokens.weight.shape == (config.vocab_size, config.hidden_size)


@pytest.mark.gpu
def test_qwen2_weight_loading():
    """测试 Qwen2 权重加载。"""
    from transformers import AutoConfig
    from minillm.models.qwen2 import Qwen2ForCausalLM
    from minillm.utils.loader import load_model

    config = AutoConfig.from_pretrained("/app/models/qwen2.5-1.5B-Instruct", trust_remote_code=True)
    model = Qwen2ForCausalLM(config).cuda().to(config.torch_dtype)

    # 加载权重
    load_model(model, "/app/models/qwen2.5-1.5B-Instruct")

    # 验证权重已加载（非全零）
    emb_weight = model.model.embed_tokens.weight.data
    assert emb_weight.abs().sum() > 0
    # 验证 attention bias 已正确创建并加载
    qkv_bias = model.model.layers[0].self_attn.qkv_proj.bias
    assert qkv_bias is not None
    assert qkv_bias.abs().sum() > 0
    # 验证 lm_head 权重已加载（tie_word_embeddings=True）
    assert model.lm_head.weight.data.abs().sum() > 0


@pytest.mark.gpu
def test_qwen2_end_to_end():
    """端到端测试：使用 LLMEngine 生成文本。"""
    from minillm.engine.llm_engine import LLM
    from minillm.sampling_params import SamplingParams

    llm = LLM(
        model="/app/models/qwen2.5-1.5B-Instruct",
        max_num_seqs=4,
        max_model_len=512,
    )
    sp = SamplingParams(temperature=0, max_tokens=16)
    outputs = llm.generate(["Hello, my name is"], sp, use_tqdm=False)

    assert len(outputs) == 1
    assert "text" in outputs[0]
    assert "token_ids" in outputs[0]
    assert len(outputs[0]["token_ids"]) > 0
    print(f"Generated: {outputs[0]['text']}")


@pytest.mark.gpu
def test_qwen2_batch_end_to_end():
    """端到端测试：批量生成。"""
    import gc
    torch.cuda.empty_cache()
    gc.collect()

    from minillm.engine.llm_engine import LLM
    from minillm.sampling_params import SamplingParams

    llm = LLM(
        model="/app/models/qwen2.5-1.5B-Instruct",
        max_num_seqs=4,
        max_model_len=512,
    )
    sp = SamplingParams(temperature=0, max_tokens=8)
    outputs = llm.generate(
        ["The capital of France is", "1 + 1 ="],
        sp, use_tqdm=False,
    )

    assert len(outputs) == 2
    for output in outputs:
        assert len(output["token_ids"]) > 0
        print(f"Generated: {output['text']}")
