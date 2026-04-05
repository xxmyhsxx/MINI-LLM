"""VLM 模型权重加载器。

处理 Qwen2.5-VL 等 VLM 模型的权重加载，包括：
1. 参数前缀映射（model.* → language_model.model.*）
2. 分离权重融合（Q/K/V → QKV, gate/up → gate_up）
3. 视觉编码器权重直接映射
"""

import os
from pathlib import Path

import torch
from safetensors import safe_open


def load_vlm_model(model: torch.nn.Module, model_path: str) -> None:
    """加载 VLM 模型权重。

    Args:
        model: VLM 模型实例
        model_path: 模型权重目录路径
    """
    model_path = Path(model_path)

    # 收集所有 safetensors 文件
    weight_files = sorted([
        f for f in os.listdir(model_path)
        if f.endswith('.safetensors') and 'index' not in f
    ])

    if not weight_files:
        raise ValueError(f"未找到权重文件: {model_path}")

    # 构建模型参数字典
    model_params = {name: param for name, param in model.named_parameters()}

    # 加载权重
    loaded_keys = set()

    for weight_file in weight_files:
        file_path = model_path / weight_file
        with safe_open(file_path, 'pt', 'cpu') as f:
            for key in f.keys():
                _load_weight(model_params, f, key, loaded_keys)


    # 检查是否所有参数都已加载
    missing_keys = set(model_params.keys()) - loaded_keys
    
    # 手动处理 lm_head 的绑定
    if 'language_model.lm_head.weight' in missing_keys and 'language_model.model.embed_tokens.weight' in loaded_keys:
        model.language_model.lm_head.weight = model.language_model.model.embed_tokens.weight
        missing_keys.remove('language_model.lm_head.weight')

    if missing_keys:
        print(f"警告: {len(missing_keys)} 个参数未加载:")
        for key in sorted(list(missing_keys)[:10]):
            print(f"  - {key}")


def _load_weight(
    model_params: dict[str, torch.nn.Parameter],
    weight_file,
    weight_key: str,
    loaded_keys: set[str],
) -> None:
    """加载单个权重。

    Args:
        model_params: 模型参数字典
        weight_file: safetensors 文件句柄
        weight_key: 权重键名
        loaded_keys: 已加载的键集合
    """
    # 视觉编码器权重直接映射
    if weight_key.startswith('visual.'):
        if weight_key in model_params:
            weight = weight_file.get_tensor(weight_key)
            model_params[weight_key].data.copy_(weight)
            loaded_keys.add(weight_key)
        return

    # 语言模型权重需要前缀映射和融合
    if weight_key.startswith('model.') or weight_key.startswith('lm_head.'):
        _load_language_model_weight(model_params, weight_file, weight_key, loaded_keys)
        return


def _load_language_model_weight(
    model_params: dict[str, torch.nn.Parameter],
    weight_file,
    weight_key: str,
    loaded_keys: set[str],
) -> None:
    """加载语言模型权重（处理前缀映射和权重融合）。

    Args:
        model_params: 模型参数字典
        weight_file: safetensors 文件句柄
        weight_key: 权重键名（格式：model.*）
        loaded_keys: 已加载的键集合
    """
    # 处理 QKV 融合（weight 和 bias）
    if '.self_attn.q_proj.weight' in weight_key or '.self_attn.q_proj.bias' in weight_key:
        _load_qkv_weight(model_params, weight_file, weight_key, loaded_keys)
        return

    # 处理 gate_up 融合（只有 weight，没有 bias）
    if '.mlp.gate_proj.weight' in weight_key:
        _load_gate_up_weight(model_params, weight_file, weight_key, loaded_keys)
        return

    # 跳过已融合的 K/V/up 权重
    if any(x in weight_key for x in ['.self_attn.k_proj.', '.self_attn.v_proj.', '.mlp.up_proj.']):
        return

    # 添加 language_model 前缀并直接映射其他权重
    if weight_key.startswith('model.'):
        model_key = weight_key.replace('model.', 'language_model.model.')
    elif weight_key.startswith('lm_head.'):
        model_key = weight_key.replace('lm_head.', 'language_model.lm_head.')
    else:
        model_key = weight_key
    if model_key in model_params:
        weight = weight_file.get_tensor(weight_key)
        model_params[model_key].data.copy_(weight)
        loaded_keys.add(model_key)


def _load_qkv_weight(
    model_params: dict[str, torch.nn.Parameter],
    weight_file,
    q_key: str,
    loaded_keys: set[str],
) -> None:
    """加载并融合 Q/K/V 权重。

    Args:
        model_params: 模型参数字典
        weight_file: safetensors 文件句柄
        q_key: Q 权重键名（格式：model.layers.X.self_attn.q_proj.weight/bias）
        loaded_keys: 已加载的键集合
    """
    # 构造 K/V 键名
    k_key = q_key.replace('.q_proj.', '.k_proj.')
    v_key = q_key.replace('.q_proj.', '.v_proj.')

    # 构造目标键名（处理 weight 和 bias）
    if '.weight' in q_key:
        target_key = q_key.replace('model.', 'language_model.model.').replace('.q_proj.weight', '.qkv_proj.weight')
    else:
        target_key = q_key.replace('model.', 'language_model.model.').replace('.q_proj.bias', '.qkv_proj.bias')

    if target_key not in model_params:
        return

    # 加载 Q/K/V 权重
    q_weight = weight_file.get_tensor(q_key)
    k_weight = weight_file.get_tensor(k_key)
    v_weight = weight_file.get_tensor(v_key)

    # 融合 [Q, K, V]
    qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
    model_params[target_key].data.copy_(qkv_weight)
    loaded_keys.add(target_key)


def _load_gate_up_weight(
    model_params: dict[str, torch.nn.Parameter],
    weight_file,
    gate_key: str,
    loaded_keys: set[str],
) -> None:
    """加载并融合 gate/up 权重。

    Args:
        model_params: 模型参数字典
        weight_file: safetensors 文件句柄
        gate_key: gate 权重键名（格式：model.layers.X.mlp.gate_proj.weight）
        loaded_keys: 已加载的键集合
    """
    # 构造 up 键名
    up_key = gate_key.replace('.gate_proj.', '.up_proj.')

    # 构造目标键名
    target_key = gate_key.replace('model.', 'language_model.model.').replace('.gate_proj.weight', '.gate_up_proj.weight')

    if target_key not in model_params:
        return

    # 加载 gate/up 权重
    gate_weight = weight_file.get_tensor(gate_key)
    up_weight = weight_file.get_tensor(up_key)

    # 融合 [gate, up]
    gate_up_weight = torch.cat([gate_weight, up_weight], dim=0)
    model_params[target_key].data.copy_(gate_up_weight)
    loaded_keys.add(target_key)
