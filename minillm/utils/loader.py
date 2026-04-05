import os
from glob import glob

import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """默认权重加载器，直接拷贝权重数据。

    Args:
        param: 目标参数
        loaded_weight: 加载的权重张量
    """
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    """从 SafeTensors 文件加载模型权重。

    支持 packed_modules_mapping，将多个权重合并加载到一个参数中。
    支持分片权重文件（model-*.safetensors）。

    Args:
        model: 目标模型
        path: 权重文件目录路径
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    files = sorted(glob(os.path.join(path, "*.safetensors")))
    # 过滤掉 index 文件
    files = [f for f in files if "index" not in f]
    for file in files:
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                matched = False
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        try:
                            param = model.get_parameter(param_name)
                        except (AttributeError, KeyError, RuntimeError):
                            matched = True
                            break
                        if param is None:
                            matched = True
                            break
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        matched = True
                        break
                if matched:
                    continue
                # 跳过不存在或为 None 的参数（如 bias=None）
                try:
                    param = model.get_parameter(weight_name)
                except (AttributeError, KeyError, RuntimeError):
                    continue
                if param is None:
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, f.get_tensor(weight_name))

    # 处理 tie_word_embeddings（lm_head 与 embed_tokens 共享权重）
    hf_config = getattr(model, "hf_config", None) or getattr(model, "config", None)
    if hf_config is not None and getattr(hf_config, "tie_word_embeddings", False):
        if hasattr(model, "lm_head") and hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            model.lm_head.weight.data = model.model.embed_tokens.weight.data
