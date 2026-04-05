"""模型注册表，支持动态注册和创建模型实例。"""

from typing import Type

from torch import nn

# 全局模型注册表
_REGISTRY: dict[str, Type[nn.Module]] = {}


def register(model_type: str, model_class: Type[nn.Module]):
    """注册模型类。

    Args:
        model_type: 模型类型标识（如 "qwen2", "qwen3"）
        model_class: 模型类
    """
    _REGISTRY[model_type] = model_class


def create(model_type: str, config) -> nn.Module:
    """根据模型类型创建模型实例。

    Args:
        model_type: 模型类型标识
        config: HuggingFace 模型配置

    Returns:
        模型实例

    Raises:
        KeyError: 未注册的模型类型
    """
    if model_type not in _REGISTRY:
        raise KeyError(f"未注册的模型类型: {model_type}，已注册: {list(_REGISTRY.keys())}")
    return _REGISTRY[model_type](config)


def list_registered() -> list[str]:
    """列出所有已注册的模型类型。

    Returns:
        模型类型列表
    """
    return list(_REGISTRY.keys())
