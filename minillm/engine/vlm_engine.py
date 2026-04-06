"""VLM Engine。"""

from PIL import Image

from minillm.engine.multimodal_engine import MultimodalEngine
from minillm.sampling_params import SamplingParams


class VLMEngine:
    """框架对外公开的唯一 VLM 引擎接口。"""

    def __init__(self, model_path: str, device: str = "cuda", **kwargs):
        """初始化 VLM 引擎。"""
        assert device == "cuda", "当前 VLMEngine 仅支持 cuda"
        self._engine = MultimodalEngine(model_path, **kwargs)

    def generate(
        self,
        prompt: str | list[dict],
        images: list[Image.Image] | None = None,
        sampling_params: SamplingParams | None = None,
        apply_chat_template: bool = True,
    ) -> dict:
        """生成单个响应。"""
        return self._engine.generate(prompt, images, sampling_params, apply_chat_template)

    def batch_generate(
        self,
        requests: list[dict],
        sampling_params: SamplingParams | list[SamplingParams] | None = None,
        apply_chat_template: bool = True,
    ) -> list[dict]:
        """批量生成响应。"""
        return self._engine.batch_generate(requests, sampling_params, apply_chat_template)

    def generate_stream(
        self,
        prompt: str | list[dict],
        images: list[Image.Image] | None = None,
        sampling_params: SamplingParams | None = None,
        apply_chat_template: bool = True,
    ):
        """流式生成响应。"""
        yield from self._engine.generate_stream(prompt, images, sampling_params, apply_chat_template)

    def __getattr__(self, name: str):
        """透传其余属性到统一引擎。"""
        return getattr(self._engine, name)
