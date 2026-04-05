"""VLM 常驻 HTTP 推理服务入口。"""

import argparse
import base64
import io
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

from minillm.cli import bytes_to_gib
from minillm.engine.vlm_engine import VLMEngine
from minillm.sampling_params import SamplingParams


def build_parser() -> argparse.ArgumentParser:
    """构建服务启动参数解析器。

    Returns:
        参数解析器
    """
    parser = argparse.ArgumentParser(description="启动 MINILLM VLM 常驻推理服务")
    parser.add_argument("--model", required=True, help="模型目录")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8001, help="监听端口")
    parser.add_argument("--device", default="cuda", help="设备（cuda 或 cpu）")
    parser.add_argument("--cache-size-tokens", type=int, default=None, help="按 token 容量指定 KV Cache 大小")
    parser.add_argument("--enforce-eager", action="store_true", help="禁用 decode CUDA Graph 及 torch.compile")
    return parser


import requests

def decode_image(img_str: str) -> Image.Image:
    """解码 base64 图像或下载 URL 图像。

    Args:
        img_str: base64 编码的图像字符串或图像 URL

    Returns:
        PIL Image
    """
    if img_str.startswith('http://') or img_str.startswith('https://'):
        response = requests.get(img_str, stream=True)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content))
    else:
        img_bytes = base64.b64decode(img_str)
        img = Image.open(io.BytesIO(img_bytes))
    return img


def build_sampling_params_from_payload(payload: dict) -> SamplingParams:
    """从 HTTP 请求体构建采样参数。

    Args:
        payload: 请求体字典

    Returns:
        采样参数对象
    """
    sampling_payload = payload.get("sampling_params", payload)
    max_tokens = sampling_payload.get("max_new_tokens", sampling_payload.get("max_tokens", 512))
    return SamplingParams(
        temperature=sampling_payload.get("temperature", 0.0),
        top_k=sampling_payload.get("top_k", 0),
        top_p=sampling_payload.get("top_p", 1.0),
        max_tokens=max_tokens,
    )


class VLMInferenceService:
    """VLM 常驻推理服务。

    通过互斥锁串行化请求。
    """

    def __init__(self, engine: VLMEngine, startup_config: dict):
        """初始化服务。

        Args:
            engine: VLM 引擎
            startup_config: 服务启动配置
        """
        self.engine = engine
        self.startup_config = startup_config
        self.lock = threading.Lock()

    def health(self) -> dict:
        """返回服务健康状态。"""
        return {
            "status": "ok",
            "model": self.startup_config["model"],
            "device": self.startup_config["device"],
            "cache_size_tokens": self.startup_config["cache_size_tokens"],
        }

    def memory(self) -> dict:
        """返回当前显存画像。"""
        with self.lock:
            profile = self.engine.get_memory_profile()
        profile["model_gib"] = bytes_to_gib(profile["model_bytes"])
        profile["kv_cache_total_gib"] = bytes_to_gib(profile["kv_cache_total_bytes"])
        profile["kv_cache_used_current_gib"] = bytes_to_gib(profile["kv_cache_used_bytes"])
        profile["kv_cache_used_peak_gib"] = bytes_to_gib(profile["kv_cache_peak_used_bytes"])
        return profile

    def generate(
        self,
        prompt: str,
        images: list[Image.Image] | None,
        sampling_params: SamplingParams,
        apply_chat_template: bool = True,
    ) -> dict:
        """执行生成。"""
        with self.lock:
            self.engine.reset_peak_memory_stats()
            return self.engine.generate(prompt, images, sampling_params, apply_chat_template=apply_chat_template)

    def generate_stream(
        self,
        prompt: str,
        images: list[Image.Image] | None,
        sampling_params: SamplingParams,
        apply_chat_template: bool = True,
    ):
        """执行流式生成。"""
        with self.lock:
            self.engine.reset_peak_memory_stats()
            yield from self.engine.generate_stream(prompt, images, sampling_params, apply_chat_template=apply_chat_template)


def create_handler(service: VLMInferenceService):
    """创建绑定到指定服务实例的 HTTP Handler。"""

    class VLMRequestHandler(BaseHTTPRequestHandler):
        """VLM HTTP 请求处理器。"""

        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            """处理 GET 请求。"""
            if self.path == "/health":
                self.send_json(HTTPStatus.OK, service.health())
                return
            if self.path == "/v1/memory":
                self.send_json(HTTPStatus.OK, service.memory())
                return
            self.send_json(HTTPStatus.NOT_FOUND, {"error": f"未知路径: {self.path}"})

        def do_POST(self) -> None:
            """处理 POST 请求。"""
            if self.path == "/v1/generate":
                self.handle_generate()
                return
            if self.path == "/v1/generate_stream":
                self.handle_generate_stream()
                return
            self.send_json(HTTPStatus.NOT_FOUND, {"error": f"未知路径: {self.path}"})

        def handle_generate(self) -> None:
            """处理生成请求。

            请求体格式：
            {
                "prompt": "描述这张图片",
                "images": ["base64_encoded_image1", "https://example.com/image.jpg"],
                "sampling_params": {"temperature": 0.7, "max_tokens": 512}
            }
            """
            try:
                payload = self.read_json_body()
                prompt = payload.get("prompt")
                assert isinstance(prompt, (str, list)), "prompt 必须是字符串或列表"

                images = None
                if "images" in payload and payload["images"]:
                    assert isinstance(payload["images"], list), "images 必须是数组"
                    images = [decode_image(img_str) for img_str in payload["images"]]

                sampling_params = build_sampling_params_from_payload(payload)
                apply_chat_template = payload.get("apply_chat_template", True)
                result = service.generate(prompt, images, sampling_params, apply_chat_template=apply_chat_template)
                if payload.get("profile", False):
                    result["metrics"] = service.memory()
                self.send_json(HTTPStatus.OK, result)
            except AssertionError as exc:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception as exc:
                self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

        def handle_generate_stream(self) -> None:
            """处理 SSE 流式生成请求。"""
            try:
                payload = self.read_json_body()
                prompt = payload.get("prompt")
                assert isinstance(prompt, (str, list)), "prompt 必须是字符串或列表"

                images = None
                if "images" in payload and payload["images"]:
                    assert isinstance(payload["images"], list), "images 必须是数组"
                    images = [decode_image(img_str) for img_str in payload["images"]]

                sampling_params = build_sampling_params_from_payload(payload)
                apply_chat_template = payload.get("apply_chat_template", True)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()

                for event in service.generate_stream(prompt, images, sampling_params, apply_chat_template=apply_chat_template):
                    self.write_sse_event(event["type"], event)
            except AssertionError as exc:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self.write_sse_event("error", {"error": str(exc)})

        def read_json_body(self) -> dict:
            """读取并解析 JSON 请求体。"""
            content_length = int(self.headers.get("Content-Length", "0"))
            assert content_length > 0, "请求体不能为空"
            body = self.rfile.read(content_length)
            payload = json.loads(body.decode("utf-8"))
            assert isinstance(payload, dict), "请求体必须是 JSON 对象"
            return payload

        def send_json(self, status_code: HTTPStatus, payload: dict) -> None:
            """发送 JSON 响应。"""
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()

        def write_sse_event(self, event_name: str, payload: dict) -> None:
            """发送单个 SSE 事件。"""
            data = json.dumps(payload, ensure_ascii=False)
            self.wfile.write(f"event: {event_name}\n".encode("utf-8"))
            self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
            self.wfile.flush()

        def log_message(self, format: str, *args) -> None:
            """关闭默认访问日志。"""

    return VLMRequestHandler


def main() -> None:
    """启动 VLM 常驻推理服务。"""
    parser = build_parser()
    args = parser.parse_args()

    engine_kwargs = {}
    if args.cache_size_tokens is not None:
        engine_kwargs["cache_size_tokens"] = args.cache_size_tokens
    if args.enforce_eager:
        engine_kwargs["enforce_eager"] = True
    else:
        # VLM currently requires eager mode to avoid torch.compile inplace errors
        engine_kwargs["enforce_eager"] = True
    engine = VLMEngine(args.model, device=args.device, **engine_kwargs)
    startup_config = {
        "model": args.model,
        "device": args.device,
        "cache_size_tokens": args.cache_size_tokens,
        "enforce_eager": True,
    }
    service = VLMInferenceService(engine, startup_config)
    handler = create_handler(service)
    server = ThreadingHTTPServer((args.host, args.port), handler)

    print(f"MINILLM VLM service listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
