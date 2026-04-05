"""常驻 HTTP 推理服务入口。"""

import argparse
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from minillm.cli import build_engine_kwargs, bytes_to_gib, validate_args
from minillm.engine.llm_engine import LLM
from minillm.sampling_params import SamplingParams


def build_parser() -> argparse.ArgumentParser:
    """构建服务启动参数解析器。

    Returns:
        参数解析器
    """
    parser = argparse.ArgumentParser(description="启动 MINILLM 常驻推理服务")
    parser.add_argument("--model", default="/app/models/qwen2.5-1.5B-Instruct", help="模型目录")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--max-num-seqs", type=int, default=1, help="最大并发序列数")
    parser.add_argument("--max-model-len", type=int, default=4096, help="最大上下文长度")
    parser.add_argument("--cache-size-mb", type=int, default=None, help="按 MB 指定 KV Cache 大小")
    parser.add_argument("--cache-size-tokens", type=int, default=None, help="按 token 容量指定 KV Cache 大小")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="自动分配 KV Cache 时的显存利用率")
    parser.add_argument("--enforce-eager", action="store_true", help="禁用 decode CUDA Graph")
    return parser


def normalize_prompt(prompt: str | list[int]) -> str | list[int]:
    """校验并规范化请求中的 prompt。

    Args:
        prompt: 文本或 token id 列表

    Returns:
        原样返回的合法 prompt
    """
    assert isinstance(prompt, (str, list)), "prompt 必须是字符串或 token id 列表"
    if isinstance(prompt, list):
        assert all(isinstance(token_id, int) for token_id in prompt), "prompt token 列表必须全部为 int"
    return prompt


def build_sampling_params_from_payload(payload: dict) -> SamplingParams:
    """从 HTTP 请求体构建采样参数。

    Args:
        payload: 请求体字典

    Returns:
        采样参数对象
    """
    sampling_payload = payload.get("sampling_params", payload)
    max_tokens = sampling_payload.get("max_new_tokens", sampling_payload.get("max_tokens", 64))
    return SamplingParams(
        temperature=sampling_payload.get("temperature", 0.0),
        top_k=sampling_payload.get("top_k", 0),
        top_p=sampling_payload.get("top_p", 1.0),
        max_tokens=max_tokens,
        ignore_eos=sampling_payload.get("ignore_eos", False),
    )


def sanitize_metrics(metrics: dict, include_token_ids: bool, include_profile: bool) -> dict:
    """整理最终返回的结果或指标。

    Args:
        metrics: 引擎最终 metrics 事件
        include_token_ids: 是否保留 token_ids
        include_profile: 是否保留完整性能画像

    Returns:
        可直接返回给客户端的字典
    """
    payload = {"text": metrics["text"]}
    if include_token_ids:
        payload["token_ids"] = metrics["token_ids"]
    if include_profile:
        profile = {
            key: value
            for key, value in metrics.items()
            if key not in {"type", "seq_id", "text", "token_ids"}
        }
        profile["model_gib"] = bytes_to_gib(metrics["model_bytes"])
        profile["kv_cache_total_gib"] = bytes_to_gib(metrics["kv_cache_total_bytes"])
        profile["kv_cache_used_current_gib"] = bytes_to_gib(metrics["kv_cache_used_bytes"])
        profile["kv_cache_used_peak_gib"] = bytes_to_gib(metrics["kv_cache_peak_used_bytes"])
        profile["cuda_allocated_gib"] = bytes_to_gib(metrics["cuda_memory_allocated_bytes"])
        profile["cuda_peak_allocated_gib"] = bytes_to_gib(metrics["cuda_max_memory_allocated_bytes"])
        profile["cuda_reserved_gib"] = bytes_to_gib(metrics["cuda_memory_reserved_bytes"])
        profile["cuda_peak_reserved_gib"] = bytes_to_gib(metrics["cuda_max_memory_reserved_bytes"])
        payload["metrics"] = profile
    return payload


class InferenceService:
    """常驻推理服务。

    单卡引擎本身不是线程安全的，因此服务层通过互斥锁串行化请求。
    """

    def __init__(self, llm: LLM, startup_config: dict):
        """初始化服务。

        Args:
            llm: 已加载模型的推理引擎
            startup_config: 服务启动配置
        """
        self.llm = llm
        self.startup_config = startup_config
        self.lock = threading.Lock()

    def health(self) -> dict:
        """返回服务健康状态。"""
        return {
            "status": "ok",
            "model": self.startup_config["model"],
            "max_model_len": self.startup_config["max_model_len"],
            "max_num_seqs": self.startup_config["max_num_seqs"],
            "cache_size_mb": self.startup_config["cache_size_mb"],
            "cache_size_tokens": self.startup_config["cache_size_tokens"],
            "gpu_memory_utilization": self.startup_config["gpu_memory_utilization"],
            "enforce_eager": self.startup_config["enforce_eager"],
        }

    def memory(self) -> dict:
        """返回当前显存画像。"""
        with self.lock:
            profile = self.llm.get_memory_profile()
        profile["model_gib"] = bytes_to_gib(profile["model_bytes"])
        profile["kv_cache_total_gib"] = bytes_to_gib(profile["kv_cache_total_bytes"])
        profile["kv_cache_used_current_gib"] = bytes_to_gib(profile["kv_cache_used_bytes"])
        profile["kv_cache_used_peak_gib"] = bytes_to_gib(profile["kv_cache_peak_used_bytes"])
        return profile

    def generate(self, prompt: str | list[int], sampling_params: SamplingParams) -> dict:
        """执行单次非流式生成并返回最终 metrics。"""
        with self.lock:
            self.llm.reset_peak_memory_stats()
            metrics = None
            for event in self.llm.generate_stream(prompt, sampling_params):
                if event["type"] == "metrics":
                    metrics = event
            assert metrics is not None, "generate_stream 必须返回 metrics 事件"
            return metrics

    def batch_generate(
        self,
        prompts: list[str | list[int]],
        sampling_params: SamplingParams,
    ) -> list[dict]:
        """执行批量非流式生成。

        Args:
            prompts: prompt 列表
            sampling_params: 采样参数（所有 prompt 共用）

        Returns:
            每个 prompt 的 metrics 列表
        """
        with self.lock:
            self.llm.reset_peak_memory_stats()
            outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
            # 重新收集 metrics（generate 不返回完整 metrics，需要重新跑）
            # 改用 generate_stream 逐个生成以获得完整 metrics
            results = []
            for prompt in prompts:
                self.llm.reset_peak_memory_stats()
                metrics = None
                for event in self.llm.generate_stream(prompt, sampling_params):
                    if event["type"] == "metrics":
                        metrics = event
                assert metrics is not None
                results.append(metrics)
            return results

    def generate_stream(self, prompt: str | list[int], sampling_params: SamplingParams, apply_chat_template: bool = True):
        """执行单次流式生成。"""
        with self.lock:
            self.llm.reset_peak_memory_stats()
            yield from self.llm.generate_stream(prompt, sampling_params, apply_chat_template)


def create_handler(service: InferenceService):
    """创建绑定到指定服务实例的 HTTP Handler。"""

    class InferenceRequestHandler(BaseHTTPRequestHandler):
        """MINILLM HTTP 请求处理器。"""

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
            if self.path == "/v1/batch_generate":
                self.handle_batch_generate()
                return
            self.send_json(HTTPStatus.NOT_FOUND, {"error": f"未知路径: {self.path}"})

        def handle_generate(self) -> None:
            """处理非流式生成请求。"""
            try:
                payload = self.read_json_body()
                prompt = normalize_prompt(payload["prompt"])
                sampling_params = build_sampling_params_from_payload(payload)
                include_token_ids = payload.get("return_token_ids", False)
                include_profile = payload.get("profile", False)
                metrics = service.generate(prompt, sampling_params)
                response = sanitize_metrics(metrics, include_token_ids, include_profile)
                self.send_json(HTTPStatus.OK, response)
            except AssertionError as exc:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

        def handle_batch_generate(self) -> None:
            """处理批量生成请求。

            请求体格式：
            {
                "prompts": ["prompt1", "prompt2", ...],
                "sampling_params": {"temperature": 0, "max_tokens": 32},
                "return_token_ids": false,
                "profile": false
            }
            """
            try:
                payload = self.read_json_body()
                raw_prompts = payload.get("prompts", [])
                assert isinstance(raw_prompts, list) and len(raw_prompts) > 0, (
                    "prompts 必须是非空数组"
                )
                prompts = [normalize_prompt(p) for p in raw_prompts]
                sampling_params = build_sampling_params_from_payload(payload)
                include_token_ids = payload.get("return_token_ids", False)
                include_profile = payload.get("profile", False)
                results = service.batch_generate(prompts, sampling_params)
                response_items = []
                for metrics in results:
                    item = sanitize_metrics(metrics, include_token_ids, include_profile)
                    response_items.append(item)
                self.send_json(HTTPStatus.OK, {"results": response_items})
            except AssertionError as exc:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

        def handle_generate_stream(self) -> None:
            """处理 SSE 流式生成请求。"""
            try:
                payload = self.read_json_body()
                prompt = normalize_prompt(payload["prompt"])
                sampling_params = build_sampling_params_from_payload(payload)
                include_token_ids = payload.get("return_token_ids", False)
                include_profile = payload.get("profile", False)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()

                apply_chat_template = payload.get("apply_chat_template", True)
                for event in service.generate_stream(prompt, sampling_params, apply_chat_template=apply_chat_template):
                    event_name = event["type"]
                    data = event
                    if event_name == "metrics":
                        data = sanitize_metrics(event, include_token_ids, include_profile)
                        data["type"] = "metrics" if include_profile else "result"
                        event_name = data["type"]
                    self.write_sse_event(event_name, data)
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

    return InferenceRequestHandler


def create_service_from_args(args: argparse.Namespace) -> InferenceService:
    """根据启动参数创建推理服务。

    Args:
        args: 启动参数

    Returns:
        推理服务实例
    """
    llm = LLM(args.model, **build_engine_kwargs(args))
    startup_config = {
        "model": args.model,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "cache_size_mb": args.cache_size_mb,
        "cache_size_tokens": args.cache_size_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
    }
    return InferenceService(llm, startup_config)


def main() -> None:
    """启动常驻推理服务。"""
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    service = create_service_from_args(args)
    handler = create_handler(service)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"MINILLM service listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
