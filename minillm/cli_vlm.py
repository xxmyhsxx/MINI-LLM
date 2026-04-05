"""VLM 命令行推理入口。"""

import argparse
import json
from pathlib import Path

from PIL import Image

from minillm.cli import bytes_to_gib, format_metrics
from minillm.engine.vlm_engine import VLMEngine
from minillm.sampling_params import SamplingParams


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。

    Returns:
        参数解析器
    """
    parser = argparse.ArgumentParser(description="运行 Qwen2.5-VL 视觉语言模型推理")
    parser.add_argument("--model", required=True, help="模型目录")
    parser.add_argument("--prompt", required=True, help="输入文本 prompt")
    parser.add_argument("--images", nargs="+", help="图像文件路径列表")
    parser.add_argument("--max-tokens", "--max-new-tokens", dest="max_tokens", type=int, default=512, help="最大生成新 token 数")
    parser.add_argument("--temperature", type=float, default=0.0, help="温度")
    parser.add_argument("--top-k", type=int, default=0, help="top-k")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p")
    parser.add_argument("--device", default="cuda", help="设备（cuda 或 cpu）")
    parser.add_argument("--cache-size-tokens", type=int, default=None, help="按 token 容量指定 KV Cache 大小")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    parser.add_argument("--stream", action="store_true", help="启用流式输出")
    parser.add_argument("--profile", action="store_true", help="输出性能分析信息（TTFT、速度、显存）")
    parser.add_argument("--enforce-eager", action="store_true", help="禁用 decode CUDA Graph 及 torch.compile")
    parser.add_argument("--no-chat-template", action="store_true", help="禁用自动对话模板拼接")
    return parser


import requests
from io import BytesIO

def load_images(image_paths: list[str] | None) -> list[Image.Image]:
    """加载图像文件或 URL。

    Args:
        image_paths: 图像文件路径或 URL 列表

    Returns:
        PIL Image 列表
    """
    if image_paths is None:
        return []

    images = []
    for path in image_paths:
        if path.startswith('http://') or path.startswith('https://'):
            response = requests.get(path, stream=True)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
        else:
            path_obj = Path(path)
            assert path_obj.exists(), f"图像文件不存在: {path}"
            img = Image.open(path)
            
        images.append(img)

    return images


def main() -> None:
    """执行 VLM 命令行推理。"""
    parser = build_parser()
    args = parser.parse_args()

    # 加载图像
    images = load_images(args.images)

    # 初始化引擎
    engine_kwargs = {}
    if args.cache_size_tokens is not None:
        engine_kwargs["cache_size_tokens"] = args.cache_size_tokens
    if args.enforce_eager:
        engine_kwargs["enforce_eager"] = True
    else:
        # VLM currently requires eager mode to avoid torch.compile inplace errors
        engine_kwargs["enforce_eager"] = True
    engine = VLMEngine(args.model, device=args.device, **engine_kwargs)

    # 构建采样参数
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    # 生成
    if args.stream or args.profile:
        metrics = None
        streamed_any_token = False
        for event in engine.generate_stream(args.prompt, images, sampling_params, apply_chat_template=not args.no_chat_template):
            if event["type"] == "token":
                if args.stream:
                    streamed_any_token = True
                    if args.json:
                        print(json.dumps(event, ensure_ascii=False), flush=True)
                    else:
                        print(event["text"], end="", flush=True)
                continue
            metrics = event

        assert metrics is not None, "流式推理必须返回 metrics 事件"
        if args.stream and not args.json and streamed_any_token:
            print()
        if args.json:
            print(json.dumps(metrics, ensure_ascii=False, indent=2))
        else:
            if not args.stream:
                print(metrics["text"])
            if args.profile:
                print(format_metrics(metrics))
        return

    result = engine.generate(args.prompt, images, sampling_params, apply_chat_template=not args.no_chat_template)

    # 输出
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result["text"])


if __name__ == "__main__":
    main()
