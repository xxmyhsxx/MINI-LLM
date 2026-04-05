# MINILLM 推理框架

极致简化的 LLM 推理框架，用于将大语言模型部署到边缘设备（单卡推理）。

## 最新状态

- 当前已支持 `qwen2.5-1.5B-Instruct` 与 `Qwen3-0.6B` 单卡文本推理
- 🎉 **新增：支持 `Qwen2.5-VL` 多模态单卡推理（端到端）**
- 已修复“输出乱码”、`max_tokens=1024` 长度溢出、`sampling_params` 静默截断、KV 占用统计歧义等问题
- 已提供命令行推理接口与常驻 HTTP 服务接口
- 当前测试结果：`73 passed`
- 最近 VLM 专项验证：`minillm/tests/test_vlm_engine.py` 为 `3 passed`
- 当前 VLM benchmark 已补齐 `prefill / decode / batch` 维度
- 当前单图 VLM benchmark：minillm `prefill 0.65s / decode 25.93 tok/s / 8.62 GiB allocated / 9.03 GiB reserved`，HuggingFace `prefill 1.08s / decode 38.73 tok/s / 10.36 GiB allocated / 11.28 GiB reserved`
- 当前 batch VLM benchmark：minillm 仍为串行单请求路径，HF 已能原生 batch，因此 decode 吞吐对比主要反映 runtime 架构差异

## 文档索引

- 排障与维护报告：`/app/minillm/docs/DEBUG_REPORT.md`
- Benchmark 报告：`/app/minillm/docs/BENCHMARK.md`
- 完成记录：`/app/minillm/docs/DONE.md`
- 待办清单：`/app/minillm/docs/TODO.md`
- 开发计划：`/app/minillm/docs/DEVELOPMENT_PLAN.md`
- Phase 2 计划：`/app/minillm/docs/PHASE2_PLAN.md`
- benchmark 原始结果：`/app/minillm/benchmark_results/`

## 项目概述

MINILLM 参考 `nano-vllm` 的设计，保留单卡推理关键路径，聚焦边缘端场景下的可维护性、可验证性和性能分析能力。

## 目录结构

```text
minillm/
  minillm/
    cli.py
    cli_vlm.py
    server.py
    server_vlm.py
    config.py
    sampling_params.py
    engine/
    layers/
    models/
    vision/
    utils/
    tests/
  docs/
    DEBUG_REPORT.md
    BENCHMARK.md
    DONE.md
    TODO.md
    DEVELOPMENT_PLAN.md
    PHASE2_PLAN.md
  benchmark_results/
  pyproject.toml
```

## 已验证能力

### 推理能力

- Qwen2.5 / Qwen3 单条生成
- Qwen2.5 / Qwen3 批量生成
- **Qwen2.5-VL 图像文本多模态生成（单图/多图）**
- greedy / temperature / top-k / top-p 采样
- KV Cache 自动分配、按 MB 分配、按 token 容量分配
- 流式输出
- TTFT / 吞吐 / 显存分析

### 稳定性修复

- 💡 **新增：引擎内置自动 Chat Template 拼接（支持开关），允许用户自由选择对话包装或纯自由续写**
- 修复模型生成 `endoftext` 或 `im_end` 时未正确截断导致无限输出无效 token 的问题
- 修复模型输出结果中未过滤特殊 token 导致明文携带结束符的问题
- 修复 VLM 在初始化时由于 `.to(dtype)` 导致的 `lm_head` 和 `embed_tokens` 权重解绑，导致完全幻觉或乱码的问题
- 修复 Qwen2.5 `q/k/v` bias 配置错误
- 修复 prefill 阶段 attention 批内串扰
- 修复 `generate()` 的 `sampling_params` 静默截断
- 修复长生成时 `max_model_len` 推导不合理导致的报错
- 将 KV 占用分析拆成 Current / Peak 两种口径

## 安装

```bash
pip install -e .
```

依赖：`torch>=2.4.0`、`triton>=3.0.0`、`transformers>=4.51.0`、`flash-attn`、`xxhash`、`safetensors`

## 支持模型

- `qwen2`：Qwen2.5 系列（如 qwen2.5-1.5B-Instruct）
- `qwen3`：Qwen3 系列（如 Qwen3-0.6B）
- `qwen2_5_vl`：Qwen2.5-VL 多模态系列（如 Qwen2.5-VL-3B-Instruct）

模型通过 `ModelRegistry` 自动识别，根据模型目录中的 `config.json` 的 `model_type` 字段路由。

## Python API

### 文本推理 (LLM)

```python
from minillm.engine.llm_engine import LLM
from minillm.sampling_params import SamplingParams

# 初始化引擎
llm = LLM(
    model="/app/models/qwen2.5-1.5B-Instruct",
    max_num_seqs=4,
    max_model_len=512
)

# 单次生成
sp = SamplingParams(temperature=0, max_tokens=16)
outputs = llm.generate(["Hello, my name is"], sp, use_tqdm=False)
print(outputs[0]["text"])

# 批量生成
prompts = ["Hello", "The capital of France is", "1 + 1 ="]
outputs = llm.generate(prompts, sp, use_tqdm=False)
for output in outputs:
    print(output["text"])

# 流式生成
for event in llm.generate_stream("Explain AI", sp):
    if event["type"] == "token":
        print(event["text"], end="", flush=True)
    elif event["type"] == "metrics":
        print(f"\n速度: {event['decode_throughput']:.2f} tok/s")
```

### 多模态推理 (VLM)

```python
from PIL import Image
from minillm.engine.vlm_engine import VLMEngine
from minillm.sampling_params import SamplingParams

# 初始化 VLM 引擎
vlm = VLMEngine(
    model_path="/app/models/Qwen2.5-VL-3B-Instruct",
    cache_size_tokens=16384  # 根据需要调整
)

# 单图推理
img = Image.open("test.jpg")
sp = SamplingParams(temperature=0.2, max_tokens=128)
result = vlm.generate(prompt="描述这张图片", images=[img], sampling_params=sp)
print(result["text"])

# 批量推理
requests = [
    {"text": "描述这张图片", "images": [img]},
    {"text": "图片中有什么？", "images": [img]},
    {"text": "这是在哪里？", "images": [img]},
]
results = vlm.batch_generate(requests, sp)
for i, result in enumerate(results):
    print(f"[{i}] {result['text']}")

# 流式生成
for event in vlm.generate_stream("这张图片有什么特点？", [img], sp):
    if event["type"] == "token":
        print(event["text"], end="", flush=True)
    elif event["type"] == "metrics":
        print(f"\n速度: {event['decode_throughput']:.2f} tok/s")
```

### 主要接口

**LLM 引擎**:
- `generate(prompts, sampling_params, apply_chat_template=True)`: 批量生成
- `generate_stream(prompt, sampling_params, apply_chat_template=True)`: 流式生成
- `get_memory_profile()`: 返回显存使用情况
- `reset_peak_memory_stats()`: 重置峰值显存统计

**VLM 引擎**:
- `generate(prompt, images, sampling_params, apply_chat_template=True)`: 单次生成
- `batch_generate(requests, sampling_params, apply_chat_template=True)`: 批量生成
- `generate_stream(prompt, images, sampling_params, apply_chat_template=True)`: 流式生成
- `get_memory_profile()`: 返回显存使用情况



## 命令行接口

### 单次推理

```bash
python -m minillm.cli \
  --model /app/models/qwen2.5-1.5B-Instruct \
  --prompt "Hello, my name is" \
  --max-tokens 32
```


### 多模态推理 (VLM)

支持本地文件和 HTTP URL 混合输入：

```bash
# 需在支持 Flash Attention 的环境运行
minillm-vlm \
  --model /app/models/Qwen2.5-VL-3B-Instruct \
  --prompt "描述这张图片" \
  --images /path/to/image1.jpg https://example.com/image2.png \
  --max-tokens 128
```

### 批量推理

```bash
# 每行一条 prompt
python -m minillm.cli \
  --model /app/models/Qwen3-0.6B \
  --prompts-file prompts.txt \
  --max-tokens 32

# JSON 数组格式
python -m minillm.cli \
  --model /app/models/Qwen3-0.6B \
  --prompts-file prompts.json \
  --max-tokens 32 --json
```

### 流式 + 显存画像

```bash
python -m minillm.cli \
  --model /app/models/qwen2.5-1.5B-Instruct \
  --prompt "Explain paged attention." \
  --max-tokens 128 \
  --cache-size-tokens 8192 \
  --stream \
  --profile
```

CLI 关键参数：
- `--prompt`：单条推理的 prompt
- `--prompts-file`：批量推理的 prompt 文件（每行一条 / JSON 数组），与 `--prompt` 二选一（仅文本 CLI 支持）
- `--images`：多模态推理输入的图像路径或 URL 列表（仅 `minillm-vlm` CLI 支持）
- `--cache-size-mb`：按显存大小配置 KV Cache
- `--cache-size-tokens`：按 token 容量配置 KV Cache
- `--stream`：按 token 输出
- `--profile`：输出 TTFT、速度、显存画像
- `--max-model-len`：最大上下文长度；若不传，CLI 自动按最长 prompt + max_tokens 推导
- `--no-chat-template`：禁用自动对话模板拼接，允许用户自由续写

显存画像输出包含：
- 模型权重显存
- KV Cache 总显存
- KV Cache 当前已用显存（Current）
- KV Cache 峰值已用显存（Peak）
- KV blocks 当前已用数 / 总数
- KV blocks 峰值已用数 / 总数
- CUDA 当前 / 峰值 allocated
- CUDA 当前 / 峰值 reserved

## 常驻 HTTP 服务

服务会常驻加载模型，避免每次请求重新初始化。由于当前单卡引擎不是线程安全的，请求在服务层会串行化处理。

### 启动服务

```bash
minillm-serve \
  --model /app/models/qwen2.5-1.5B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --cache-size-tokens 8192 \
  --max-model-len 4096
```


### 启动多模态服务 (VLM)

```bash
minillm-vlm-serve \
  --model /app/models/Qwen2.5-VL-3B-Instruct \
  --host 0.0.0.0 \
  --port 8000
```

### 服务接口

- `GET /health`：返回模型与启动配置
- `GET /v1/memory`：返回当前显存画像
- `POST /v1/generate`：非流式生成（单条）
- `POST /v1/generate_stream`：SSE 流式生成
- `POST /v1/batch_generate`：批量生成

### VLM 多模态请求示例

使用 `/v1/generate` 接口同时发送图像和文本进行推理，`images` 数组支持本地路径（服务端）、Base64 编码以及 HTTP/HTTPS URL：

```bash
curl -X POST http://127.0.0.1:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "这张图片有什么？",
    "images": [
      "https://www.w3.org/html/logo/downloads/HTML5_Logo_512.png"
    ],
    "sampling_params": {
      "temperature": 0.2,
      "max_tokens": 128
    }
  }'
```

### 非流式请求示例 (文本)

```bash
curl -X POST http://127.0.0.1:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, my name is",
    "sampling_params": {
      "temperature": 0.0,
      "max_tokens": 32
    },
    "return_token_ids": true,
    "profile": true
  }'
```

### 流式请求示例

```bash
curl -N -X POST http://127.0.0.1:8000/v1/generate_stream \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain CUDA Graph.",
    "sampling_params": {
      "temperature": 0.0,
      "max_tokens": 32
    },
    "profile": true
  }'
```

### 批量请求示例

```bash
curl -X POST http://127.0.0.1:8000/v1/batch_generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": ["Hello", "The capital of France is", "1 + 1 ="],
    "sampling_params": {
      "temperature": 0.0,
      "max_tokens": 16
    },
    "return_token_ids": true,
    "profile": true
  }'
```

### VLM 批量请求示例

```bash
# 批量多模态推理
curl -X POST http://127.0.0.1:8000/v1/batch_generate \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {
        "text": "描述这张图片",
        "images": ["https://example.com/image1.jpg"]
      },
      {
        "text": "这是什么？",
        "images": ["https://example.com/image2.jpg"]
      }
    ],
    "sampling_params": {
      "temperature": 0.2,
      "max_tokens": 64
    }
  }'
```

请求体字段（`/v1/generate` 和 `/v1/generate_stream`）：
- `prompt`：`str` 或 `list[int]`
- `images`：`list[str]`（仅 VLM 服务支持，支持 Base64 或 HTTP URL）
- `sampling_params.temperature`
- `sampling_params.top_k`
- `sampling_params.top_p`
- `sampling_params.max_tokens`
- `sampling_params.ignore_eos`
- `return_token_ids`：是否在最终结果中返回 token ids
- `profile`：是否返回 TTFT、速度、显存画像
- `apply_chat_template`：是否自动使用对话模板拼接（默认 true）

说明：
- 服务启动时的 `max_model_len` 是全局上限，单个请求不能超过这个预算
- 服务复用同一套 `generate_stream()` 实现，因此 CLI 与 HTTP 输出指标口径一致

## 测试

```bash
pytest tests/ -v
```

当前结果：`3 passed` (VLM 测试套件)

完整功能验证：
- ✅ 文本单次推理
- ✅ 文本批量推理  
- ✅ 文本流式生成
- ✅ VLM 单图推理
- ✅ VLM 批量推理
- ✅ HTTP 服务接口

## 性能数据

### 文本推理 (Qwen2.5-1.5B)
- **MiniLLM**: ~310 tok/s
- **HuggingFace**: ~34 tok/s
- **加速比**: 9.2x

### VLM 推理 (Qwen2.5-VL-3B)
- **Prefill**: 0.65s (MiniLLM) vs 1.08s (HF)
- **Decode**: 25.93 tok/s (MiniLLM) vs 38.73 tok/s (HF)
- **显存**: 8.62 GiB allocated (MiniLLM) vs 10.36 GiB (HF)

详细性能报告：`/app/minillm/docs/performance_report.md`

## Benchmark

详细性能报告：`/app/minillm/docs/performance_report.md`

### 快速对比

**文本推理 (Qwen2.5-1.5B)**
```bash
python benchmarks/benchmark_text_qwen2.py
```
- MiniLLM: ~310 tok/s
- HuggingFace: ~34 tok/s  
- 加速比: 9.2x

**VLM 推理 (Qwen2.5-VL-3B)**
```bash
python benchmarks/benchmark_vlm_qwen2_5.py
```
- Prefill: 0.65s (MiniLLM) vs 1.08s (HF)
- Decode: 25.93 tok/s (MiniLLM) vs 38.73 tok/s (HF)
- 显存: 8.62 GiB (MiniLLM) vs 10.36 GiB (HF)

## 当前限制

- 仅支持单卡推理
- 服务层串行化处理请求（非高并发调度）
- VLM 批量推理为串行单请求路径，未实现原生 batch
- 尚未实现 InternVL、Qwen3-VL、量化后端

## 路线图

- [ ] VLM 原生 batch 推理
- [ ] 支持更多多模态模型 (InternVL, Qwen3-VL)
- [ ] 量化支持 (INT8, INT4)
- [ ] 更多采样策略
- [ ] 性能优化与 benchmark 扩展


