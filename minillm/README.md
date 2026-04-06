# MINILLM 推理框架

极致简化的 LLM 推理框架，用于将大语言模型部署到边缘设备（单卡推理）。

## 最新状态

- 当前已支持 `qwen2.5-1.5B-Instruct` 与 `Qwen3-0.6B` 单卡文本推理
- 🎉 **新增：支持 `Qwen2.5-VL` 多模态单卡推理（端到端）**
- 已修复“输出乱码”、`max_tokens=1024` 长度溢出、`sampling_params` 静默截断、KV 占用统计歧义等问题
- 已提供命令行推理接口与常驻 HTTP 服务接口
- 当前测试结果：`79 passed`（全部测试，基于 `conda run -n ramc pytest minillm/tests tests -q`）
- 最近 VLM 专项验证：`minillm/tests/test_vlm_engine.py` 为 `3 passed`
- 当前 VLM benchmark 已补齐 `prefill / decode / batch` 维度
- **2026-04-06 更新**：修复 VLM 视觉编码器 RoPE 应用后，语义输出已与 HF 对齐（cosine similarity 0.922）
- **2026-04-06 更新**：统一公开 VLM 入口为 `minillm.engine.vlm_engine.VLMEngine`，`MultimodalEngine` 仅保留为内部核心实现
- **2026-04-06 更新**：已在 `ramc` 环境重跑文本与 VLM benchmark；VLM `batch=16` 时，repeat 为 `272.55 vs 29.08 / 40.84 tok/s`，diverse 为 `174.54 vs 35.77 / 62.80 tok/s`（minillm vs HF sdpa / HF flash_attention_2）

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

说明：`flash-attn` 是本项目文本模型与多模态模型共用的基础依赖，应在项目运行环境中统一安装；当前验证环境为 `conda` 的 `ramc` 环境。

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
- 对外公开接口统一为 `minillm.engine.vlm_engine.VLMEngine`
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
  --images /app/minillm/img/image.png https://example.com/image2.png \
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
- `--profile`：输出精简后的请求级性能摘要
- `--max-model-len`：最大上下文长度；若不传，CLI 自动按最长 prompt + max_tokens 推导
- `--no-chat-template`：禁用自动对话模板拼接，允许用户自由续写

显存画像输出已精简为 4 行摘要：
- `Latency`：TTFT 与总耗时
- `Tokens`：prompt token 数与生成 token 数
- `Throughput`：整体吞吐与 decode 吞吐
- `Memory`：模型显存、KV 峰值/总量、CUDA 峰值 allocated/reserved

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
- `GET /v1/memory`：返回当前显存画像（精简字段，保留 used/peak 区分）
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

### `profile` 响应字段

当请求体中传入 `"profile": true` 时，`metrics` 会返回精简后的请求级摘要：
- `ttft_seconds`
- `total_time_seconds`
- `prompt_tokens`
- `generated_tokens`
- `overall_tokens_per_second`
- `decode_tokens_per_second`
- `model_gib`
- `kv_cache_total_gib`
- `kv_cache_peak_gib`
- `kv_cache_peak_blocks`
- `kv_cache_total_blocks`
- `cuda_peak_allocated_gib`
- `cuda_peak_reserved_gib`

`GET /v1/memory` 返回的是实时显存画像，字段为：
- `model_gib`
- `kv_cache_total_gib`
- `kv_cache_used_gib`
- `kv_cache_peak_gib`
- `kv_cache_used_blocks`
- `kv_cache_peak_blocks`
- `kv_cache_total_blocks`
- `cuda_allocated_gib`
- `cuda_peak_allocated_gib`
- `cuda_reserved_gib`
- `cuda_peak_reserved_gib`

简化后的非流式响应示例：

```json
{
  "text": "Hello, my name is...",
  "token_ids": [9707, 11, 847, 374],
  "metrics": {
    "ttft_seconds": 0.12,
    "total_time_seconds": 0.56,
    "prompt_tokens": 8,
    "generated_tokens": 16,
    "overall_tokens_per_second": 28.5,
    "decode_tokens_per_second": 40.2,
    "model_gib": 7.02,
    "kv_cache_total_gib": 0.56,
    "kv_cache_peak_gib": 0.46,
    "kv_cache_peak_blocks": 26,
    "kv_cache_total_blocks": 32,
    "cuda_peak_allocated_gib": 8.63,
    "cuda_peak_reserved_gib": 14.88
  }
}
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
- `profile`：是否返回精简后的请求级性能摘要
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

### 文本推理
- **Qwen2.5-1.5B / batch=8 / input=128**：MiniLLM `1271.41 tok/s`，HF `sdpa` `317.25 tok/s`，HF `flash_attention_2` `272.53 tok/s`
- **Qwen3-0.6B / batch=8 / input=128**：MiniLLM `1372.83 tok/s`，HF `sdpa` `278.50 tok/s`，HF `flash_attention_2` `238.41 tok/s`
- **结论**：本轮 `ramc` 实测下，文本路径 MiniLLM 对 HF 的优势约为 `4x` 到 `5.76x`
- **备注**：在这台机器上，HF 文本路径的 `flash_attention_2` 没有跑赢 `sdpa`

### VLM 推理 (Qwen2.5-VL-3B)
- **repeat / batch=16 / overall**：MiniLLM `272.55 tok/s`，HF `sdpa` `29.08 tok/s`，HF `flash_attention_2` `40.84 tok/s`
- **diverse / batch=16 / overall**：MiniLLM `174.54 tok/s`，HF `sdpa` `35.77 tok/s`，HF `flash_attention_2` `62.80 tok/s`
- **repeat / batch=16 / decode**：MiniLLM `584.23 tok/s`，HF `sdpa` `631.67 tok/s`，HF `flash_attention_2` `710.22 tok/s`
- **diverse / batch=16 / decode**：MiniLLM `587.53 tok/s`，HF `sdpa` `164.79 tok/s`，HF `flash_attention_2` `365.03 tok/s`
- **高 batch 显存保留**：repeat 为 `23.39 / 70.84 / 69.79 GiB`，diverse 为 `23.63 / 57.55 / 55.87 GiB`（MiniLLM / HF sdpa / HF flash_attention_2）
- **备注**：小 batch 不是 MiniLLM 的绝对优势区间，例如 `repeat / batch=1` 时 HF overall throughput 略高于 MiniLLM
- **语义对齐**：cosine similarity `0.922`

详细性能报告：`/app/minillm/docs/BENCHMARK.md`

## Benchmark

详细使用说明、主线结果文件与归档策略见：`/app/minillm/docs/BENCHMARK.md`

当前主线只保留：
- `benchmarks/benchmark_text_qwen2.py`
- `benchmarks/benchmark_vlm_qwen2_5.py`
- `benchmark_results/qwen2_text_benchmark.*`
- `benchmark_results/qwen3_text_benchmark.*`
- `benchmark_results/qwen2_5_vlm_benchmark.*`

补充实验结果和旧脚本已归档到：`/app/minillm/archive/benchmark_assets/`

## 当前限制

- 仅支持单卡推理
- 服务层串行化处理请求（非高并发调度）
- VLM 已支持原生 batch decode，当前视觉 prefill 仍按请求顺序执行
- **VLM decode 阶段暂时禁用 CUDA Graph**（因图像 token 数量不可预测导致 block_tables 维度不匹配，详见 `docs/CUDA_GRAPH_ANALYSIS.md`）
- 尚未实现 InternVL、Qwen3-VL、量化后端

## 路线图

- [ ] VLM 首轮视觉 prefill 合批与视觉编码并行化
- [ ] VLM CUDA Graph 支持（需解决动态 block_tables 维度问题，详见 `docs/CUDA_GRAPH_ANALYSIS.md`）
- [ ] 支持更多多模态模型 (InternVL, Qwen3-VL)
- [ ] 量化支持 (INT8, INT4)
- [ ] 更多采样策略
- [ ] 性能优化与 benchmark 扩展


