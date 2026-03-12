# vLLM Batch Runner

面向大规模 rollout 的 YAML 驱动批量推理框架，适配 OpenAI-compatible / vLLM 服务，重点优化了多服务器、高并发、超大规模 JSONL 数据集场景。

## 适用场景

- 40 到 400 个 vLLM server
- 2048 到 8196 并发批量请求
- 文本与多模态混合数据
- 数亿级 JSONL / Directory JSONL rollout
- 需要可恢复、可观测、可扩展的长期任务

## 现在的关键能力

- 共享 `httpx.AsyncClient`，请求走单一异步热路径
- producer / scheduler / writer 三段式流水线
- saver 批量写入，结果落盘与请求协程解耦
- `p2c_cost_aware` 负载均衡，适合大 server 池
- bitmap resume，避免启动时扫描全部历史输出
- `minimal/full` 输出投影，降低序列化和磁盘开销
- 多模态图片编码线程池，避免阻塞主事件循环
- async health check，支持有界并发探测

## 安装

```bash
pip install -r requirements.txt
pip install -e ".[dev]"
```

如果希望启用更快的 JSON 序列化，可额外安装：

```bash
pip install -e ".[perf]"
```

## 快速开始

### 1. 准备 server 标记文件

`VLLMServerManager` 当前按文件名发现 server，约定格式为 `server_<ip>_<port>`。这里是“标记文件”，不是目录。

```text
vllm_servers/
├── server_127.0.0.1_8000
├── server_127.0.0.1_8001
└── server_127.0.0.1_8002
```

### 2. 准备输入数据

```jsonl
{"id": "1", "prompt": "法国的首都是什么？"}
{"id": "2", "prompt": "解释量子计算。"}
```

### 3. 编写配置

```yaml
loader:
  class: JSONLDataLoader
  params:
    file_path: data/prompts.jsonl
    streaming: true

saver:
  class: JSONLResultSaver
  params:
    output_path: outputs/results.jsonl
    output_projection: minimal
    immediate_flush: false

runner:
  model_name: meta-llama/Llama-2-7b-chat-hf
  servers_dir: ./vllm_servers
  max_concurrency: 256
  load_balancing_strategy: p2c_cost_aware
  producer_prefetch: 512
  writer_queue_size: 1024
  writer_batch_size: 128
  writer_flush_interval_ms: 100
  resume: true
  resume_backend: bitmap

logging:
  level: INFO
```

### 4. 运行

```bash
python -m src.cli --config configs/config.yaml
```

也可以使用 console entrypoint：

```bash
vllm-batch --config configs/config.yaml
```

## 运行模型时的真实数据流

1. loader 在后台 producer 线程里读取数据。
2. scheduler 维持 `max_concurrency` 个 in-flight 请求。
3. 请求完成后只把 `SaveResult` 放进 completion queue。
4. writer worker 批量调用 `saver.save_batch()` 落盘。
5. progress 和 token 统计在批量写入后更新。
6. resume backend 在保存成功后标记完成状态。

这意味着请求协程不再逐条等待磁盘写入。

## 核心配置

### Loader

内建 loader：

- `JSONDataLoader`
- `JSONLDataLoader`
- `CSVDataLoader`
- `PromptListLoader`
- `DirectoryJSONLDataLoader`
- `MultimodalJSONDataLoader`
- `MultimodalJSONLDataLoader`
- `MultimodalDirectoryJSONLDataLoader`

`JSONLDataLoader` 和 `DirectoryJSONLDataLoader` 是大规模 rollout 的主路径。

### Saver

内建 saver：

- `JSONResultSaver`
- `JSONLResultSaver`
- `CSVResultSaver`
- `ConsoleResultSaver`
- `DirectoryJSONLResultSaver`

推荐：

- 单文件输出用 `JSONLResultSaver`
- 多目录输入镜像输出用 `DirectoryJSONLResultSaver`

### Runner

高并发相关参数：

- `max_concurrency`
- `http_max_connections`
- `http_max_keepalive_connections`
- `load_balancing_strategy`
- `selection_sample_size`
- `max_inflight_cost`
- `producer_prefetch`
- `writer_queue_size`
- `writer_batch_size`
- `writer_flush_interval_ms`
- `writer_workers`
- `resume`
- `resume_backend`
- `image_encode_workers`

### 负载均衡策略

当前支持：

- `round_robin`
- `least_connections`
- `adaptive_round_robin`
- `load_aware_round_robin`
- `p2c_cost_aware`
- `random`

对于 40 到 400 台 server 的高并发 rollout，推荐优先使用 `p2c_cost_aware`。

## Resume

### `legacy_output_scan`

- 兼容旧行为
- 通过 saver 扫描已有输出判断是否完成
- 对超大历史输出启动成本高

### `bitmap`

- 适合 `JSONLDataLoader` / `DirectoryJSONLDataLoader` 主路径
- 使用 `resume_key=(source_file, line_num, item_idx)` 精确记录完成状态
- 内存占用稳定，不需要把全部 `request_id` 放进内存

如果 loader 不能提供 `resume_key`，会自动回退到 legacy 行为。

## 输出投影

`JSONLResultSaver` 和 `DirectoryJSONLResultSaver` 支持：

- `output_projection: full`
- `output_projection: minimal`

`full` 保持兼容，默认包含完整 `model_output` 与更多元数据。  
`minimal` 只保留高吞吐场景常用字段，更适合海量 rollout。

可选字段：

- `output_fields`
- `include_timestamp`

## 多模态

多模态 loader 支持：

- `image`
- `images`
- 相对路径 + `image_base_dir`
- URL
- 已编码的 `data:` URI

关键参数：

- `encode_images`
- `image_encode_workers`
- `image_base_dir`

当 `encode_images: true` 且 `image_encode_workers > 1` 时，图片编码会转移到线程池。

## 自定义组件

### 使用注册器

```python
from pathlib import Path

from src.loaders.base import DataLoader, LoadResult
from src.utils.registry import register_loader


@register_loader
class MyLoader(DataLoader):
    def _initialize(self):
        self.path = Path(self.config["file_path"])

    def load(self):
        for index, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            yield LoadResult(
                messages=[{"role": "user", "content": line}],
                request_id=f"line_{index}",
            )
```

```yaml
custom_modules:
  - custom_components.py

loader:
  class: MyLoader
  params:
    file_path: data/raw.txt
```

### JSONL 定制入口

如果你需要改 JSONL 行解析逻辑，优先继承现有 loader 并覆写：

- `parse_line()`
- `should_skip_item()`
- `extract_request_id()`
- `extract_prompt()`
- `extract_additional_data()`

详见 [docs/JSONL_CUSTOMIZATION.md](docs/JSONL_CUSTOMIZATION.md)。

## 推荐高并发配置

仓库内置了 [configs/high_concurrency_config.yaml](configs/high_concurrency_config.yaml)，已经启用：

- `DirectoryJSONLDataLoader`
- `DirectoryJSONLResultSaver`
- `p2c_cost_aware`
- `resume_backend: bitmap`
- `output_projection: minimal`
- batched writer

运行方式：

```bash
python -m src.cli --config configs/high_concurrency_config.yaml
```

## 基准和 soak 脚本

### 负载均衡微基准

```bash
python scripts/benchmark_load_balancer.py --servers 400 --iterations 50000
```

### JSONL writer 微基准

```bash
python scripts/benchmark_writer.py --rows 50000 --projection minimal
```

### 本地 stub soak

```bash
python scripts/soak_openai_stub.py \
  --servers 8 \
  --requests 2000 \
  --concurrency 512 \
  --files 8 \
  --strategy p2c_cost_aware \
  --report-json
```

这个脚本会：

- 启动本地 OpenAI-compatible mock server
- 生成 directory JSONL 数据集
- 创建 server 标记文件
- 运行真实 `BatchRunner`
- 输出吞吐、失败数、重试数、token 数

## 文档导航

- [docs/FRAMEWORK_GUIDE.md](docs/FRAMEWORK_GUIDE.md)
- [docs/HIGH_CONCURRENCY_OPTIMIZATION.md](docs/HIGH_CONCURRENCY_OPTIMIZATION.md)
- [docs/MULTIMODAL.md](docs/MULTIMODAL.md)
- [docs/JSONL_CUSTOMIZATION.md](docs/JSONL_CUSTOMIZATION.md)

## 常用命令

```bash
python -m pytest tests/ -v
python -m pytest tests/test_load_balancer.py -v
python -m pytest tests/test_multimodal.py -v
black src tests scripts
flake8 src tests scripts
```
