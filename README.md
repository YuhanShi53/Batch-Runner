# vLLM Batch Runner

一个灵活、生产级的 vLLM 批量推理框架，支持可插拔的数据加载器、结果保存器和模型 API 适配器。

## 功能特性

### 核心功能

- **可插拔架构**: 支持自定义数据加载器、结果保存器和模型 API 适配器
- **流式处理**: 支持流式数据加载和结果保存，内存占用恒定
- **并发处理**: 多 vLLM 服务器间的高效负载均衡
- **容错机制**: 指数退避的自动重试
- **多次 Rollout**: 对每个样本运行多次以进行一致性分析
- **配置驱动**: 基于 YAML 的全组件配置
- **服务器发现**: 从目录结构自动发现 vLLM 服务器
- **健康检查**: 自动服务器健康监控和动态故障转移
- **断点续跑**: 支持任务中断后从检查点恢复

### 新增功能 🆕

- **流式处理模式**: 所有 Loader/Saver 支持流式处理，内存占用恒定
- **灵活的定制钩子**: 清晰的可重写接口，支持自定义 prompt、messages 和多模态构造
- **注册机制**: 无需修改源码即可注册自定义 Loader/Saver
- **多模态支持**: 原生支持文本+图像的混合输入输出
- **增强的 Mixin 系统**: 丰富的 Mixin 类，轻松实现自定义功能

### 多模态支持

- 支持文本+图像的混合输入
- 自动图像 base64 编码
- OpenAI Vision API 兼容格式
- 灵活的图像路径解析

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 准备 vLLM 服务器

启动 vLLM 服务器。服务器目录命名格式应为 `server_{ip}_{port}`：

```
vllm_servers/
├── server_127.0.0.1_8000/
├── server_127.0.0.1_8001/
└── server_127.0.0.1_8002/
```

### 2. 准备数据

创建 JSON 文件：

```json
[
  {"id": "1", "prompt": "法国的首都是什么？"},
  {"id": "2", "prompt": "解释量子计算"}
]
```

### 3. 配置批处理任务

创建 `configs/config.yaml`：

```yaml
loader:
  class: JSONDataLoader
  params:
    file_path: data/prompts.json

saver:
  class: JSONResultSaver
  params:
    output_path: outputs/results.json

runner:
  max_concurrency: 10
  model_name: meta-llama/Llama-2-7b-chat-hf
  servers_dir: ./vllm_servers
```

### 4. 运行批量推理

```bash
python -m src.cli --config configs/config.yaml
```

## 配置说明

### DataLoader 选项

#### 文本数据加载器

- **JSONDataLoader**: 从 JSON 文件加载（支持流式处理）
- **JSONLDataLoader**: 从 JSONL 文件加载（**推荐：流式处理**）
- **CSVDataLoader**: 从 CSV 文件加载
- **PromptListLoader**: 从配置文件加载提示列表

#### 目录遍历加载器

- **DirectoryJSONLDataLoader**: 从目录树递归加载 conv.jsonl 文件（**推荐：流式处理**）
  - 保留输入目录结构
  - 自动添加源文件元数据
  - 支持大规模数据集

#### 多模态数据加载器

- **MultimodalJSONDataLoader**: JSON + 图像
- **MultimodalJSONLDataLoader**: JSONL + 图像
- **MultimodalDirectoryJSONLDataLoader**: 目录 JSONL + 图像

**多模态数据格式示例**:

```jsonl
{"id": "1", "prompt": "描述这张图片", "image": "path/to/image.jpg"}
{"id": "2", "prompt": "比较这些图片", "images": ["img1.jpg", "img2.png"]}
```

### ResultSaver 选项

- **JSONResultSaver**: 保存到 JSON 文件（批量写入）
- **JSONLResultSaver**: 保存到 JSONL 文件（**推荐：流式写入**）
- **CSVResultSaver**: 保存到 CSV 文件
- **ConsoleResultSaver**: 打印到控制台（用于调试）
- **DirectoryJSONLResultSaver**: 按目录结构保存结果

### 流式处理配置 🆕

启用流式处理模式（推荐用于大数据集）：

```yaml
loader:
  class: JSONLDataLoader
  params:
    file_path: data/large_dataset.jsonl
    streaming: true  # 启用流式处理（默认）

saver:
  class: JSONLResultSaver
  params:
    output_path: results/output.jsonl
    streaming: true  # 启用流式写入（默认）
    immediate_flush: true  # 立即刷新到磁盘
```

**流式处理优势**:
- ✅ 内存占用恒定（O(队列大小)）
- ✅ 立即开始处理
- ✅ 更高的容错性
- ✅ 适合百万级数据集

### 注册自定义组件 🆕

#### 方式 1: 使用装饰器（推荐）

```python
# custom_components.py
from src.loaders.base import DataLoader, LoadResult
from src.utils.registry import register_loader

@register_loader
class MyCustomLoader(DataLoader):
    def _initialize(self):
        self.file_path = Path(self.config['file_path'])

    def load(self):
        with open(self.file_path) as f:
            for line in f:
                yield LoadResult(
                    messages=[{"role": "user", "content": line.strip()}],
                    request_id=str(hash(line))
                )
```

在配置中使用：

```yaml
# config.yaml
custom_modules:
  - custom_components.py

loader:
  class: MyCustomLoader
  params:
    file_path: data/my_data.txt
```

#### 方式 2: 直接导入

```python
# main.py
import custom_components  # 注册组件
from src.cli import main
main()
```

### Model Adapter 选项

- **OpenAIAdapter**: OpenAI 兼容 API（默认，适用于 vLLM）
- **SimpleAdapter**: 简单 prompt-based API
- **自定义**: 继承 `ModelAdapter` 创建自己的适配器

### System Prompts

为所有请求设置系统提示：

```yaml
runner:
  system_prompt: "你是一个有帮助的助手，提供准确的答案。"
```

### 断点续跑 (Checkpoint / Resume)

启用检查点功能以支持中断任务恢复：

```yaml
runner:
  enable_checkpoint: true
  checkpoint_path: checkpoints/my_job_checkpoint.json
  checkpoint_interval: 50  # 每 50 个请求保存一次检查点
```

启用后：
- 进度会定期自动保存
- 任务中断后重新运行相同命令即可恢复
- 已完成的请求会被跳过
- 任务成功完成后自动删除检查点

### 负载均衡策略

- **round_robin**: 顺序分发请求
- **least_connections**: 发送到当前活动请求数最少的服务器
- **random**: 随机分发

## 自定义组件开发

### 使用 Mixin 快速开发 🆕

#### 1. 流式 JSONL Loader

```python
from src.loaders.format_mixins import StreamingJSONLLoader

class MyLoader(StreamingJSONLLoader):
    """继承所有流式处理逻辑"""
    pass
```

#### 2. 自定义 Prompt 提取

```python
from src.loaders.jsonl_mixin import JSONLLoaderMixin
from src.loaders.base import DataLoader

class MyLoader(JSONLLoaderMixin, DataLoader):
    def extract_prompt(self, item):
        # 尝试多个字段
        for field in ['prompt', 'question', 'text', 'input']:
            if field in item:
                return str(item[field])
        return None
```

#### 3. 自定义 Messages 构建

```python
from src.loaders.streaming_mixin import MessagesBuilderMixin
from src.loaders.base import DataLoader

class MyLoader(MessagesBuilderMixin, DataLoader):
    def build_messages(self, prompt, additional_data=None):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

        # 添加对话历史
        if additional_data and 'history' in additional_data:
            messages.extend(additional_data['history'])

        messages.append({"role": "user", "content": prompt})
        return messages
```

#### 4. 自定义 Saver 输出格式

```python
from src.savers.jsonl_mixin import JSONLSaverMixin
from src.savers.base import ResultSaver

class MySaver(JSONLSaverMixin, ResultSaver):
    def format_result(self, result):
        content = result.model_output['choices'][0]['message']['content']
        return {
            "id": result.request_id,
            "response": content,
            "tokens": result.model_output.get('usage', {}).get('total_tokens', 0)
        }
```

### 传统方式创建组件

#### 自定义 Loader

```python
# src/loaders/my_loader.py
from .base import DataLoader, LoadResult

class MyCustomLoader(DataLoader):
    def _initialize(self):
        self.data = [...]  # 加载数据

    def load(self):
        for item in self.data:
            yield LoadResult(
                messages=[{"role": "user", "content": item['text']}],
                request_id=item['id'],
                additional_data={'metadata': item['metadata']}
            )
```

#### 自定义 Saver

```python
# src/savers/my_saver.py
from .base import ResultSaver, SaveResult

class MyCustomSaver(ResultSaver):
    def _initialize(self):
        self.file = open(self.config['output_path'], 'w')

    def save(self, result: SaveResult):
        content = result.model_output['choices'][0]['message']['content']
        self.file.write(f"{result.request_id}\t{content}\n")

    def cleanup(self):
        self.file.close()
```

#### 自定义 Adapter

如果模型 API 不是 OpenAI 格式，创建自定义适配器：

```python
# src/adapters/my_adapter.py
from .base import ModelAdapter
from typing import Dict, Any, List
import requests

class MyCustomAdapter(ModelAdapter):
    def build_request(self, model_name, messages, temperature, max_tokens, **kwargs):
        # 转换为你的 API 格式
        return {
            "model": model_name,
            "input": messages[0]["content"],
            "max_length": max_tokens,
        }

    def parse_response(self, response: requests.Response) -> Dict[str, Any]:
        data = response.json()
        # 转换响应为 OpenAI 格式
        return {
            "choices": [{
                "message": {"role": "assistant", "content": data["result"]},
                "finish_reason": "stop"
            }],
            "usage": {"total_tokens": data.get("tokens", 0)}
        }

    def get_chat_url(self, base_url: str) -> str:
        return f"{base_url}/my_custom_endpoint"
```

在配置中使用：

```yaml
runner:
  adapter_class: MyCustomAdapter
  adapter_params: {}
```

## 命令行选项

```bash
python -m src.cli --config configs/config.yaml [选项]

选项:
  --config, -c       配置文件路径
  --concurrency      覆盖最大并发数
  --rollouts         覆盖每个样本的 rollout 次数
  --model            覆盖模型名称
  --temperature      覆盖采样温度
  --max-tokens       覆盖最大生成 token 数
  --verbose, -v      启用详细日志
```

## 使用示例

### 流式处理大数据集 🆕

```yaml
# configs/streaming_config.yaml
loader:
  class: DirectoryJSONLDataLoader
  params:
    input_dir: data/millions_of_conversations
    streaming: true

saver:
  class: DirectoryJSONLResultSaver
  params:
    output_dir: results
    streaming: true
    immediate_flush: true

runner:
  max_concurrency: 20
  enable_checkpoint: true
  checkpoint_interval: 100
```

### 多模态推理 🆕

```yaml
# configs/multimodal_config.yaml
loader:
  class: MultimodalJSONLDataLoader
  params:
    file_path: data/vqa_data.jsonl
    image_base_dir: data/images
    encode_images: true

saver:
  class: JSONLResultSaver
  params:
    output_path: results/vqa_output.jsonl

runner:
  model_name: "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
```

### 多次 Rollout

每个样本运行 3 次：

```bash
python -m src.cli --config configs/config.yaml --rollouts 3
```

### 高并发

```bash
python -m src.cli --config configs/config.yaml --concurrency 50
```

### 自定义模型参数

```bash
python -m src.cli --config configs/config.yaml --temperature 0.5 --max-tokens 2000
```

### 使用断点续跑

1. 启用检查点功能运行任务：
```bash
python -m src.cli --config configs/examples/with_checkpoint.yaml
```

2. 如果任务中断（Ctrl+C 或错误），重新运行相同命令即可恢复

3. 任务完成后检查点自动删除

## 项目结构

```
vllm_runner/
├── src/
│   ├── loaders/          # 数据加载器实现
│   │   ├── base.py              # 抽象基类
│   │   ├── streaming_mixin.py   # 流式处理 Mixin
│   │   ├── jsonl_mixin.py       # JSONL 解析 Mixin
│   │   ├── format_mixins.py     # 格式专用 Mixin
│   │   ├── json_loader.py       # JSON 加载器
│   │   ├── jsonl_loader.py      # JSONL 加载器（流式）
│   │   ├── csv_loader.py        # CSV 加载器
│   │   └── directory_jsonl_loader.py  # 目录 JSONL 加载器（流式）
│   ├── savers/           # 结果保存器实现
│   │   ├── base.py              # 抽象基类
│   │   ├── streaming_mixin.py   # 流式处理 Mixin
│   │   ├── jsonl_mixin.py       # JSONL 格式化 Mixin
│   │   ├── format_mixins.py     # 格式专用 Mixin
│   │   ├── json_saver.py        # JSON 保存器
│   │   ├── jsonl_saver.py       # JSONL 保存器（流式）
│   │   ├── csv_saver.py         # CSV 保存器
│   │   └── directory_jsonl_saver.py  # 目录 JSONL 保存器（流式）
│   ├── adapters/         # 模型 API 适配器实现
│   ├── servers/          # 服务器管理和负载均衡
│   ├── utils/            # 工具模块
│   │   ├── config.py            # 配置加载
│   │   └── registry.py          # 组件注册系统 🆕
│   ├── batch_runner.py   # 主协调器
│   └── cli.py            # 命令行接口
├── docs/                 # 文档 🆕
│   ├── FRAMEWORK_GUIDE.md # 框架完整指南
│   └── REFACTORING_SUMMARY.md  # 重构总结
├── examples/             # 示例代码 🆕
│   └── custom_components.py    # 自定义组件示例
├── configs/              # 配置文件
│   └── examples/         # 示例配置
├── logs/                 # 日志文件
├── outputs/              # 输出文件
├── checkpoints/          # 检查点文件
└── requirements.txt
```

## 核心模块说明

### 1. DataLoader ([`src/loaders/`](src/loaders/))

#### 基类和 Mixin
- **base.py**: 抽象基类，定义数据加载接口
- **streaming_mixin.py**: 流式处理、消息构建、提示提取 Mixin
- **jsonl_mixin.py**: JSONL 解析 Mixin
- **format_mixins.py**: JSON/JSONL/CSV/目录 格式 Mixin

#### 内置实现
- **json_loader.py**: JSON 文件加载器
- **jsonl_loader.py**: JSONL 文件加载器（支持流式）
- **csv_loader.py**: CSV 文件加载器
- **prompt_list_loader.py**: 简单列表加载器
- **directory_jsonl_loader.py**: 目录 JSONL 加载器（流式）
- **multimodal_base.py**: 多模态支持基类

### 2. ResultSaver ([`src/savers/`](src/savers/))

#### 基类和 Mixin
- **base.py**: 抽象基类，定义结果保存接口
- **streaming_mixin.py**: 流式写入、输出格式化 Mixin
- **jsonl_mixin.py**: JSONL 格式化 Mixin
- **format_mixins.py**: JSON/JSONL/CSV/目录 格式 Mixin

#### 内置实现
- **json_saver.py**: JSON 文件保存器（批量写入）
- **jsonl_saver.py**: JSONL 文件保存器（流式写入）
- **csv_saver.py**: CSV 文件保存器
- **console_saver.py**: 控制台输出
- **directory_jsonl_saver.py**: 目录 JSONL 保存器（流式）

### 3. ModelAdapter ([`src/adapters/`](src/adapters/))
- **base.py**: 抽象基类，定义 API 适配器接口
- **openai_adapter.py**: OpenAI 兼容 API 适配器
- **simple_adapter.py**: 简单 prompt API 适配器

### 4. 注册系统 ([`src/utils/registry.py`](src/utils/registry.py)) 🆕

装饰器式组件注册：
- `@register_loader`: 注册自定义 Loader
- `@register_saver`: 注册自定义 Saver
- 支持从配置文件自动加载自定义模块
- 无需修改项目源码

### 5. BatchRunner ([`src/batch_runner.py`](src/batch_runner.py))
主协调器，负责：
- 加载数据并生成 rollout
- 初始化服务器管理器和负载均衡器
- 并发处理请求（支持流式模式）
- 失败重试
- 断点续跑

### 6. 工具模块 ([`src/utils/`](src/utils/))
- **config.py**: YAML 配置加载和动态导入
- **logger.py**: 日志配置
- **retry.py**: 带指数退避的重试装饰器
- **progress.py**: 进度跟踪
- **checkpoint.py**: 断点续跑检查点管理
- **registry.py**: 组件注册系统 🆕

## 最佳实践

### 1. 大规模数据处理 🆕

启用流式处理和断点续跑：

```yaml
loader:
  class: DirectoryJSONLDataLoader
  params:
    input_dir: data/millions_of_files
    streaming: true  # 恒定内存占用

runner:
  enable_checkpoint: true
  checkpoint_interval: 100
  max_concurrency: 20
```

### 2. 调试模式

使用 ConsoleResultSaver 和详细日志：

```yaml
saver:
  class: ConsoleResultSaver
  params:
    show_details: true

logging:
  level: DEBUG
```

### 3. 生产环境

使用 JSONL Saver（流式）和合理的并发数：

```yaml
saver:
  class: JSONLResultSaver
  params:
    output_path: outputs/results_$(date).jsonl
    streaming: true
    immediate_flush: true

runner:
  max_concurrency: 10
  enable_checkpoint: true
```

### 4. 自定义数据处理 🆕

使用 Mixin 快速实现自定义逻辑：

```python
from src.loaders.jsonl_mixin import JSONLLoaderMixin
from src.loaders.streaming_mixin import MessagesBuilderMixin
from src.loaders.base import DataLoader

class CustomLoader(JSONLLoaderMixin, MessagesBuilderMixin, DataLoader):
    """结合多个 Mixin 获得完整功能"""

    def extract_prompt(self, item):
        # 自定义 prompt 提取
        return item.get('custom_field')

    def build_messages(self, prompt, additional_data=None):
        # 自定义 messages 构建
        return [
            {"role": "system", "content": "Custom system prompt"},
            {"role": "user", "content": prompt}
        ]
```

## 文档

- 📘 [框架完整指南](docs/FRAMEWORK_GUIDE.md) - 详细的框架使用文档
- 📗 [重构总结](docs/REFACTORING_SUMMARY.md) - 最新重构的详细说明
- 💡 [自定义组件示例](examples/custom_components.py) - 完整的代码示例

## 许可证

MIT License
