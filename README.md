# vLLM Batch Runner

一个灵活、生产级的 vLLM 批量推理框架，支持可插拔的数据加载器、结果保存器和模型 API 适配器。

## 功能特性

- **可插拔架构**: 支持自定义数据加载器、结果保存器和模型 API 适配器
- **并发处理**: 多 vLLM 服务器间的高效负载均衡
- **容错机制**: 指数退避的自动重试
- **多次 Rollout**: 对每个样本运行多次以进行一致性分析
- **配置驱动**: 基于 YAML 的全组件配置
- **服务器发现**: 从目录结构自动发现 vLLM 服务器
- **健康检查**: 自动服务器健康监控
- **自定义 System Prompt**: 为所有请求设置系统提示
- **灵活的 API 支持**: 通过适配器支持不同的 API 格式（OpenAI、自定义等）
- **断点续跑**: 支持任务中断后从检查点恢复

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

- **JSONDataLoader**: 从 JSON 文件加载
- **CSVDataLoader**: 从 CSV 文件加载
- **PromptListLoader**: 从配置文件加载提示列表

### ResultSaver 选项

- **JSONResultSaver**: 保存到 JSON 文件（支持批量写入）
- **CSVResultSaver**: 保存到 CSV 文件
- **ConsoleResultSaver**: 打印到控制台（用于调试）

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

## 创建自定义组件

### 自定义 Loader

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

### 自定义 Saver

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

### 自定义 Adapter

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
│   ├── savers/           # 结果保存器实现
│   ├── adapters/         # 模型 API 适配器实现
│   ├── servers/          # 服务器管理和负载均衡
│   ├── utils/            # 工具模块（配置、日志、重试、进度、检查点）
│   ├── batch_runner.py   # 主协调器
│   └── cli.py            # 命令行接口
├── configs/              # 配置文件
│   └── examples/         # 示例配置
├── logs/                 # 日志文件
├── outputs/              # 输出文件
├── checkpoints/          # 检查点文件
└── requirements.txt
```

## 核心模块说明

### 1. DataLoader ([`src/loaders/`](src/loaders/))
- **base.py**: 抽象基类，定义数据加载接口
- **json_loader.py**: JSON 文件加载器
- **csv_loader.py**: CSV 文件加载器
- **prompt_list_loader.py**: 简单列表加载器

### 2. ResultSaver ([`src/savers/`](src/savers/))
- **base.py**: 抽象基类，定义结果保存接口
- **json_saver.py**: JSON 文件保存器（批量写入）
- **csv_saver.py**: CSV 文件保存器
- **console_saver.py**: 控制台输出

### 3. ModelAdapter ([`src/adapters/`](src/adapters/))
- **base.py**: 抽象基类，定义 API 适配器接口
- **openai_adapter.py**: OpenAI 兼容 API 适配���
- **simple_adapter.py**: 简单 prompt API 适配器

### 4. BatchRunner ([`src/batch_runner.py`](src/batch_runner.py))
主协调器，负责：
- 加载数据并生成 rollout
- 初始化服务器管理器和负载均衡器
- 并发处理请求
- 失败重试
- 断点续跑

### 5. 工具模块 ([`src/utils/`](src/utils/))
- **config.py**: YAML 配置加载和动态导入
- **logger.py**: 日志配置
- **retry.py**: 带指数退避的重试装饰器
- **progress.py**: 进度跟踪
- **checkpoint.py**: 断点续跑检查点管理

## 最佳实践

### 1. 大规模数据处理

启用断点续跑并使用较大的检查点间隔：

```yaml
runner:
  enable_checkpoint: true
  checkpoint_interval: 100  # 每 100 个请求保存一次
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

使用 JSON saver 和合理的并发数：

```yaml
saver:
  class: JSONResultSaver
  params:
    output_path: outputs/results_$(date).json
    batch_size: 100

runner:
  max_concurrency: 10
  enable_checkpoint: true
```

## 许可证

MIT License
