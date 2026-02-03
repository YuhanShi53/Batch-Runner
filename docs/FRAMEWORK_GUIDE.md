# vLLM Runner Framework Guide

## Overview

This guide explains the refactored vLLM Runner framework, which provides a highly extensible and efficient system for batch inference with custom data loaders and result savers.

## Table of Contents

1. [Architecture](#architecture)
2. [Streaming Support](#streaming-support)
3. [Built-in Loaders and Savers](#built-in-loaders-and-savers)
4. [Customization Hooks](#customization-hooks)
5. [Creating Custom Components](#creating-custom-components)
6. [Registration System](#registration-system)
7. [Examples](#examples)

---

## Architecture

The framework follows a **plugin architecture** with these key components:

### Core Components

- **DataLoader**: Abstract base class for all data loaders
- **ResultSaver**: Abstract base class for all result savers
- **ModelAdapter**: Adapters for different API formats (OpenAI, Simple, etc.)
- **BatchRunner**: Main orchestrator for concurrent processing
- **VLLMServerManager**: Server discovery and health checking

### Mixin System

The framework uses a **mixin pattern** to provide reusable functionality:

#### Loader Mixins

- **`StreamingLoaderMixin`**: Streaming data loading with on-demand processing
- **`MessagesBuilderMixin`**: Flexible message construction
- **`PromptExtractorMixin`**: Customizable prompt extraction
- **`MultimodalInputMixin`**: Multimodal (text + image) input handling
- **`JSONLLoaderMixin`**: JSONL-specific parsing logic

#### Saver Mixins

- **`StreamingSaverMixin`**: Streaming result writing with immediate flush
- **`OutputFormatterMixin`**: Flexible output formatting
- **`MultimodalOutputMixin`**: Multimodal output handling
- **`JSONLSaverMixin`**: JSONL-specific formatting logic

---

## Streaming Support

### Why Streaming?

Streaming mode provides several benefits:

1. **Constant Memory Usage**: Processes data on-demand regardless of dataset size
2. **Immediate Processing**: Starts processing as soon as first data is available
3. **Better Resource Utilization**: I/O and computation happen concurrently
4. **Fault Tolerance**: Data is saved immediately, reducing loss on failures

### Enabling Streaming

Streaming is enabled by default for most loaders/savers:

```yaml
loader:
  class: JSONLDataLoader
  params:
    file_path: data/input.jsonl
    streaming: true  # Enable streaming (default)
```

### Streaming vs Batch Mode

| Feature | Streaming Mode | Batch Mode |
|---------|---------------|------------|
| Memory Usage | Constant (queue size) | O(dataset size) |
| Startup Time | Immediate | Wait for full load |
| Progress | Shows completed count | Shows percentage |
| Fault Tolerance | High (data saved as processed) | Lower (data in memory) |
| Use Case | Large datasets (>10K items) | Small datasets, debugging |

---

## Built-in Loaders and Savers

### Built-in Loaders

| Loader | Description | Streaming | Multimodal |
|--------|-------------|-----------|------------|
| `JSONDataLoader` | Load from JSON array files | No | Yes |
| `JSONLDataLoader` | Load from JSONL files | Yes | Yes |
| `CSVDataLoader` | Load from CSV files | No | No |
| `PromptListLoader` | Load from config list | No | No |
| `DirectoryJSONLDataLoader` | Load from directory of JSONL files | Yes | Yes |

### Built-in Savers

| Saver | Description | Streaming | Directory Support |
|-------|-------------|-----------|-------------------|
| `JSONResultSaver` | Save to JSON array | Batch | No |
| `JSONLResultSaver` | Save to JSONL | Yes | No |
| `CSVResultSaver` | Save to CSV | Yes | No |
| `ConsoleResultSaver` | Print to console | Yes | No |
| `DirectoryJSONLResultSaver` | Save to directory structure | Yes | Yes |

### Usage Examples

#### JSONL Loader (Streaming)

```yaml
loader:
  class: JSONLDataLoader
  params:
    file_path: data/input.jsonl
    prompt_field: prompt  # Default
    id_field: id          # Default
    streaming: true       # Enable streaming
```

#### Directory JSONL Loader (Streaming)

```yaml
loader:
  class: DirectoryJSONLDataLoader
  params:
    input_dir: data/conversations
    file_pattern: conv.jsonl  # Default
    recursive: true           # Default
    streaming: true           # Enable streaming
```

#### JSONL Saver (Streaming)

```yaml
saver:
  class: JSONLResultSaver
  params:
    output_path: results/output.jsonl
    append: true           # Default
    streaming: true        # Default
    immediate_flush: true  # Default
```

---

## Customization Hooks

The framework provides clear, well-defined hooks for customization at each step of data processing.

### Loader Customization Hooks

#### 1. Prompt Extraction

Override `extract_prompt()` to customize how prompts are extracted from source data:

```python
class MyLoader(JSONLLoaderMixin, DataLoader):
    def extract_prompt(self, item):
        # Try multiple fields in priority order
        for field in ['prompt', 'question', 'text', 'input']:
            if field in item:
                return str(item[field])
        return None
```

#### 2. Message Construction

Override `build_messages()` to customize how prompts are transformed into API messages:

```python
class MyLoader(MessagesBuilderMixin, DataLoader):
    def build_messages(self, prompt, additional_data=None):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

        # Add conversation history
        if additional_data and 'history' in additional_data:
            messages.extend(additional_data['history'])

        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        return messages
```

#### 3. Image Extraction (Multimodal)

Override `extract_images()` to customize image extraction:

```python
class MyLoader(MultimodalInputMixin, MultimodalDataLoader):
    def extract_images(self, item):
        # Support nested media structure
        if 'media' in item:
            return [m['path'] for m in item['media'] if m['type'] == 'image']
        return super().extract_images(item)
```

#### 4. Line Parsing (JSONL)

Override `parse_line()` to handle custom JSONL formats:

```python
class MyLoader(JSONLLoaderMixin, DataLoader):
    def parse_line(self, line, line_num, source):
        obj = json.loads(line)
        # Handle list-format JSONL: [{"key": "value"}]
        if isinstance(obj, list):
            return {"items": obj, "id": str(line_num)}
        return obj
```

#### 5. Item Filtering

Override `should_skip_item()` to filter items:

```python
class MyLoader(JSONLLoaderMixin, DataLoader):
    def should_skip_item(self, item):
        # Skip items without required fields
        return 'prompt' not in item or 'id' not in item
```

### Saver Customization Hooks

#### 1. Output Formatting

Override `format_output()` to customize the output structure:

```python
class MySaver(OutputFormatterMixin, ResultSaver):
    def format_output(self, result):
        content = result.model_output['choices'][0]['message']['content']
        return {
            "id": result.request_id,
            "response": content,
            "tokens": result.model_output.get('usage', {}).get('total_tokens', 0)
        }
```

#### 2. JSON Serialization (JSONL)

Override `serialize_output()` to customize JSON serialization:

```python
class MySaver(JSONLSaverMixin, ResultSaver):
    def serialize_output(self, output_data):
        return json.dumps(output_data, ensure_ascii=False, indent=None)
```

#### 3. Result Filtering

Override `should_save_result()` to filter results:

```python
class MySaver(StreamingSaverMixin, ResultSaver):
    def should_save_result(self, result):
        # Only save successful results
        return result.error is None
```

---

## Creating Custom Components

### Option 1: In-Project Custom Components

Create your custom loader/saver in the appropriate directory:

```python
# src/loaders/my_custom_loader.py
from src.loaders.base import DataLoader, LoadResult
from src.loaders.jsonl_mixin import JSONLLoaderMixin

class MyCustomLoader(JSONLLoaderMixin, DataLoader):
    def _initialize(self):
        self.file_path = Path(self.config['file_path'])

    def load(self):
        # Your loading logic
        pass
```

Then use it in your config:

```yaml
loader:
  class: MyCustomLoader
  params:
    file_path: data/input.jsonl
```

### Option 2: Out-of-Project Custom Components (Recommended)

Create your custom components anywhere and use the registration system:

```python
# custom_components.py
from src.loaders.base import DataLoader, LoadResult
from src.utils.registry import register_loader

@register_loader
class MyCustomLoader(DataLoader):
    def _initialize(self):
        # Your initialization
        pass

    def load(self):
        # Your loading logic
        yield LoadResult(...)
```

Then either:

1. **Auto-register via config**:
```yaml
custom_modules:
  - custom_components.py

loader:
  class: MyCustomLoader
  params:
    # ...
```

2. **Import before running**:
```python
import custom_components  # Registers the loaders/savers
from src.cli import main
main()
```

---

## Registration System

The registration system allows you to use custom loaders/savers without modifying the project source code.

### Using Decorators

```python
from src.utils.registry import register_loader, register_saver

@register_loader
class MyCustomLoader(DataLoader):
    # Implementation
    pass

@register_saver
class MyCustomSaver(ResultSaver):
    # Implementation
    pass
```

### Registration Functions

```python
from src.utils.registry import register_loader_class, register_saver_class

class MyCustomLoader(DataLoader):
    pass

# Register with a custom name
register_loader_class('CustomLoader', MyCustomLoader)
```

### Listing Registered Components

```python
from src.utils.registry import list_registered_loaders, list_registered_savers

print("Loaders:", list_registered_loaders())
print("Savers:", list_registered_savers())
```

---

## Examples

### Example 1: Custom JSONL Loader with Multi-Field Prompt Extraction

```python
@register_loader
class MultiFieldJSONLLoader(JSONLLoaderMixin, DataLoader):
    def _initialize(self):
        self.file_path = Path(self.config['file_path'])
        self.prompt_fields = self.config.get('prompt_fields', ['prompt', 'question', 'text'])

    def extract_prompt(self, item):
        for field in self.prompt_fields:
            if field in item:
                return str(item[field])
        return None

    def load(self):
        with open(self.file_path) as f:
            for line_num, line in enumerate(f, 1):
                result = self.process_line_to_load_result(
                    line=line.strip(),
                    line_num=line_num,
                    source=str(self.file_path),
                    default_id=f"req_{line_num}"
                )
                if result:
                    yield result
```

### Example 2: Custom Saver with Simplified Output

```python
@register_saver
class SimpleJSONLSaver(JSONLSaverMixin, ResultSaver):
    def _initialize(self):
        self.output_path = Path(self.config['output_path'])
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.output_path, 'a')
        self._lock = threading.Lock()

    def format_result(self, result):
        content = result.model_output['choices'][0]['message']['content']
        return {
            'id': result.request_id,
            'response': content
        }

    def save(self, result):
        line = self.process_result_to_line(result)
        with self._lock:
            self.file.write(line + '\n')
            self.file.flush()

    def cleanup(self):
        with self._lock:
            self.file.close()
```

### Example 3: Chat Messages with History

```python
@register_loader
class ChatLoader(MessagesBuilderMixin, DataLoader):
    def build_messages(self, prompt, additional_data=None):
        messages = []

        # Add system prompt
        if self.config.get('system_prompt'):
            messages.append({
                "role": "system",
                "content": self.config['system_prompt']
            })

        # Add conversation history
        if additional_data and 'history' in additional_data:
            messages.extend(additional_data['history'])

        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        return messages

    def _initialize(self):
        self.file_path = Path(self.config['file_path'])
        with open(self.file_path) as f:
            self.conversations = json.load(f)

    def load(self):
        for idx, conv in enumerate(self.conversations, 1):
            messages = self.build_messages(
                conv['prompt'],
                {'history': conv.get('history', [])}
            )
            yield LoadResult(
                messages=messages,
                request_id=conv.get('id', f"conv_{idx}")
            )
```

### Example 4: Multimodal Loader with Custom Image Handling

```python
@register_loader
class CustomMultimodalLoader(MultimodalInputMixin, MultimodalDataLoader):
    def extract_images(self, item):
        # Support multiple image field formats
        if 'image' in item:
            return [item['image']] if isinstance(item['image'], str) else item['image']
        if 'images' in item:
            return item['images'] if isinstance(item['images'], list) else [item['images']]
        if 'media' in item:
            return [m['path'] for m in item['media'] if m.get('type') == 'image']
        return None

    def _initialize(self):
        super()._initialize()
        self.file_path = Path(self.config['file_path'])
        with open(self.file_path) as f:
            self.data = json.load(f)

    def load(self):
        for idx, item in enumerate(self.data, 1):
            prompt = item.get('prompt', '')
            images = self.extract_images(item)
            yield self._create_multimodal_result(
                text=prompt,
                images=images,
                request_id=item.get('id', f"mm_{idx}")
            )
```

---

## Complete Example Config

```yaml
# config.yaml

# Auto-register custom components
custom_modules:
  - custom_components.py

loader:
  class: MultiFieldJSONLLoader
  params:
    file_path: data/conversations.jsonl
    prompt_fields:
      - prompt
      - question
      - text
    streaming: true

saver:
  class: SimpleJSONLSaver
  params:
    output_path: results/output.jsonl

runner:
  max_concurrency: 20
  model_name: "meta-llama/Llama-3-8b"
  servers_dir: ./servers
  load_balancing_strategy: round_robin
  temperature: 0.7
  max_tokens: 1000
```

---

## Migration Guide

### From Old Code to New Framework

If you have existing custom loaders/savers, here's how to migrate:

#### Old Loader (Batch Mode)

```python
# Before
class MyLoader(DataLoader):
    def _initialize(self):
        with open(self.config['file']) as f:
            self.data = json.load(f)

    def load(self):
        for item in self.data:
            yield LoadResult(
                messages=[{"role": "user", "content": item['prompt']}],
                request_id=item['id']
            )
```

#### New Loader (Streaming Mode)

```python
# After
class MyLoader(StreamingLoaderMixin, DataLoader):
    def _initialize(self):
        self._initialize_streaming()
        self.file_path = Path(self.config['file_path'])

    def _discover_sources(self):
        return [self.file_path]

    def _process_source(self, source):
        with open(source) as f:
            data = json.load(f)
        for item in data:
            yield LoadResult(
                messages=[{"role": "user", "content": item['prompt']}],
                request_id=item['id']
            )
```

---

## Best Practices

1. **Use Streaming for Large Datasets**: Enable streaming mode for datasets with >10K items
2. **Leverage Mixins**: Use provided mixins instead of rewriting common functionality
3. **Override Specific Hooks**: Only override the hooks you need, not entire methods
4. **Thread Safety**: Ensure your custom components are thread-safe
5. **Resource Cleanup**: Always implement `cleanup()` to release resources
6. **Error Handling**: Handle errors gracefully and log appropriately

---

## API Reference

See the individual module documentation for detailed API references:

- [src/loaders/base.py](../src/loaders/base.py) - DataLoader base class
- [src/savers/base.py](../src/savers/base.py) - ResultSaver base class
- [src/loaders/streaming_mixin.py](../src/loaders/streaming_mixin.py) - Streaming mixins
- [src/savers/streaming_mixin.py](../src/savers/streaming_mixin.py) - Streaming mixins
- [src/loaders/jsonl_mixin.py](../src/loaders/jsonl_mixin.py) - JSONL mixins
- [src/savers/jsonl_mixin.py](../src/savers/jsonl_mixin.py) - JSONL mixins
- [src/utils/registry.py](../src/utils/registry.py) - Registration system
- [src/utils/config.py](../src/utils/config.py) - Configuration loading
