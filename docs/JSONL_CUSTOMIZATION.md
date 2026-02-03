# JSONL Customization Guide

This guide explains how to customize JSONL loaders and savers by overriding methods from the mixin classes.

## Overview

The JSONL loaders and savers now use a **Template Method pattern** with mixin classes that provide customizable processing hooks. Instead of rewriting entire classes, you can override specific methods to customize:

- **For Loaders**: How each JSONL line is parsed and converted to a `LoadResult`
- **For Savers**: How each `SaveResult` is formatted and serialized to JSONL

## Loader Customization

### Available Mixin Methods

All JSONL loaders inherit from `JSONLLoaderMixin`, which provides these overridable methods:

#### 1. `parse_line(self, line: str, line_num: int, source: str) -> Optional[Dict[str, Any]]`

Parse a single JSONL line into a dictionary. Override this to support custom JSONL formats.

**Default behavior**: Uses `json.loads(line)` to parse standard JSON objects.

**Example: Handle list-format JSONL**

```python
from src.loaders.jsonl_loader import JSONLDataLoader
import json

class ListFormatLoader(JSONLDataLoader):
    """Handle JSONL files where each line is a list: [{"prompt": "hello"}]"""

    def parse_line(self, line: str, line_num: int, source: str) -> Optional[Dict[str, Any]]:
        obj = json.loads(line)

        # If line is a list, convert to dict
        if isinstance(obj, list):
            return {
                "items": obj,
                "id": f"{source}:{line_num}",
                "prompt": " ".join(item.get("text", "") for item in obj)
            }

        return obj
```

**Example: Handle nested JSON structure**

```python
class NestedJSONLLoader(JSONLDataLoader):
    """Handle nested structures like {"data": {"prompt": "hello", "id": "1"}}"""

    def parse_line(self, line: str, line_num: int, source: str) -> Optional[Dict[str, Any]]:
        obj = json.loads(line)

        # Flatten nested structure
        if "data" in obj and isinstance(obj["data"], dict):
            return obj["data"]

        return obj
```

#### 2. `should_skip_item(self, item: Dict[str, Any]) -> bool`

Determine if an item should be skipped. Override this to implement custom filtering logic.

**Default behavior**: Never skips items (returns `False`).

**Example: Filter by language**

```python
class EnglishOnlyLoader(JSONLDataLoader):
    """Only load items where language is 'en'"""

    def should_skip_item(self, item: Dict[str, Any]) -> bool:
        # Skip non-English items
        return item.get("language") != "en"
```

**Example: Filter by quality score**

```python
class HighQualityLoader(JSONLDataLoader):
    """Only load items with quality score >= 0.8"""

    def should_skip_item(self, item: Dict[str, Any]) -> bool:
        quality = item.get("quality_score", 0.0)
        return quality < 0.8
```

#### 3. `extract_request_id(self, item: Dict[str, Any], default_id: str) -> str`

Extract the request ID from a parsed item. Override this to customize ID extraction.

**Default behavior**: Uses configured `id_field` (default: `"id"`).

**Example: Use composite key**

```python
class CompositeKeyLoader(JSONLDataLoader):
    """Create ID from doc_id and line_num"""

    def extract_request_id(self, item: Dict[str, Any], default_id: str) -> str:
        doc_id = item.get("doc_id", "unknown")
        line_num = item.get("line_num", "0")
        return f"{doc_id}_{line_num}"
```

**Example: Use hash-based ID**

```python
import hashlib

class HashIDLoader(JSONLDataLoader):
    """Generate ID from content hash"""

    def extract_request_id(self, item: Dict[str, Any], default_id: str) -> str:
        content = item.get("prompt", "")
        return hashlib.md5(content.encode()).hexdigest()[:12]
```

#### 4. `extract_prompt(self, item: Dict[str, Any]) -> Optional[str]`

Extract the prompt text from a parsed item. Override this to customize prompt extraction.

**Default behavior**: Uses configured `prompt_field` (default: `"prompt"`).

**Example: Try multiple fields in order**

```python
class MultiFieldPromptLoader(JSONLDataLoader):
    """Try multiple fields for prompt"""

    def extract_prompt(self, item: Dict[str, Any]) -> Optional[str]:
        for field in ['prompt', 'question', 'text', 'input', 'instruction']:
            if field in item:
                return str(item[field])
        return None
```

**Example: Combine multiple fields**

```python
class CombinedPromptLoader(JSONLDataLoader):
    """Combine instruction and input fields"""

    def extract_prompt(self, item: Dict[str, Any]) -> Optional[str]:
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")

        if instruction and input_text:
            return f"{instruction}\n\nInput: {input_text}"
        return instruction or input_text or None
```

#### 5. `extract_additional_data(self, item: Dict[str, Any]) -> Dict[str, Any]`

Extract additional data to pass through to the saver. Override this to customize what metadata is preserved.

**Default behavior**: Excludes `prompt_field` and `id_field` (and image fields for multimodal loaders).

**Example: Preserve specific fields only**

```python
class SelectiveDataLoader(JSONLDataLoader):
    """Only preserve category and tags fields"""

    def extract_additional_data(self, item: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        if "category" in item:
            result["category"] = item["category"]
        if "tags" in item:
            result["tags"] = item["tags"]
        return result
```

**Example: Transform additional data**

```python
class TransformingLoader(JSONLDataLoader):
    """Transform tags from list to comma-separated string"""

    def extract_additional_data(self, item: Dict[str, Any]) -> Dict[str, Any]:
        data = super().extract_additional_data(item)

        # Transform tags list to string
        if "tags" in data and isinstance(data["tags"], list):
            data["tags"] = ",".join(data["tags"])

        return data
```

### Complete Loader Example

Here's a complete example combining multiple customizations:

```python
from src.loaders.jsonl_loader import JSONLDataLoader
import json
from typing import Dict, Any, Optional

class CustomJSONLLoader(JSONLDataLoader):
    """
    Custom loader that:
    1. Handles list-format lines
    2. Filters out short prompts
    3. Uses composite IDs
    4. Combines multiple fields for prompts
    """

    def parse_line(self, line: str, line_num: int, source: str) -> Optional[Dict[str, Any]]:
        obj = json.loads(line)

        # Handle list format: [{"text": "hello"}]
        if isinstance(obj, list):
            if len(obj) == 1 and isinstance(obj[0], dict):
                return obj[0]
            return {"items": obj, "id": str(line_num)}

        return obj

    def should_skip_item(self, item: Dict[str, Any]) -> bool:
        # Filter out items with very short prompts
        prompt = self.extract_prompt(item)
        return prompt is not None and len(prompt) < 10

    def extract_request_id(self, item: Dict[str, Any], default_id: str) -> str:
        # Use composite key: category_docId
        category = item.get("category", "unknown")
        doc_id = item.get("id", default_id)
        return f"{category}_{doc_id}"

    def extract_prompt(self, item: Dict[str, Any]) -> Optional[str]:
        # Try multiple fields
        for field in ['prompt', 'question', 'text']:
            if field in item:
                return str(item[field])
        return None

    def extract_additional_data(self, item: Dict[str, Any]) -> Dict[str, Any]:
        # Only preserve metadata fields
        metadata_fields = ['category', 'tags', 'language', 'quality_score']
        return {
            k: v for k, v in item.items()
            if k in metadata_fields and v is not None
        }
```

## Saver Customization

### Available Mixin Methods

All JSONL savers inherit from `JSONLSaverMixin`, which provides these overridable methods:

#### 1. `format_result(self, result: SaveResult) -> Dict[str, Any]`

Format a `SaveResult` into a dictionary for JSONL output. Override this to customize the output structure.

**Default behavior**: Creates a dictionary with `request_id`, `model_output`, `additional_data`, and `timestamp`.

**Example: Minimal output format**

```python
from src.savers.jsonl_saver import JSONLResultSaver

class MinimalSaver(JSONLResultSaver):
    """Only save ID and generated text"""

    def format_result(self, result) -> Dict[str, Any]:
        content = result.model_output['choices'][0]['message']['content']
        return {
            "id": result.request_id,
            "response": content
        }
```

**Example: Include token usage**

```python
class TokenAwareSaver(JSONLResultSaver):
    """Include token counts in output"""

    def format_result(self, result) -> Dict[str, Any]:
        content = result.model_output['choices'][0]['message']['content']
        usage = result.model_output.get('usage', {})

        return {
            "id": result.request_id,
            "response": content,
            "prompt_tokens": usage.get('prompt_tokens', 0),
            "completion_tokens": usage.get('completion_tokens', 0),
            "total_tokens": usage.get('total_tokens', 0)
        }
```

**Example: Flatten nested structure**

```python
class FlatSaver(JSONLResultSaver):
    """Flatten model_output and additional_data"""

    def format_result(self, result) -> Dict[str, Any]:
        output = {
            "request_id": result.request_id,
        }

        # Extract content
        if 'choices' in result.model_output and len(result.model_output['choices']) > 0:
            output['response'] = result.model_output['choices'][0]['message']['content']

        # Flatten additional_data
        if result.additional_data:
            output.update(result.additional_data)

        # Add error if present
        if result.error:
            output['error'] = result.error

        return output
```

#### 2. `serialize_output(self, output_data: Dict[str, Any]) -> str`

Serialize the formatted dictionary to a JSON string. Override this to customize serialization.

**Default behavior**: Uses `json.dumps(output_data, ensure_ascii=False)`.

**Example: Sort keys for consistent output**

```python
class SortedSaver(JSONLResultSaver):
    """Sort keys for deterministic output"""

    def serialize_output(self, output_data: Dict[str, Any]) -> str:
        return json.dumps(output_data, ensure_ascii=False, sort_keys=True)
```

**Example: Custom datetime format**

```python
from datetime import datetime

class CustomDatetimeSaver(JSONLResultSaver):
    """Use custom datetime format"""

    def format_result(self, result) -> Dict[str, Any]:
        content = result.model_output['choices'][0]['message']['content']
        return {
            "id": result.request_id,
            "response": content,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
```

### Complete Saver Example

Here's a complete example combining multiple customizations:

```python
from src.savers.directory_jsonl_saver import DirectoryJSONLResultSaver
from typing import Dict, Any

class CustomDirectorySaver(DirectoryJSONLResultSaver):
    """
    Custom saver that:
    1. Extracts only the generated text
    2. Includes token usage and metadata
    3. Formats timestamps nicely
    4. Preserves source information
    """

    def format_result(self, result) -> Dict[str, Any]:
        # Extract generated content
        content = ""
        if 'choices' in result.model_output and len(result.model_output['choices']) > 0:
            content = result.model_output['choices'][0]['message']['content']

        # Extract token usage
        usage = result.model_output.get('usage', {})

        # Build output
        output = {
            "id": result.request_id,
            "text": content,
            "prompt_tokens": usage.get('prompt_tokens', 0),
            "completion_tokens": usage.get('completion_tokens', 0),
            "total_tokens": usage.get('total_tokens', 0),
            "timestamp": result.model_output.get('created', 0)
        }

        # Include additional metadata
        if result.additional_data:
            # Selectively include metadata
            for key in ['category', 'tags', 'language', '_source_file']:
                if key in result.additional_data:
                    output[key] = result.additional_data[key]

        # Include error if present
        if result.error:
            output['error'] = result.error

        return output

    def serialize_output(self, output_data: Dict[str, Any]) -> str:
        # Serialize with sorted keys for consistent ordering
        return json.dumps(output_data, ensure_ascii=False, sort_keys=True)
```

## Directory-Based Loaders

For `DirectoryJSONLDataLoader` and `MultimodalDirectoryJSONLDataLoader`, you can override the same methods. **Both streaming and non-streaming modes will automatically use your customizations.**

### Streaming vs Non-Streaming Modes

Directory loaders support two modes of operation:

1. **Streaming Mode** (`streaming: true` in config, default):
   - Processes files on-demand using `parse_line()` method
   - Minimal memory footprint
   - Best for large datasets

2. **Non-Streaming Mode** (`streaming: false` in config):
   - Pre-loads all data into memory during initialization
   - Also uses `parse_line()` method for consistency
   - Easier for debugging and development

**Your custom `parse_line()` and other methods work in both modes!**

### Example: Custom directory loader with source tracking

```python
from src.loaders.directory_jsonl_loader import DirectoryJSONLDataLoader
from typing import Dict, Any

class SourceAwareDirectoryLoader(DirectoryJSONLDataLoader):
    """Track source file in request_id"""

    def extract_request_id(self, item: Dict[str, Any], default_id: str) -> str:
        # Include source file in ID
        source_file = item.get('_source_file', 'unknown')
        item_id = item.get('id', default_id)
        # Extract just the filename
        filename = source_file.split('/')[-1] if '/' in source_file else source_file
        return f"{filename}:{item_id}"

    def extract_additional_data(self, item: Dict[str, Any]) -> Dict[str, Any]:
        # Include source information in additional_data
        data = super().extract_additional_data(item)
        data['_source_file'] = item.get('_source_file', '')
        data['_source_dir'] = item.get('_source_dir', '')
        return data
```

### Example: Handle list-format in directory files

```python
import json
from typing import Dict, Any, Optional

class ListFormatDirectoryLoader(DirectoryJSONLDataLoader):
    """
    Handle directory files where each line is a list.
    Works in both streaming and non-streaming modes.
    """

    def parse_line(self, line: str, line_num: int, source: str) -> Optional[Dict[str, Any]]:
        obj = json.loads(line)

        # Convert list to dict
        if isinstance(obj, list):
            if len(obj) == 1 and isinstance(obj[0], dict):
                return obj[0]
            # Multiple items - combine prompts
            return {
                "items": obj,
                "id": f"{source}:{line_num}",
                "prompt": " ".join(
                    item.get("text", item.get("prompt", "")) for item in obj
                )
            }

        return obj
```

### Example: Multi-field prompt extraction for directories

```python
class MultiFieldDirectoryLoader(DirectoryJSONLDataLoader):
    """
    Try multiple fields for prompt extraction.
    Useful for directories with mixed data formats.
    """

    def extract_prompt(self, item: Dict[str, Any]) -> Optional[str]:
        # Try fields in order of preference
        for field in ['prompt', 'question', 'text', 'input', 'instruction']:
            if field in item:
                prompt = item[field]
                # Handle list-type prompts
                if isinstance(prompt, list):
                    return " ".join(str(p) for p in prompt)
                return str(prompt)
        return None
```

### Example: Handle conversation format in directory files

```python
class ConversationDirectoryLoader(DirectoryJSONLDataLoader):
    """
    Handle conversation format with messages array.

    Example format:
    {
        "id": "conv1",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"}
        ]
    }
    """

    def parse_line(self, line: str, line_num: int, source: str) -> Optional[Dict[str, Any]]:
        obj = json.loads(line)

        # Handle conversation format
        if "messages" in obj and isinstance(obj["messages"], list):
            # Find the last user message
            for msg in reversed(obj["messages"]):
                if msg.get("role") == "user":
                    return {
                        "id": obj.get("id", f"{source}:{line_num}"),
                        "prompt": msg.get("content"),
                        "conversation": obj["messages"]
                    }

        return obj
```

## Multimodal Loaders

For `MultimodalJSONLDataLoader` and `MultimodalDirectoryJSONLDataLoader`, you can also override:

### `extract_images(self, item: Dict[str, Any]) -> Optional[List[str]]`

Customize how images are extracted from each item.

**Example: Support nested media structure**

```python
from src.loaders.jsonl_loader import MultimodalJSONLDataLoader
from typing import Dict, Any, Optional, List

class CustomMediaLoader(MultimodalJSONLDataLoader):
    """Support nested media: {"media": [{"type": "image", "path": "img.jpg"}]}"""

    def extract_images(self, item: Dict[str, Any]) -> Optional[List[str]]:
        # Try custom media structure first
        if 'media' in item and isinstance(item['media'], list):
            images = [
                m['path'] for m in item['media']
                if m.get('type') == 'image' and 'path' in m
            ]
            return images if images else None

        # Fall back to default behavior
        return super().extract_images(item)
```

## Configuration Examples

### Using custom loader in YAML config

```yaml
loader:
  class: my_module.CustomJSONLLoader  # Your custom class
  params:
    file_path: data/input.jsonl
    prompt_field: instruction
    id_field: doc_id

saver:
  class: src.savers.jsonl_saver.JSONLResultSaver  # Or custom saver
  params:
    output_path: output/results.jsonl
```

### Using custom saver in YAML config

```yaml
saver:
  class: my_module.MinimalSaver  # Your custom class
  params:
    output_path: output/results.jsonl
```

## Best Practices

1. **Always call `super()`** when extending behavior (e.g., in `extract_additional_data`)

2. **Return `None`** from `extract_prompt()` to skip items (same as returning `True` from `should_skip_item`)

3. **Preserve source metadata** in directory loaders by extending `extract_additional_data()`

4. **Keep overrides focused** - each method should do one thing well

5. **Handle edge cases** - check for missing keys, None values, and unexpected types

6. **Use type hints** - the base classes include proper type annotations

7. **Test with sample data** - verify your customizations work with actual data before processing large datasets
