# JSONL Customization Guide

JSONL loaders are the main extension point for large rollout jobs. This guide focuses on the current customization hooks that matter for throughput, resume correctness, and metadata preservation.

## Recommended base classes

Use one of these and override only the hooks you need:

- `JSONLDataLoader`
- `DirectoryJSONLDataLoader`
- `MultimodalJSONLDataLoader`
- `MultimodalDirectoryJSONLDataLoader`

These classes already integrate:

- prompt extraction
- request ID generation
- `resume_key` generation
- `dispatch_cost` estimation
- multimodal message construction where applicable

## Main hooks

### `parse_line(line, line_num, source)`

Use this when one JSONL line needs custom parsing or fan-out.

Default behavior:

- `json.loads(line)`
- returns one dict

You may return:

- one dict
- a list of dicts
- `None` to skip the line

Example:

```python
import json

from src.loaders.jsonl_loader import JSONLDataLoader


class FissionLoader(JSONLDataLoader):
    def parse_line(self, line, line_num, source):
        obj = json.loads(line)
        prompts = obj.get("variations")
        if not prompts:
            return obj
        return [
            {
                **obj,
                "prompt": prompt,
                "id": f"{obj['id']}_{index}",
            }
            for index, prompt in enumerate(prompts)
        ]
```

When a line fans out into multiple items, the loader assigns distinct `resume_key` values by using different `item_idx` values.

### `should_skip_item(item)`

Use this to filter records early.

Example:

```python
class EnglishOnlyLoader(JSONLDataLoader):
    def should_skip_item(self, item):
        return item.get("language") != "en"
```

### `extract_request_id(item, default_id)`

Use this when IDs are stored differently or need a stable composite key.

If you do not override this and the configured `id_field` is missing, the loader falls back to a deterministic SHA-256 hash derived from prompt content and stable metadata.

### `extract_prompt(item)`

Use this when prompt text can come from multiple fields.

Example:

```python
class MultiFieldPromptLoader(JSONLDataLoader):
    def extract_prompt(self, item):
        for field in ("prompt", "question", "input", "text"):
            value = item.get(field)
            if value:
                return str(value)
        return None
```

### `extract_additional_data(item)`

Use this to control what metadata reaches the saver.

This is also a good place to preserve:

- source labels
- quality scores
- sharding metadata
- explicit token estimates

If your upstream dataset already knows token counts, pass them through here as `dispatch_cost`, `estimated_tokens`, `input_tokens`, or `prompt_tokens`.

## Resume behavior

JSONL loaders now produce:

`resume_key = (source_file, line_num, item_idx)`

That key is consumed by the bitmap resume backend. If you override parsing hooks, keep these rules in mind:

- one logical request should map to one stable output item
- the number and order of fissioned items should stay deterministic
- changes to fan-out logic can invalidate old resume artifacts

## Cost-aware routing

Each `LoadResult` may also carry `dispatch_cost`.

Default behavior:

- prefer explicit cost metadata from `additional_data`
- otherwise estimate cost from prompt length

If your dataset has better estimates, override or enrich the loader so the router can separate short and long requests more effectively.

## Customizing `dispatch_cost`

For non-JSONL loaders, override `estimate_dispatch_cost()`.  
For JSONL loaders, the easiest approach is usually to expose a numeric field in the record.

Example:

```python
class CostAwareLoader(JSONLDataLoader):
    def extract_additional_data(self, item):
        data = super().extract_additional_data(item)
        if "estimated_tokens" in item:
            data["estimated_tokens"] = item["estimated_tokens"]
        return data
```

## Saver-side JSONL customization

Built-in JSONL savers support:

- `output_projection`
- `output_fields`
- `include_timestamp`

If you need a custom line format, subclass `JSONLResultSaver` or `DirectoryJSONLResultSaver` and override the output formatting hook from `JSONLSaverMixin`.

## Example: custom directory loader

```python
from src.loaders.directory_jsonl_loader import DirectoryJSONLDataLoader


class ConversationLoader(DirectoryJSONLDataLoader):
    def extract_prompt(self, item):
        turns = item.get("conversation", [])
        if not turns:
            return None
        return "\n".join(
            f"{turn['role']}: {turn['content']}"
            for turn in turns
            if "content" in turn
        )

    def extract_additional_data(self, item):
        data = super().extract_additional_data(item)
        data["domain"] = item.get("domain")
        return data
```

## Best practices

- Prefer overriding a built-in loader instead of rewriting `load()` from scratch.
- Keep `request_id` generation stable across reruns.
- Preserve upstream token estimates when you have them.
- Be careful when changing sample fission logic if old bitmap resume files exist.
- Test custom loaders with both normal data and resume enabled.
