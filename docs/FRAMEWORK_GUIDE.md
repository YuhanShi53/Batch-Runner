# Framework Guide

This guide documents the current architecture and extension points of `vllm_runner`.

## Core components

- `DataLoader`: yields `LoadResult`
- `ResultSaver`: persists `SaveResult`
- `ModelAdapter`: builds requests and parses responses
- `BatchRunner`: orchestrates loading, routing, HTTP, retries, and saving
- `VLLMServerManager`: discovers servers and maintains health state
- `LoadBalancer`: chooses the next server

## Main data structures

### `LoadResult`

`LoadResult` now carries:

- `messages`
- `request_id`
- `additional_data`
- `resume_key`
- `dispatch_cost`

`resume_key` is used by bitmap resume backends.  
`dispatch_cost` is used by load balancing and admission heuristics.

### `SaveResult`

`SaveResult` carries:

- `request_id`
- `model_output`
- `additional_data`
- `error`
- `resume_key`

## Execution flow

1. `load_config()` reads YAML and loads `custom_modules`.
2. Loader, saver, and adapter classes are resolved.
3. `BatchRunner` creates the server manager, load balancer, progress tracker, and shared HTTP client.
4. The producer reads `LoadResult` objects from the loader.
5. The scheduler keeps a bounded number of in-flight async requests.
6. The adapter builds the request payload for the selected server.
7. Completed responses enter the writer queue as `SaveResult`.
8. Writer workers batch-save results and update resume/progress state.

## Built-in loaders

- `JSONDataLoader`
- `JSONLDataLoader`
- `CSVDataLoader`
- `PromptListLoader`
- `DirectoryJSONLDataLoader`
- `MultimodalJSONDataLoader`
- `MultimodalJSONLDataLoader`
- `MultimodalDirectoryJSONLDataLoader`

## Built-in savers

- `JSONResultSaver`
- `JSONLResultSaver`
- `CSVResultSaver`
- `ConsoleResultSaver`
- `DirectoryJSONLResultSaver`

## Built-in adapters

- `OpenAIAdapter`
- `SimpleAdapter`

## Configuration layout

Required top-level sections:

- `loader`
- `saver`
- `runner`

Optional:

- `logging`
- `custom_modules`

## Important runner settings

### Concurrency and HTTP

- `max_concurrency`
- `request_timeout`
- `http_max_connections`
- `http_max_keepalive_connections`
- `http2`

### Routing and retries

- `load_balancing_strategy`
- `selection_sample_size`
- `max_active_requests`
- `max_inflight_cost`
- `max_retries`
- `retry_delay`
- `allow_unhealthy_fallback`

### Pipeline behavior

- `streaming`
- `producer_prefetch`
- `writer_queue_size`
- `writer_batch_size`
- `writer_flush_interval_ms`
- `writer_workers`
- `progress_report_interval`

### Resume and multimodal

- `resume`
- `resume_backend`
- `image_encode_workers`

## Server discovery

The current implementation discovers server marker files under `servers_dir`. Names must match:

`server_<ip>_<port>`

Example:

```text
servers/
├── server_10.0.0.1_8000
├── server_10.0.0.2_8000
└── server_10.0.0.3_8000
```

## Extending loaders

Subclass `DataLoader` directly for brand-new sources, or extend a built-in JSONL loader for structured text data.

### Minimal custom loader

```python
from pathlib import Path

from src.loaders.base import DataLoader, LoadResult


class TextFileLoader(DataLoader):
    def _initialize(self):
        self.path = Path(self.config["file_path"])

    def load(self):
        with self.path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle, start=1):
                yield LoadResult(
                    messages=[{"role": "user", "content": line.strip()}],
                    request_id=f"line_{index}",
                )
```

### JSONL customization hooks

`JSONLLoaderMixin` provides the main override points:

- `parse_line()`
- `should_skip_item()`
- `extract_request_id()`
- `extract_prompt()`
- `extract_additional_data()`

See [JSONL_CUSTOMIZATION.md](/Users/yuhan/code/vllm_runner/docs/JSONL_CUSTOMIZATION.md).

## Extending savers

Subclass `ResultSaver` when you need a custom storage backend.

### Minimal custom saver

```python
import json
from pathlib import Path

from src.savers.base import ResultSaver, SaveResult


class MySaver(ResultSaver):
    def _initialize(self):
        self.path = Path(self.config["output_path"])
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, result: SaveResult):
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(self.format_output(result), ensure_ascii=False) + "\n")
```

If your saver can persist efficiently in groups, override `save_batch()` as well.

## Registration system

You can register custom loaders and savers without modifying core source files.

```python
from src.utils.registry import register_loader, register_saver
```

Then preload your module from YAML:

```yaml
custom_modules:
  - examples/custom_components.py
```

Resolution order:

1. Registered custom class
2. Built-in class map
3. Auto-import from `src.loaders.*` or `src.savers.*`

## Streaming and batch mode

`streaming: true` is still the default and recommended mode. Both streaming and batch now share the same bounded scheduler and writer pipeline in `BatchRunner`; the difference is mainly how the loader behaves.

Choose streaming when:

- the dataset is large
- you want steady memory usage
- you want outputs to appear continuously

Choose batch only when:

- the loader naturally precomputes a small dataset
- you are debugging or prototyping

## Resume behavior

Resume is saver-backed plus optional bitmap acceleration.

### `legacy_output_scan`

- Works with any saver that implements `is_completed()`
- Scans output state using `request_id`

### `bitmap`

- Best for JSONL and directory JSONL loaders
- Uses exact `(source_file, line_num, item_idx)` tracking

If a request has no `resume_key`, the runtime falls back to legacy checks automatically.

## Output formats

Built-in JSONL savers support:

- `output_projection: full`
- `output_projection: minimal`
- `output_fields`
- `include_timestamp`

This is the main knob for reducing serialization and disk pressure during rollout.

## Multimodal architecture

Multimodal loaders extend `MultimodalDataLoader`, which handles:

- image path resolution
- MIME-type detection
- optional base64 encoding
- thread-pooled image encoding
- OpenAI-compatible message formatting

For details, see [MULTIMODAL.md](/Users/yuhan/code/vllm_runner/docs/MULTIMODAL.md).

## Benchmark and soak utilities

The repository includes:

- `scripts/benchmark_load_balancer.py`
- `scripts/benchmark_writer.py`
- `scripts/soak_openai_stub.py`

These are the fastest way to validate performance regressions after runtime changes.
