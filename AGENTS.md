# AGENTS.md

Guidance for coding agents working in this repository.

## Quick Commands

```bash
# Install runtime deps
pip install -r requirements.txt

# Install editable package with dev tools
pip install -e ".[dev]"

# Run the main CLI
python -m src.cli --config configs/config.yaml

# Common CLI overrides
python -m src.cli --config configs/config.yaml --concurrency 20 --verbose
python -m src.cli --config configs/config.yaml --temperature 0.5 --max-tokens 2000

# Console script entry point
vllm-batch --config configs/config.yaml

# Tests
python -m pytest tests/ -v
python -m pytest tests/test_load_balancer.py -v
python -m pytest tests/test_multimodal.py -v

# Formatting / linting
black src tests
flake8 src tests
```

## Practical Setup Notes

- Prefer `pip install -r requirements.txt` before running the system. `setup.py` does not currently include every runtime dependency from `requirements.txt` such as `httpx[http2]`.
- On macOS, installing `brotli` can help enable better HTTP/2 support for `httpx`.

## Repository Snapshot

This project is a YAML-driven batch inference framework for OpenAI-compatible / vLLM-style servers.

Main responsibilities:

- Load requests from pluggable loaders.
- Route them across discovered servers with health checks and load balancing.
- Send requests concurrently with a shared `httpx.AsyncClient`.
- Normalize responses through a model adapter.
- Persist outputs through pluggable savers.

## Key Entry Points

- `src/cli.py`: CLI parsing, config loading, component construction.
- `src/batch_runner.py`: Main orchestration, async request flow, streaming vs batch mode.
- `src/utils/config.py`: Config defaults plus dynamic class loading.
- `src/utils/registry.py`: Decorator-based registration and `custom_modules` loading.
- `src/servers/manager.py`: Server discovery and background health checking.
- `src/servers/load_balancer.py`: Request routing strategies.

## High-Level Flow

1. `load_config()` reads YAML and imports any `custom_modules`.
2. `get_loader_class()`, `get_saver_class()`, and `get_adapter_class()` resolve components.
3. `BatchRunner` creates `VLLMServerManager`, `LoadBalancer`, `ProgressTracker`, and a shared async HTTP client.
4. Loader yields `LoadResult(messages, request_id, additional_data)`.
5. Runner builds the API request through the active adapter and sends it to a selected server.
6. Saver writes `SaveResult(request_id, model_output, additional_data, error)`.

## Built-In Components

Loaders:

- `JSONDataLoader`
- `MultimodalJSONDataLoader`
- `JSONLDataLoader`
- `MultimodalJSONLDataLoader`
- `CSVDataLoader`
- `PromptListLoader`
- `DirectoryJSONLDataLoader`
- `MultimodalDirectoryJSONLDataLoader`

Savers:

- `JSONResultSaver`
- `JSONLResultSaver`
- `CSVResultSaver`
- `ConsoleResultSaver`
- `DirectoryJSONLResultSaver`

Adapters:

- `OpenAIAdapter`
- `SimpleAdapter`

## Plugin Resolution Rules

Loader and saver lookup order in `src/utils/config.py`:

1. Custom classes registered with `@register_loader` / `@register_saver`
2. Built-in class maps
3. Auto-import from `src.loaders.<snake_case_name>` or `src.savers.<snake_case_name>`

Custom modules can be preloaded from config:

```yaml
custom_modules:
  - examples/custom_components.py
```

Important:

- `custom_modules` import failures are logged as warnings and do not fail config loading.
- Adapters do not use the registry helpers; they are resolved from built-ins or direct module import.

## Configuration Shape

Required top-level keys:

- `loader`
- `saver`
- `runner`

`logging` is optional, but keep it in configs if you plan to use `--verbose`, because `src/cli.py` mutates `config["logging"]["level"]`.

Useful runner fields:

- `max_concurrency`
- `max_retries`
- `retry_delay`
- `request_timeout`
- `http_max_connections`
- `http_max_keepalive_connections`
- `http2`
- `model_name`
- `temperature`
- `max_tokens`
- `top_p`
- `frequency_penalty`
- `presence_penalty`
- `system_prompt`
- `adapter_class`
- `adapter_params`
- `servers_dir`
- `load_balancing_strategy`
- `health_check_interval`
- `max_failures`
- `allow_unhealthy_fallback`
- `success_rate_threshold`
- `success_rate_window`
- `max_active_requests`
- `streaming`
- `stream_queue_size`
- `resume`
- `progress_report_interval`

CLI overrides are intentionally narrow: `--concurrency`, `--model`, `--temperature`, `--max-tokens`, and `--verbose`.

## Execution Modes

The runner supports two modes:

- Streaming mode: producer/consumer pipeline with bounded queue and immediate saving.
- Batch mode: preload all items, then process them.

Current defaults in code:

- `BatchConfig.streaming` defaults to `True`.
- `StreamingLoaderMixin` defaults loaders to `streaming=True`.
- `StreamingSaverMixin` defaults savers to `streaming=True` and `immediate_flush=True`.

When editing behavior, trust the code defaults over older prose comments in docs.

## Load Balancing and Server Management

Supported load-balancing strategies in code:

- `round_robin`
- `least_connections`
- `adaptive_round_robin`
- `random`

Notable behavior:

- `least_connections` and `adaptive_round_robin` both consider recent success rate.
- `allow_unhealthy_fallback` permits routing to unhealthy servers when no healthy server remains.
- `max_active_requests` is used by `adaptive_round_robin` as a congestion threshold.

Server discovery gotcha:

- `src/servers/manager.py` currently discovers entries in `servers_dir` whose names match `server_(.+)_(\d+)`, but the implementation skips directories and only inspects non-directory entries.
- Some existing docs describe directory-based discovery. If you touch this area, verify whether you are preserving current behavior or fixing an inconsistency.

## Resume and Request IDs

- Resume behavior is saver-driven through `ResultSaver.is_completed()`.
- Stable request IDs matter. In JSONL loaders, if no explicit ID field is present, `JSONLLoaderMixin` generates a deterministic SHA-256 hash from content.
- `DirectoryJSONLResultSaver` reconstructs completed IDs by scanning output files recursively.

## JSONL Customization

`src/loaders/jsonl_mixin.py` is the main extension point for custom JSONL behavior.

Useful hooks:

- `parse_line()`
- `should_skip_item()`
- `extract_request_id()`
- `extract_prompt()`
- `extract_additional_data()`
- `build_messages()` via `MessagesBuilderMixin`

Sample fission is supported:

- `parse_line()` may return either one dict or a list of dicts.
- Multi-item lines become multiple requests.
- Request IDs for split items get suffixes such as `_0`, `_1`, etc.

## Editing Guidance

- If you change request sending or retry logic, stay in the async `httpx` path used by `BatchRunner`.
- If you change loader or saver semantics, check whether resume behavior or streaming assumptions also need updates.
- If you change load balancing, run `tests/test_load_balancer.py`.
- If you change multimodal loading, run `tests/test_multimodal.py`.
- Docs under `docs/` and example configs under `configs/` are useful, but a few comments are stale. Prefer the source code when they disagree.

## Useful Files for Common Tasks

- New loader: start from `src/loaders/base.py`, `src/loaders/jsonl_loader.py`, and `src/loaders/jsonl_mixin.py`
- New saver: start from `src/savers/base.py`, `src/savers/jsonl_saver.py`, and `src/savers/streaming_mixin.py`
- New adapter: start from `src/adapters/base.py`
- Config loading changes: `src/utils/config.py`
- Server health / discovery changes: `src/servers/manager.py`
- Routing changes: `src/servers/load_balancer.py`
