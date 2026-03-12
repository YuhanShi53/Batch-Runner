# High Concurrency Optimization

This document describes the current high-throughput runtime in `vllm_runner`. It replaces older notes that referred to per-request event loops or per-thread async clients.

## Runtime model

The hot path is now a bounded async pipeline:

1. A background producer reads `LoadResult` objects from the loader.
2. The scheduler keeps at most `runner.max_concurrency` requests in flight.
3. Each request coroutine sends HTTP with the shared `httpx.AsyncClient`.
4. Completed responses are pushed into a completion queue.
5. Writer workers batch `SaveResult` objects and call `saver.save_batch()`.
6. Resume state and progress counters are updated after a successful batch flush.

This removes the old "request completes only after synchronous save" bottleneck.

## What changed

### Batched writer pipeline

Request coroutines no longer block on:

- `saver.save()`
- thread-pool progress updates
- per-result flushes

The write path is controlled by:

- `writer_queue_size`
- `writer_batch_size`
- `writer_flush_interval_ms`
- `writer_workers`

Built-in JSONL savers implement efficient `save_batch()`.

### Shared async HTTP client

`BatchRunner` owns one shared `httpx.AsyncClient` with configurable:

- `http_max_connections`
- `http_max_keepalive_connections`
- `http2`

Payloads are serialized once per request and sent as bytes.

### Constant-time-friendly load balancing

`LoadBalancer` supports:

- `round_robin`
- `least_connections`
- `adaptive_round_robin`
- `load_aware_round_robin`
- `p2c_cost_aware`
- `random`

For large server pools, `p2c_cost_aware` is the preferred default. It uses small-sample selection instead of scanning every server on every request.

### Cost-aware admission

Each request can carry `dispatch_cost`, and each server tracks:

- `active_requests`
- `inflight_cost`

This lets routing differentiate short and long prompts instead of relying only on connection count.

### Bitmap resume

For JSONL-style workloads, `bitmap` resume stores completion state using:

`resume_key = (source_file, line_num, item_idx)`

This avoids loading millions of `request_id` strings into RAM on startup.

### Async health checks

Server health checks now run as an async background task with bounded concurrency. Network I/O does not happen while holding the manager lock.

## Recommended profile

For large rollout jobs, start from:

```yaml
loader:
  class: DirectoryJSONLDataLoader
  params:
    input_dir: data/input
    streaming: true

saver:
  class: DirectoryJSONLResultSaver
  params:
    output_dir: outputs/results
    output_projection: minimal
    immediate_flush: false

runner:
  max_concurrency: 4096
  max_retries: 1
  retry_delay: 0.5
  request_timeout: 300

  http_max_connections: 4096
  http_max_keepalive_connections: 1000
  http2: true

  load_balancing_strategy: p2c_cost_aware
  selection_sample_size: 2
  max_active_requests: 512
  max_inflight_cost: 0

  producer_prefetch: 2048
  writer_queue_size: 4096
  writer_batch_size: 128
  writer_flush_interval_ms: 100
  writer_workers: 1

  resume: true
  resume_backend: bitmap
```

For a full example, see [configs/high_concurrency_config.yaml](/Users/yuhan/code/vllm_runner/configs/high_concurrency_config.yaml).

## Tuning guide

### `max_concurrency`

- Set this from the client-side rollout target.
- Increase gradually while watching client CPU, file descriptors, and server queueing.

### `http_max_connections`

- Usually set near `max_concurrency` for a single shared client.
- Too low throttles the client.
- Too high wastes memory and sockets.

### `writer_batch_size`

- Larger batches reduce lock and flush overhead.
- Too large increases tail latency before results appear on disk.
- `64` to `256` is a good starting range.

### `writer_flush_interval_ms`

- Lower values make progress more visible.
- Higher values reduce write amplification.
- `50` to `250` ms is usually a good tradeoff.

### `selection_sample_size`

- Used by `p2c_cost_aware`.
- `2` is the cheapest and usually enough.
- `3` or `4` can help if server heterogeneity is high.

### `max_inflight_cost`

- Set this if long prompts cause head-of-line blocking.
- Leave `0` to disable cost cap.
- When enabled, pick a value that reflects the maximum prompt budget you want per server.

### `output_projection`

- Use `minimal` for throughput-oriented rollout.
- Use `full` when downstream consumers need the entire raw response.

## Loader-specific advice

### JSONL and directory JSONL

These are the best fit for large jobs because they provide stable `resume_key` values and work well with bitmap resume.

### Multimodal

When `encode_images: true`, use:

- `image_encode_workers`
- `output_projection: minimal`
- conservative `max_concurrency` relative to text-only jobs

This reduces event-loop stalls and disk pressure.

## Benchmark scripts

### Load balancer microbenchmark

```bash
python scripts/benchmark_load_balancer.py --servers 400 --iterations 50000
```

This measures raw selection throughput across strategies.

### Writer microbenchmark

```bash
python scripts/benchmark_writer.py --rows 50000 --projection minimal
```

This measures JSONL batch write throughput.

### End-to-end stub soak

```bash
python scripts/soak_openai_stub.py \
  --servers 8 \
  --requests 2000 \
  --concurrency 512 \
  --strategy p2c_cost_aware \
  --report-json
```

This runs a full local pipeline with:

- mock OpenAI-compatible servers
- synthetic directory JSONL input
- real `BatchRunner`
- bitmap resume
- batched saver

## Validation checklist

- The server marker directory contains files named `server_<ip>_<port>`.
- `output_projection` is set intentionally.
- `resume_backend` is `bitmap` for JSONL / directory JSONL jobs.
- `load_balancing_strategy` is `p2c_cost_aware` unless you have a strong reason otherwise.
- `immediate_flush` is `false` for throughput-oriented jobs.
- `httpx[http2]` is installed if you enable `http2: true`.

## Failure patterns to watch

- Rising `failed_requests` with low server utilization: the client may be under-provisioned or retrying the same hot server.
- High client CPU with low disk activity: load balancing or serialization is likely dominating.
- High disk activity and low req/s: writer batch size is too small or `immediate_flush` is enabled.
- Slow multimodal throughput: image encoding is happening synchronously or concurrency is too high for the model.
