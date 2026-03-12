# Refactoring Summary

This summary reflects the current runtime after the high-concurrency rewrite.

## Main outcomes

- `BatchRunner` now uses a bounded async scheduler instead of per-request save/progress blocking.
- Result persistence is handled by a dedicated batched writer pipeline.
- JSONL and directory JSONL are the primary optimized data path.
- Resume can use compact bitmap state through `resume_key`.
- Load balancing supports `p2c_cost_aware` for large server pools.
- Multimodal image encoding can run in a worker pool instead of on the main event loop.

## New runtime building blocks

- [src/batch_runner.py](/Users/yuhan/code/vllm_runner/src/batch_runner.py)
- [src/servers/load_balancer.py](/Users/yuhan/code/vllm_runner/src/servers/load_balancer.py)
- [src/servers/manager.py](/Users/yuhan/code/vllm_runner/src/servers/manager.py)
- [src/utils/resume.py](/Users/yuhan/code/vllm_runner/src/utils/resume.py)
- [src/utils/json_codec.py](/Users/yuhan/code/vllm_runner/src/utils/json_codec.py)

## Configuration additions

Important runner settings added in the current runtime:

- `producer_prefetch`
- `writer_queue_size`
- `writer_batch_size`
- `writer_flush_interval_ms`
- `writer_workers`
- `resume_backend`
- `selection_sample_size`
- `max_inflight_cost`
- `image_encode_workers`

Important saver settings:

- `output_projection`
- `output_fields`
- `include_timestamp`

## Recommended references

- [README.md](/Users/yuhan/code/vllm_runner/README.md)
- [docs/FRAMEWORK_GUIDE.md](/Users/yuhan/code/vllm_runner/docs/FRAMEWORK_GUIDE.md)
- [docs/HIGH_CONCURRENCY_OPTIMIZATION.md](/Users/yuhan/code/vllm_runner/docs/HIGH_CONCURRENCY_OPTIMIZATION.md)
- [docs/MULTIMODAL.md](/Users/yuhan/code/vllm_runner/docs/MULTIMODAL.md)
