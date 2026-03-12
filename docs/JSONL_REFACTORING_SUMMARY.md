# JSONL Refactoring Summary

This document is kept as a short index after the high-concurrency runtime update.

## Current status

The JSONL path is now the primary large-scale rollout path in `vllm_runner`.

Key properties:

- JSONL loaders expose stable `request_id`
- JSONL and directory JSONL loaders emit `resume_key`
- bitmap resume works best on JSONL-style inputs
- loaders can surface `dispatch_cost` for cost-aware routing
- JSONL savers support batched `save_batch()`
- `output_projection: minimal` reduces serialization and disk overhead

## Main files

- [src/loaders/jsonl_mixin.py](/Users/yuhan/code/vllm_runner/src/loaders/jsonl_mixin.py)
- [src/loaders/jsonl_loader.py](/Users/yuhan/code/vllm_runner/src/loaders/jsonl_loader.py)
- [src/loaders/directory_jsonl_loader.py](/Users/yuhan/code/vllm_runner/src/loaders/directory_jsonl_loader.py)
- [src/savers/jsonl_mixin.py](/Users/yuhan/code/vllm_runner/src/savers/jsonl_mixin.py)
- [src/savers/jsonl_saver.py](/Users/yuhan/code/vllm_runner/src/savers/jsonl_saver.py)
- [src/savers/directory_jsonl_saver.py](/Users/yuhan/code/vllm_runner/src/savers/directory_jsonl_saver.py)

## Use these docs instead

- [docs/JSONL_CUSTOMIZATION.md](/Users/yuhan/code/vllm_runner/docs/JSONL_CUSTOMIZATION.md)
- [docs/HIGH_CONCURRENCY_OPTIMIZATION.md](/Users/yuhan/code/vllm_runner/docs/HIGH_CONCURRENCY_OPTIMIZATION.md)
- [docs/FRAMEWORK_GUIDE.md](/Users/yuhan/code/vllm_runner/docs/FRAMEWORK_GUIDE.md)
