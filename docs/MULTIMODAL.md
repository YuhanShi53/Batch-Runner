# Multimodal Guide

This guide covers the current multimodal path in `vllm_runner`.

## Supported loaders

- `MultimodalJSONDataLoader`
- `MultimodalJSONLDataLoader`
- `MultimodalDirectoryJSONLDataLoader`

All of them build OpenAI-compatible vision messages and inherit the same image-processing behavior from `MultimodalDataLoader`.

## Input formats

### Single image

```jsonl
{"id": "1", "prompt": "Describe this image", "image": "images/sample.jpg"}
```

### Multiple images

```jsonl
{"id": "2", "prompt": "Compare these images", "images": ["a.jpg", "b.png"]}
```

### Directory JSONL

Each `conv.jsonl` line can carry the same fields:

```jsonl
{"id": "3", "prompt": "What is happening here?", "image": "frames/frame_001.png"}
```

## Configuration

```yaml
loader:
  class: MultimodalJSONLDataLoader
  params:
    file_path: data/vision_prompts.jsonl
    prompt_field: prompt
    id_field: id
    image_field: image
    images_field: images
    image_base_dir: data/images
    encode_images: true
    image_encode_workers: 8

saver:
  class: JSONLResultSaver
  params:
    output_path: results/vision_results.jsonl
    output_projection: minimal

runner:
  model_name: llava-hf/llava-1.5-7b-hf
  max_concurrency: 32
  request_timeout: 180
  load_balancing_strategy: p2c_cost_aware
  resume_backend: bitmap
```

## Image sources

The multimodal path supports:

- absolute file paths
- relative file paths resolved against `image_base_dir`
- remote URLs
- pre-encoded `data:` URIs

## Encoding behavior

When `encode_images: true`:

- file paths are read from disk
- bytes are converted to base64 data URIs
- a small in-memory cache avoids repeat work
- encoding can run in a thread pool using `image_encode_workers`

When `encode_images: false`:

- image strings are passed through
- this is best for URLs or already-encoded inputs

## Event-loop safety

The current implementation avoids doing heavy image encoding on the main async event loop:

- `image_encode_workers > 1` enables a `ThreadPoolExecutor`
- repeated image references can hit the loader cache
- the request pipeline remains responsive under mixed text and image workloads

## Message format

Loaders generate OpenAI-compatible content lists.

Example:

```json
[
  {
    "role": "user",
    "content": [
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
      {"type": "text", "text": "Describe this image"}
    ]
  }
]
```

The OpenAI adapter uses this structure directly when building the request.

## Resume and metadata

JSONL-based multimodal loaders still emit:

- stable `request_id`
- `resume_key` for bitmap resume
- `dispatch_cost`

This means multimodal jobs can use the same resume and cost-aware routing as text jobs.

## Throughput tips

- Use `output_projection: minimal` unless you need the entire raw response.
- Increase `image_encode_workers` when image files are local and CPU is available.
- Reduce `max_concurrency` relative to text-only jobs if the model is GPU-heavy.
- Prefer URLs or cached local paths over repeatedly decoding large images.
- Use `DirectoryJSONLResultSaver` for large directory-based datasets.

## Troubleshooting

### Image file not found

Check:

- the path in the JSON/JSONL record
- `image_base_dir`
- file permissions on the image file

### Unsupported image format

Supported suffixes are:

- `.jpg`
- `.jpeg`
- `.png`
- `.gif`
- `.webp`
- `.bmp`

### Vision model rejects the request

Check:

- the server is running a vision-capable model
- the model supports OpenAI-style image input
- your input is either a valid URL or a valid `data:` URI

### Mixed workload is too slow

Try:

- lower `max_concurrency`
- higher `image_encode_workers`
- `output_projection: minimal`
- `load_balancing_strategy: p2c_cost_aware`

## Local smoke test

You can exercise the multimodal runtime with the stub soak script:

```bash
python scripts/soak_openai_stub.py \
  --servers 4 \
  --requests 200 \
  --concurrency 32 \
  --files 4 \
  --multimodal-rate 0.5 \
  --long-input-rate 0.5 \
  --report-json
```
