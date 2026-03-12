#!/usr/bin/env python3
"""
Microbenchmark for JSONL saver batch throughput.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.savers.base import SaveResult
from src.savers.jsonl_saver import JSONLResultSaver


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark JSONL saver batch throughput.")
    parser.add_argument("--rows", type=int, default=50000, help="Number of synthetic rows to write.")
    parser.add_argument(
        "--projection",
        choices=["full", "minimal"],
        default="minimal",
        help="Output projection to benchmark.",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep the generated JSONL file instead of deleting the temp directory.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    temp_dir_ctx = tempfile.TemporaryDirectory(prefix="vllm_runner_writer_")
    tmp_dir = temp_dir_ctx.name
    try:
        output_path = Path(tmp_dir) / "results.jsonl"
        saver = JSONLResultSaver(
            {
                "output_path": str(output_path),
                "output_projection": args.projection,
                "immediate_flush": False,
            }
        )
        results = [
            SaveResult(
                request_id=f"req_{index}",
                model_output={
                    "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
                    "usage": {"total_tokens": 10},
                },
                additional_data={"source": "bench", "index": index},
            )
            for index in range(args.rows)
        ]

        start = time.perf_counter()
        saver.save_batch(results)
        saver.cleanup()
        elapsed = time.perf_counter() - start
        print(f"Wrote {args.rows} rows in {elapsed:.4f}s ({args.rows / elapsed:.1f} rows/s)")
        if args.keep_artifacts:
            print(f"Artifacts: {output_path}")
        else:
            print("Artifacts removed after benchmark")
    finally:
        if not args.keep_artifacts:
            temp_dir_ctx.cleanup()


if __name__ == "__main__":
    main()
