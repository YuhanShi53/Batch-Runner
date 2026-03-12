#!/usr/bin/env python3
"""
End-to-end soak harness against local OpenAI-compatible mock servers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for entry in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from perf_utils import (
    create_directory_jsonl_dataset,
    create_server_markers,
    create_temp_workspace,
    start_mock_openai_servers,
    stop_mock_openai_servers,
)
from src.adapters.openai_adapter import OpenAIAdapter
from src.batch_runner import BatchConfig, BatchRunner
from src.loaders.directory_jsonl_loader import (
    DirectoryJSONLDataLoader,
    MultimodalDirectoryJSONLDataLoader,
)
from src.savers.directory_jsonl_saver import DirectoryJSONLResultSaver


def ensure_localhost_no_proxy() -> None:
    """Bypass system HTTP proxies for local stub servers."""
    no_proxy_values = ["127.0.0.1", "localhost"]
    for key in ("NO_PROXY", "no_proxy"):
        existing = os.environ.get(key, "")
        parts = [part.strip() for part in existing.split(",") if part.strip()]
        merged = parts[:]
        for value in no_proxy_values:
            if value not in merged:
                merged.append(value)
        os.environ[key] = ",".join(merged)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an end-to-end soak test with local stub servers.")
    parser.add_argument("--servers", type=int, default=8, help="Number of local mock servers.")
    parser.add_argument("--requests", type=int, default=2000, help="Total synthetic requests.")
    parser.add_argument("--concurrency", type=int, default=512, help="Runner max concurrency.")
    parser.add_argument("--files", type=int, default=8, help="Number of input conv.jsonl shards.")
    parser.add_argument("--multimodal-rate", type=float, default=0.0, help="Fraction of requests with one image.")
    parser.add_argument("--long-input-rate", type=float, default=0.5, help="Fraction of long prompts.")
    parser.add_argument(
        "--strategy",
        default="p2c_cost_aware",
        help="Load balancing strategy to use.",
    )
    parser.add_argument("--base-delay-ms", type=int, default=5, help="Base server latency in milliseconds.")
    parser.add_argument("--jitter-ms", type=int, default=5, help="Additional random server latency.")
    parser.add_argument("--selection-sample-size", type=int, default=2, help="Sample size for p2c_cost_aware.")
    parser.add_argument("--max-inflight-cost", type=float, default=0.0, help="Optional per-server cost cap.")
    parser.add_argument("--writer-batch-size", type=int, default=128, help="Writer batch size.")
    parser.add_argument(
        "--writer-flush-interval-ms",
        type=int,
        default=100,
        help="Writer flush interval in milliseconds.",
    )
    parser.add_argument(
        "--output-projection",
        choices=["full", "minimal"],
        default="minimal",
        help="Saver output projection.",
    )
    parser.add_argument(
        "--report-json",
        action="store_true",
        help="Emit a final JSON summary in addition to the human-readable line.",
    )
    parser.add_argument("--keep-artifacts", action="store_true", help="Keep generated input/output artifacts.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ensure_localhost_no_proxy()
    workspace_ctx = create_temp_workspace()
    workspace = Path(workspace_ctx.name)
    servers = []

    try:
        dataset_meta = create_directory_jsonl_dataset(
            root=workspace,
            total_requests=args.requests,
            num_files=args.files,
            multimodal_rate=args.multimodal_rate,
            long_input_rate=args.long_input_rate,
        )
        servers, ports = start_mock_openai_servers(
            args.servers,
            base_delay_ms=args.base_delay_ms,
            jitter_ms=args.jitter_ms,
        )
        create_server_markers(workspace / "servers", ports)

        loader_class = (
            MultimodalDirectoryJSONLDataLoader
            if args.multimodal_rate > 0
            else DirectoryJSONLDataLoader
        )
        loader = loader_class(
            {
                "input_dir": dataset_meta["input_dir"],
                "streaming": True,
                "encode_images": args.multimodal_rate > 0,
                "image_encode_workers": 8,
            }
        )
        saver = DirectoryJSONLResultSaver(
            {
                "output_dir": str(workspace / "outputs"),
                "output_projection": args.output_projection,
                "immediate_flush": False,
            }
        )
        config = BatchConfig(
            max_concurrency=args.concurrency,
            max_retries=1,
            retry_delay=0.1,
            request_timeout=120,
            http_max_connections=max(args.concurrency, 1024),
            http_max_keepalive_connections=max(args.servers * 2, 64),
            http2=False,
            model_name="mock-model",
            streaming=True,
            producer_prefetch=max(args.concurrency, 256),
            writer_queue_size=max(args.concurrency, 512),
            writer_batch_size=max(1, args.writer_batch_size),
            writer_flush_interval_ms=max(1, args.writer_flush_interval_ms),
            writer_workers=1,
            resume=True,
            resume_backend="bitmap",
            servers_dir=str(workspace / "servers"),
            load_balancing_strategy=args.strategy,
            max_active_requests=max(32, args.concurrency // max(1, args.servers) * 4),
            selection_sample_size=max(2, args.selection_sample_size),
            max_inflight_cost=max(0.0, args.max_inflight_cost),
            progress_report_interval=5,
            image_encode_workers=8,
        )
        config.adapter = OpenAIAdapter()

        runner = BatchRunner(config, loader, saver)
        start = time.perf_counter()
        runner.run()
        elapsed = time.perf_counter() - start

        completed = runner.stats.completed_requests
        failed = runner.stats.failed_requests
        retried = runner.stats.retried_requests
        total_tokens = runner.stats.total_tokens
        throughput = completed / elapsed if elapsed > 0 else 0.0
        print(f"Completed {completed} requests in {elapsed:.2f}s ({throughput:.1f} req/s)")
        print(f"Failed {failed}, retried {retried}, total_tokens {total_tokens}")
        if args.report_json:
            print(
                json.dumps(
                    {
                        "completed_requests": completed,
                        "failed_requests": failed,
                        "retried_requests": retried,
                        "total_tokens": total_tokens,
                        "duration_seconds": round(elapsed, 4),
                        "throughput_req_per_sec": round(throughput, 4),
                        "workspace": str(workspace),
                        "strategy": args.strategy,
                        "concurrency": args.concurrency,
                        "servers": args.servers,
                    },
                    ensure_ascii=False,
                )
            )
        print(f"Artifacts: {workspace}")
    finally:
        stop_mock_openai_servers(servers)
        if args.keep_artifacts:
            print(f"Kept artifacts in {workspace}")
        else:
            workspace_ctx.cleanup()


if __name__ == "__main__":
    main()
