#!/usr/bin/env python3
"""
Microbenchmark for load balancer selection throughput.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.servers.load_balancer import LoadBalancer
from src.servers.manager import VLLMServer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark load balancer strategy throughput.")
    parser.add_argument("--servers", type=int, default=400, help="Number of synthetic servers.")
    parser.add_argument("--iterations", type=int, default=50000, help="Selection iterations per strategy.")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["round_robin", "load_aware_round_robin", "p2c_cost_aware", "random"],
        help="Strategies to benchmark.",
    )
    parser.add_argument("--sample-size", type=int, default=2, help="Sample size for p2c_cost_aware.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    servers = [VLLMServer(name=f"server_{i}", ip="127.0.0.1", port=8000 + i) for i in range(args.servers)]

    for index, server in enumerate(servers):
        server.active_requests = index % 32
        server.inflight_cost = float((index % 32) * 16)
        server.success_count = 100
        server.error_count = index % 3

    print(f"Benchmarking {len(servers)} synthetic servers, {args.iterations} selections/strategy")
    for strategy in args.strategies:
        lb = LoadBalancer(
            servers,
            strategy=strategy,
            max_active_requests=512,
            success_rate_window=5,
            selection_sample_size=args.sample_size,
        )
        start = time.perf_counter()
        for _ in range(args.iterations):
            lb.get_server()
        elapsed = time.perf_counter() - start
        print(f"{strategy:24s} {elapsed:8.4f}s  {args.iterations / elapsed:12.1f} ops/s")


if __name__ == "__main__":
    main()
