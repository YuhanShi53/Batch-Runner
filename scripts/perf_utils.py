"""
Shared helpers for local benchmark and soak scripts.
"""
from __future__ import annotations

import base64
import math
import json
import random
import socket
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


SMALL_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/w8AAgMBgN7x7sQAAAAASUVORK5CYII="
)


def find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def build_prompt(index: int, long_input: bool = False) -> str:
    """Build a deterministic short or long prompt."""
    if long_input:
        body = " ".join(f"token_{index}_{i}" for i in range(512))
        return f"Sample {index}: summarize and reason over this long context. {body}"
    return f"Sample {index}: answer briefly."


def write_small_png(path: Path) -> Path:
    """Write a tiny valid PNG to disk."""
    path.write_bytes(base64.b64decode(SMALL_PNG_BASE64))
    return path


def create_server_markers(servers_dir: Path, ports: Iterable[int]) -> None:
    """Create empty server marker files for VLLMServerManager discovery."""
    servers_dir.mkdir(parents=True, exist_ok=True)
    for port in ports:
        marker = servers_dir / f"server_127.0.0.1_{port}"
        marker.write_text("", encoding="utf-8")


def create_directory_jsonl_dataset(
    root: Path,
    total_requests: int,
    num_files: int = 8,
    multimodal_rate: float = 0.0,
    long_input_rate: float = 0.5,
    seed: int = 7,
) -> Dict[str, str]:
    """Create a synthetic directory JSONL dataset for soak tests."""
    rng = random.Random(seed)
    input_dir = root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    image_path = write_small_png(root / "tiny.png")

    requests_per_file = max(1, math.ceil(total_requests / max(1, num_files)))
    created = 0
    for file_index in range(num_files):
        shard_dir = input_dir / f"shard_{file_index:03d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        file_path = shard_dir / "conv.jsonl"

        with file_path.open("w", encoding="utf-8") as handle:
            for _ in range(requests_per_file):
                if created >= total_requests:
                    break
                long_input = rng.random() < long_input_rate
                multimodal = rng.random() < multimodal_rate
                record = {
                    "id": f"req_{created}",
                    "prompt": build_prompt(created, long_input=long_input),
                    "category": "long" if long_input else "short",
                    "dispatch_cost": 1024 if long_input else 32,
                }
                if multimodal:
                    record["image"] = str(image_path)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                created += 1

    return {
        "input_dir": str(input_dir),
        "image_path": str(image_path),
    }


def start_mock_openai_servers(
    num_servers: int,
    base_delay_ms: int = 5,
    jitter_ms: int = 5,
    completion_tokens_short: int = 32,
    completion_tokens_long: int = 256,
) -> Tuple[List[ThreadingHTTPServer], List[int]]:
    """Start local OpenAI-compatible mock servers."""
    servers = []
    ports = []

    def handler_factory(server_index: int):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):  # noqa: A003
                return

            def do_GET(self):  # noqa: N802
                if self.path == "/health":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"status":"ok"}')
                    return

                self.send_response(404)
                self.end_headers()

            def do_POST(self):  # noqa: N802
                if self.path != "/v1/chat/completions":
                    self.send_response(404)
                    self.end_headers()
                    return

                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                messages = payload.get("messages", [])

                prompt_chars = 0
                image_count = 0
                for message in messages:
                    content = message.get("content")
                    if isinstance(content, str):
                        prompt_chars += len(content)
                    elif isinstance(content, list):
                        for part in content:
                            if part.get("type") == "text":
                                prompt_chars += len(part.get("text", ""))
                            elif part.get("type") == "image_url":
                                image_count += 1

                prompt_tokens = max(1, prompt_chars // 4) + (image_count * 256)
                completion_tokens = completion_tokens_long if prompt_tokens > 512 else completion_tokens_short
                sleep_ms = base_delay_ms + random.randint(0, jitter_ms) + min(prompt_tokens // 16, 100)
                sleep_ms += server_index % 7
                time.sleep(sleep_ms / 1000.0)

                content = "ok " * completion_tokens
                response = {
                    "id": f"mock-{server_index}-{time.time_ns()}",
                    "object": "chat.completion",
                    "model": payload.get("model", "mock-model"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content.strip()},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                }
                body = json.dumps(response, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return Handler

    for index in range(num_servers):
        port = find_free_port()
        server = ThreadingHTTPServer(("127.0.0.1", port), handler_factory(index))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        server._thread = thread  # type: ignore[attr-defined]
        servers.append(server)
        ports.append(port)

    return servers, ports


def stop_mock_openai_servers(servers: Iterable[ThreadingHTTPServer]) -> None:
    """Stop and join local mock servers."""
    for server in servers:
        server.shutdown()
        server.server_close()
        thread = getattr(server, "_thread", None)
        if thread is not None:
            thread.join(timeout=5)


def create_temp_workspace(prefix: str = "vllm_runner_perf_") -> tempfile.TemporaryDirectory:
    """Create a temporary workspace for benchmark artifacts."""
    return tempfile.TemporaryDirectory(prefix=prefix)
