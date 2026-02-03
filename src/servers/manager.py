"""
vLLM server manager module.

Manages a pool of vLLM servers with automatic discovery and health checking.
"""
import re
import time
import threading
import logging
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

try:
    import requests
except ImportError:
    requests = None


class ServerState(Enum):
    """Server state enumeration."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


@dataclass
class VLLMServer:
    """
    Represents a single vLLM server endpoint.

    Attributes:
        name: Server directory name (e.g., "server_127.0.0.1_8000")
        ip: Server IP address
        port: Server port
        healthy: Whether the server is healthy
        request_count: Number of requests handled
        failure_count: Number of consecutive failures
        last_health_check: Timestamp of last health check
        last_state_change: Timestamp of last state change
    """
    name: str
    ip: str
    port: int
    healthy: bool = True
    request_count: int = 0
    failure_count: int = 0
    last_health_check: float = field(default_factory=time.time)
    last_state_change: float = field(default_factory=time.time)

    @property
    def base_url(self) -> str:
        """Get the base URL for the server."""
        return f"http://{self.ip}:{self.port}"

    def health_url(self) -> str:
        """Get the health check URL."""
        return f"{self.base_url}/health"

    def chat_url(self) -> str:
        """Get the chat completions URL."""
        return f"{self.base_url}/v1/chat/completions"


class VLLMServerManager:
    """
    Manages a pool of vLLM servers with load balancing and health checking.

    Handles:
    - Server discovery from directory names
    - Health checking with automatic failover
    - Server recovery detection
    - Failure tracking and logging
    - Dynamic server list updates

    Configuration:
        servers_dir: Directory containing server subdirectories
        server_pattern: Regex pattern to parse server names (default: r'server_(.+)_(\\d+)')
        health_check_interval: Seconds between health checks (default: 30)
        max_failures: Max failures before marking unhealthy (default: 5)
        request_timeout: Request timeout in seconds (default: 120)
    """

    def __init__(self, config: Dict[str, Any]):
        self.servers_dir = Path(config.get('servers_dir', '.'))
        self.server_pattern = config.get('server_pattern', r'server_(.+)_(\d+)')
        self.health_check_interval = config.get('health_check_interval', 30)
        self.max_failures = config.get('max_failures', 5)
        self.timeout = config.get('request_timeout', 120)

        if requests is None:
            raise ImportError("requests library is required. Install it with: pip install requests")

        self.servers: List[VLLMServer] = []
        self._lock = threading.Lock()
        self._health_check_thread: Optional[threading.Thread] = None
        self._stop_health_check = threading.Event()
        self._state_change_callbacks: List[Callable[[VLLMServer, bool], None]] = []

        # Setup logger
        self.logger = logging.getLogger(__name__)

        self._discover_servers()
        self._start_health_checker()

    def _discover_servers(self):
        """Discover vLLM servers from directory names."""
        if not self.servers_dir.exists():
            raise ValueError(f"Servers directory not found: {self.servers_dir}")

        for item in self.servers_dir.iterdir():
            if not item.is_dir():
                continue

            match = re.match(self.server_pattern, item.name)
            if match:
                ip, port = match.groups()
                server = VLLMServer(
                    name=item.name,
                    ip=ip,
                    port=int(port)
                )
                self.servers.append(server)

        if not self.servers:
            raise ValueError(f"No vLLM servers found matching pattern '{self.server_pattern}' in {self.servers_dir}")

        print(f"[VLLMServerManager] Discovered {len(self.servers)} vLLM servers")

    def _start_health_checker(self):
        """Start background health checking thread."""
        def health_check_loop():
            while not self._stop_health_check.is_set():
                self._check_all_servers()
                self._stop_health_check.wait(self.health_check_interval)

        self._health_check_thread = threading.Thread(target=health_check_loop, daemon=True)
        self._health_check_thread.start()

    def _check_all_servers(self):
        """Check health of all servers."""
        with self._lock:
            for server in self.servers:
                server.last_health_check = time.time()

                try:
                    response = requests.get(
                        server.health_url(),
                        timeout=5
                    )
                    is_healthy = response.status_code == 200

                    if is_healthy:
                        server.failure_count = 0
                        if not server.healthy:
                            # Server recovered
                            server.healthy = True
                            server.last_state_change = time.time()
                            self.logger.info(
                                f"[HealthCheck] Server {server.name} ({server.ip}:{server.port}) "
                                f"is now HEALTHY (recovered after {server.failure_count} failures)"
                            )
                            self._notify_state_change(server, True)
                    else:
                        server.failure_count += 1
                        self.logger.warning(
                            f"[HealthCheck] Server {server.name} returned status {response.status_code} "
                            f"(failure {server.failure_count}/{self.max_failures})"
                        )
                        if server.failure_count >= self.max_failures and server.healthy:
                            server.healthy = False
                            server.last_state_change = time.time()
                            self.logger.error(
                                f"[HealthCheck] Server {server.name} ({server.ip}:{server.port}) "
                                f"is now UNHEALTHY (marked as down after {server.failure_count} consecutive failures)"
                            )
                            self._notify_state_change(server, False)

                except requests.exceptions.Timeout:
                    server.failure_count += 1
                    self.logger.warning(
                        f"[HealthCheck] Server {server.name} timed out "
                        f"(failure {server.failure_count}/{self.max_failures})"
                    )
                    if server.failure_count >= self.max_failures and server.healthy:
                        server.healthy = False
                        server.last_state_change = time.time()
                        self.logger.error(
                            f"[HealthCheck] Server {server.name} ({server.ip}:{server.port}) "
                            f"is now UNHEALTHY (timeout after {server.failure_count} consecutive failures)"
                        )
                        self._notify_state_change(server, False)

                except requests.exceptions.ConnectionError as e:
                    server.failure_count += 1
                    self.logger.warning(
                        f"[HealthCheck] Server {server.name} connection error: {str(e)} "
                        f"(failure {server.failure_count}/{self.max_failures})"
                    )
                    if server.failure_count >= self.max_failures and server.healthy:
                        server.healthy = False
                        server.last_state_change = time.time()
                        self.logger.error(
                            f"[HealthCheck] Server {server.name} ({server.ip}:{server.port}) "
                            f"is now UNHEALTHY (connection failed after {server.failure_count} consecutive failures)"
                        )
                        self._notify_state_change(server, False)

                except Exception as e:
                    server.failure_count += 1
                    self.logger.error(
                        f"[HealthCheck] Server {server.name} unexpected error: {str(e)} "
                        f"(failure {server.failure_count}/{self.max_failures})"
                    )
                    if server.failure_count >= self.max_failures and server.healthy:
                        server.healthy = False
                        server.last_state_change = time.time()
                        self.logger.error(
                            f"[HealthCheck] Server {server.name} ({server.ip}:{server.port}) "
                            f"is now UNHEALTHY (error after {server.failure_count} consecutive failures)"
                        )
                        self._notify_state_change(server, False)

    def get_healthy_servers(self) -> List[VLLMServer]:
        """Get list of healthy servers."""
        with self._lock:
            return [s for s in self.servers if s.healthy]

    def get_server_count(self) -> int:
        """Get count of healthy servers."""
        return len(self.get_healthy_servers())

    def get_all_servers(self) -> List[VLLMServer]:
        """Get all servers (including unhealthy ones)."""
        with self._lock:
            return list(self.servers)

    def register_state_change_callback(self, callback: Callable[[VLLMServer, bool], None]):
        """
        Register a callback to be notified when a server changes state.

        Args:
            callback: Function that takes (server, is_healthy) as arguments
        """
        with self._lock:
            self._state_change_callbacks.append(callback)

    def _notify_state_change(self, server: VLLMServer, is_healthy: bool):
        """Notify all registered callbacks of a state change."""
        for callback in self._state_change_callbacks:
            try:
                callback(server, is_healthy)
            except Exception as e:
                self.logger.error(f"Error in state change callback: {e}")

    def shutdown(self):
        """Stop health checker and cleanup resources."""
        self._stop_health_check.set()
        if self._health_check_thread:
            self._health_check_thread.join(timeout=5)
