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
        success_count: Number of successful requests
        error_count: Number of request errors (timeouts, failures, etc.)
        active_requests: Number of currently active (in-flight) requests
        _lock: Thread lock for atomic updates to stats
    """
    name: str
    ip: str
    port: int
    healthy: bool = True
    request_count: int = 0
    failure_count: int = 0
    last_health_check: float = field(default_factory=time.time)
    last_state_change: float = field(default_factory=time.time)
    success_count: int = 0
    error_count: int = 0
    active_requests: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

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

    @property
    def success_rate(self) -> float:
        """
        Calculate success rate for this server.

        Returns:
            Success rate as a float between 0.0 and 1.0.
            Returns 1.0 if no requests have been made.
        """
        with self._lock:
            total = self.success_count + self.error_count
            if total == 0:
                return 1.0
            return self.success_count / total

    @property
    def effective_load(self) -> int:
        """
        Get the effective load considering both active requests and error rate.

        Servers with low success rates have higher effective load to discourage
        routing when they're underperforming.

        Returns:
            Effective load score (higher = more loaded/less desirable)
        """
        with self._lock:
            # Base load is active requests
            base_load = self.active_requests

            # Apply penalty for low success rate
            # If success rate is 50%, effective load doubles
            # If success rate is 25%, effective load quadruples
            total = self.success_count + self.error_count
            if total == 0:
                success_rate = 1.0
            else:
                success_rate = self.success_count / total

            if success_rate < 1.0:
                penalty_factor = 1.0 / max(success_rate, 0.1)  # Cap at 10x penalty
                return int(base_load * penalty_factor)
            return base_load

    def increment_active(self) -> int:
        """
        Increment active request count.

        Returns:
            New active request count
        """
        with self._lock:
            self.active_requests += 1
            return self.active_requests

    def decrement_active(self) -> int:
        """
        Decrement active request count.

        Returns:
            New active request count
        """
        with self._lock:
            self.active_requests = max(0, self.active_requests - 1)
            return self.active_requests

    def record_success(self) -> None:
        """Record a successful request completion."""
        with self._lock:
            self.success_count += 1
            self.request_count += 1
            self.active_requests = max(0, self.active_requests - 1)

    def record_error(self) -> None:
        """Record a failed request."""
        with self._lock:
            self.error_count += 1
            self.request_count += 1
            self.active_requests = max(0, self.active_requests - 1)


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
            if item.is_dir():
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
        if self._health_check_thread and self._health_check_thread.is_alive():
            # Wait for thread to finish (may be in middle of HTTP request)
            self._health_check_thread.join(timeout=10)
