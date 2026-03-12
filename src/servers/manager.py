"""
vLLM server manager module.

Manages a pool of vLLM servers with automatic discovery and health checking.
"""
import asyncio
import re
import time
import threading
import logging
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

try:
    import httpx
except ImportError:
    httpx = None


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
    inflight_cost: float = 0.0
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
        base_load = max(self.active_requests, int(self.inflight_cost))

        total = self.success_count + self.error_count
        success_rate = 1.0 if total == 0 else self.success_count / total

        if success_rate < 1.0:
            penalty_factor = 1.0 / max(success_rate, 0.1)  # Cap at 10x penalty
            return int(base_load * penalty_factor)
        return base_load

    def increment_active(self, cost: float = 1.0) -> int:
        """
        Increment active request count.

        Returns:
            New active request count
        """
        self.active_requests += 1
        self.inflight_cost += max(1.0, float(cost))
        return self.active_requests

    def decrement_active(self, cost: float = 1.0) -> int:
        """
        Decrement active request count.

        Returns:
            New active request count
        """
        self.active_requests = max(0, self.active_requests - 1)
        self.inflight_cost = max(0.0, self.inflight_cost - max(1.0, float(cost)))
        return self.active_requests

    def record_success(self, cost: float = 1.0) -> None:
        """Record a successful request completion."""
        self.success_count += 1
        self.request_count += 1
        self.active_requests = max(0, self.active_requests - 1)
        self.inflight_cost = max(0.0, self.inflight_cost - max(1.0, float(cost)))

    def record_error(self, cost: float = 1.0) -> None:
        """Record a failed request."""
        self.error_count += 1
        self.request_count += 1
        self.active_requests = max(0, self.active_requests - 1)
        self.inflight_cost = max(0.0, self.inflight_cost - max(1.0, float(cost)))


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
        self.health_check_timeout = config.get('health_check_timeout', 5)
        self.health_check_concurrency = max(1, config.get('health_check_concurrency', 32))
        self.http2 = config.get('http2', False)

        if httpx is None:
            raise ImportError("httpx library is required. Install it with: pip install httpx[http2]")

        self.servers: List[VLLMServer] = []
        self._lock = threading.Lock()
        self._stop_health_check = threading.Event()
        self._health_check_task: Optional[asyncio.Task] = None
        self._health_client = None
        self._state_change_callbacks: List[Callable[[VLLMServer, bool], None]] = []

        # Setup logger
        self.logger = logging.getLogger(__name__)

        self._discover_servers()

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

    async def start_async(self):
        """Start background async health checking if it is enabled."""
        if self.health_check_interval <= 0 or self._health_check_task is not None:
            return

        limits = httpx.Limits(
            max_connections=min(len(self.servers), self.health_check_concurrency),
            max_keepalive_connections=min(len(self.servers), self.health_check_concurrency),
        )
        self._health_client = httpx.AsyncClient(
            timeout=self.health_check_timeout,
            limits=limits,
            http2=self.http2,
        )
        await self._check_all_servers_async()
        self._stop_health_check.clear()
        self._health_check_task = asyncio.create_task(self._health_check_loop())

    async def _health_check_loop(self):
        """Periodically refresh server health in the active event loop."""
        try:
            while not self._stop_health_check.is_set():
                await asyncio.sleep(self.health_check_interval)
                if self._stop_health_check.is_set():
                    break
                await self._check_all_servers_async()
        except asyncio.CancelledError:
            raise

    async def _check_all_servers_async(self):
        """Check health of all servers with bounded parallelism."""
        if self._health_client is None:
            return

        servers = self.get_all_servers()
        semaphore = asyncio.Semaphore(self.health_check_concurrency)

        async def check(server: VLLMServer):
            async with semaphore:
                server.last_health_check = time.time()
                try:
                    response = await self._health_client.get(server.health_url())
                    return server, response.status_code == 200, response.status_code, None
                except httpx.TimeoutException as exc:
                    return server, False, None, exc
                except httpx.ConnectError as exc:
                    return server, False, None, exc
                except Exception as exc:  # pragma: no cover - defensive
                    return server, False, None, exc

        results = await asyncio.gather(*(check(server) for server in servers), return_exceptions=False)
        for server, is_healthy, status_code, error in results:
            self._apply_health_result(server, is_healthy, status_code=status_code, error=error)

    def _apply_health_result(
        self,
        server: VLLMServer,
        is_healthy: bool,
        status_code: Optional[int] = None,
        error: Optional[Exception] = None,
    ):
        """Apply a completed health probe result without holding a lock during I/O."""
        notify_state = None

        with self._lock:
            if is_healthy:
                previous_failures = server.failure_count
                server.failure_count = 0
                if not server.healthy:
                    server.healthy = True
                    server.last_state_change = time.time()
                    notify_state = True
                    self.logger.info(
                        "[HealthCheck] Server %s (%s:%s) is now HEALTHY (recovered after %s failures)",
                        server.name,
                        server.ip,
                        server.port,
                        previous_failures,
                    )
            else:
                server.failure_count += 1

                if status_code is not None:
                    self.logger.warning(
                        "[HealthCheck] Server %s returned status %s (failure %s/%s)",
                        server.name,
                        status_code,
                        server.failure_count,
                        self.max_failures,
                    )
                elif isinstance(error, httpx.TimeoutException):
                    self.logger.warning(
                        "[HealthCheck] Server %s timed out (failure %s/%s)",
                        server.name,
                        server.failure_count,
                        self.max_failures,
                    )
                elif isinstance(error, httpx.ConnectError):
                    self.logger.warning(
                        "[HealthCheck] Server %s connection error: %s (failure %s/%s)",
                        server.name,
                        str(error),
                        server.failure_count,
                        self.max_failures,
                    )
                else:
                    self.logger.error(
                        "[HealthCheck] Server %s unexpected error: %s (failure %s/%s)",
                        server.name,
                        str(error),
                        server.failure_count,
                        self.max_failures,
                    )

                if server.failure_count >= self.max_failures and server.healthy:
                    server.healthy = False
                    server.last_state_change = time.time()
                    notify_state = False
                    self.logger.error(
                        "[HealthCheck] Server %s (%s:%s) is now UNHEALTHY",
                        server.name,
                        server.ip,
                        server.port,
                    )

        if notify_state is not None:
            self._notify_state_change(server, notify_state)

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

    async def aclose(self):
        """Stop health checking and release async resources."""
        self._stop_health_check.set()
        if self._health_check_task is not None:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None

        if self._health_client is not None:
            await self._health_client.aclose()
            self._health_client = None

    def shutdown(self):
        """Compatibility wrapper for callers that do not await cleanup."""
        self._stop_health_check.set()
        if self._health_check_task is not None:
            self._health_check_task.cancel()
