"""
Load balancer module for distributing requests across vLLM servers.
"""
import random
import threading
import logging
from typing import List, Optional

from .manager import VLLMServer


class LoadBalancer:
    """
    Load balancer for distributing requests across vLLM servers.

    Supports multiple strategies:
    - round_robin: Distribute requests sequentially
    - least_connections: Send to server with fewest active requests
    - random: Distribute randomly

    Features:
    - Dynamic server list updates from health check callbacks
    - Automatic failover to healthy servers
    - Configurable fallback behavior when no healthy servers available

    Configuration:
        servers: List of VLLMServer objects
        strategy: Load balancing strategy (default: "round_robin")
        allow_fallback: Allow routing to unhealthy servers if no healthy ones (default: False)
    """

    def __init__(self, servers: List[VLLMServer], strategy: str = 'round_robin', allow_fallback: bool = False):
        """
        Initialize load balancer.

        Args:
            servers: List of VLLMServer objects
            strategy: Load balancing strategy ('round_robin', 'least_connections', 'random')
            allow_fallback: If True, allows routing to unhealthy servers when no healthy ones available
        """
        self.servers = servers
        self.strategy = strategy
        self.allow_fallback = allow_fallback
        self._lock = threading.Lock()
        self._round_robin_index = 0
        self.logger = logging.getLogger(__name__)

        # Validate strategy
        valid_strategies = ['round_robin', 'least_connections', 'random']
        if strategy not in valid_strategies:
            raise ValueError(f"Unknown strategy: {strategy}. Must be one of {valid_strategies}")

    def get_server(self) -> Optional[VLLMServer]:
        """
        Get next server based on load balancing strategy.

        Returns:
            VLLMServer object or None if no servers available
        """
        if not self.servers:
            self.logger.warning("No servers available in load balancer")
            return None

        with self._lock:
            # Get healthy servers
            healthy_servers = [s for s in self.servers if s.healthy]

            if not healthy_servers:
                if self.allow_fallback and self.servers:
                    # No healthy servers but fallback is allowed
                    self.logger.warning(
                        f"No healthy servers available, falling back to unhealthy server "
                        f"(total servers: {len(self.servers)})"
                    )
                    healthy_servers = self.servers
                else:
                    # No healthy servers and fallback not allowed
                    self.logger.error(
                        f"No healthy servers available and fallback disabled "
                        f"(total servers: {len(self.servers)}, healthy: 0)"
                    )
                    return None

            if self.strategy == 'round_robin':
                return self._round_robin(healthy_servers)
            elif self.strategy == 'least_connections':
                return self._least_connections(healthy_servers)
            elif self.strategy == 'random':
                return self._random(healthy_servers)
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

    def _round_robin(self, servers: List[VLLMServer]) -> VLLMServer:
        """Round-robin load balancing."""
        server = servers[self._round_robin_index]
        self._round_robin_index = (self._round_robin_index + 1) % len(servers)
        return server

    def _least_connections(self, servers: List[VLLMServer]) -> VLLMServer:
        """Least connections load balancing."""
        return min(servers, key=lambda s: s.request_count)

    def _random(self, servers: List[VLLMServer]) -> VLLMServer:
        """Random load balancing."""
        return random.choice(servers)

    def update_servers(self, servers: List[VLLMServer]):
        """
        Update the list of servers (e.g., after health check).

        This is called automatically when server health state changes.

        Args:
            servers: New list of VLLMServer objects
        """
        with self._lock:
            old_healthy_count = sum(1 for s in self.servers if s.healthy)
            self.servers = servers
            new_healthy_count = sum(1 for s in self.servers if s.healthy)

            # Reset round-robin index if server count changed significantly
            if old_healthy_count != new_healthy_count:
                self._round_robin_index = 0
                self.logger.info(
                    f"[LoadBalancer] Server list updated: "
                    f"healthy servers changed from {old_healthy_count} to {new_healthy_count}"
                )

    def get_stats(self) -> dict:
        """Get statistics about the servers."""
        with self._lock:
            healthy_count = sum(1 for s in self.servers if s.healthy)
            return {
                'total_servers': len(self.servers),
                'healthy_servers': healthy_count,
                'unhealthy_servers': len(self.servers) - healthy_count,
                'strategy': self.strategy,
                'allow_fallback': self.allow_fallback,
                'total_requests': sum(s.request_count for s in self.servers)
            }
