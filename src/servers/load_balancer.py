"""
Load balancer module for distributing requests across vLLM servers.
"""
import random
import threading
from typing import List, Optional

from .manager import VLLMServer


class LoadBalancer:
    """
    Load balancer for distributing requests across vLLM servers.

    Supports multiple strategies:
    - round_robin: Distribute requests sequentially
    - least_connections: Send to server with fewest active requests
    - random: Distribute randomly

    Configuration:
        servers: List of VLLMServer objects
        strategy: Load balancing strategy (default: "round_robin")
    """

    def __init__(self, servers: List[VLLMServer], strategy: str = 'round_robin'):
        """
        Initialize load balancer.

        Args:
            servers: List of VLLMServer objects
            strategy: Load balancing strategy ('round_robin', 'least_connections', 'random')
        """
        self.servers = servers
        self.strategy = strategy
        self._lock = threading.Lock()
        self._round_robin_index = 0

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
            return None

        with self._lock:
            if self.strategy == 'round_robin':
                return self._round_robin()
            elif self.strategy == 'least_connections':
                return self._least_connections()
            elif self.strategy == 'random':
                return self._random()
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

    def _round_robin(self) -> VLLMServer:
        """Round-robin load balancing."""
        # Filter to healthy servers
        healthy_servers = [s for s in self.servers if s.healthy]
        if not healthy_servers:
            # Fallback to any server if all unhealthy
            healthy_servers = self.servers

        server = healthy_servers[self._round_robin_index]
        self._round_robin_index = (self._round_robin_index + 1) % len(healthy_servers)
        return server

    def _least_connections(self) -> VLLMServer:
        """Least connections load balancing."""
        healthy_servers = [s for s in self.servers if s.healthy]
        if not healthy_servers:
            healthy_servers = self.servers

        return min(healthy_servers, key=lambda s: s.request_count)

    def _random(self) -> VLLMServer:
        """Random load balancing."""
        healthy_servers = [s for s in self.servers if s.healthy]
        if not healthy_servers:
            healthy_servers = self.servers

        return random.choice(healthy_servers)

    def update_servers(self, servers: List[VLLMServer]):
        """
        Update the list of servers (e.g., after health check).

        Args:
            servers: New list of VLLMServer objects
        """
        with self._lock:
            self.servers = servers
            self._round_robin_index = 0

    def get_stats(self) -> dict:
        """Get statistics about the servers."""
        with self._lock:
            return {
                'total_servers': len(self.servers),
                'healthy_servers': sum(1 for s in self.servers if s.healthy),
                'strategy': self.strategy,
                'total_requests': sum(s.request_count for s in self.servers)
            }
