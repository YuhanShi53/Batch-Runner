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
    - least_connections: Send to server with fewest active requests (success rate aware)
    - random: Distribute randomly

    Features:
    - Dynamic server list updates from health check callbacks
    - Automatic failover to healthy servers
    - Configurable fallback behavior when no healthy servers available
    - Success rate tracking for least_connections strategy

    Configuration:
        servers: List of VLLMServer objects
        strategy: Load balancing strategy (default: "round_robin")
        allow_fallback: Allow routing to unhealthy servers if no healthy ones (default: False)
        success_rate_threshold: Minimum success rate (0.0-1.0) for least_connections (default: 0.5)
        success_rate_window: Number of requests needed to trust success rate (default: 10)
    """

    def __init__(
        self,
        servers: List[VLLMServer],
        strategy: str = 'round_robin',
        allow_fallback: bool = False,
        success_rate_threshold: float = 0.5,
        success_rate_window: int = 10
    ):
        """
        Initialize load balancer.

        Args:
            servers: List of VLLMServer objects
            strategy: Load balancing strategy ('round_robin', 'least_connections', 'random')
            allow_fallback: If True, allows routing to unhealthy servers when no healthy ones available
            success_rate_threshold: Minimum success rate for a server to be preferred in least_connections
            success_rate_window: Minimum requests before success rate is considered reliable
        """
        self.servers = servers
        self.strategy = strategy
        self.allow_fallback = allow_fallback
        self.success_rate_threshold = success_rate_threshold
        self.success_rate_window = success_rate_window
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
        """
        Least connections load balancing with success rate awareness.

        This enhanced strategy:
        1. Prioritizes servers with lower effective load (active requests adjusted by success rate)
        2. Avoids routing to servers with low success rates when better alternatives exist
        3. Temporarily deprioritizes servers that are accumulating errors

        Effective load calculation:
        - Base load = active_requests
        - Penalty factor = 1 / success_rate (capped at 10x)
        - Effective load = base_load * penalty_factor

        Example: A server with 5 active requests and 50% success rate has
        effective load = 5 * 2 = 10, making it less preferable than a
        server with 8 active requests but 100% success rate.
        """
        if not servers:
            return None

        # Filter out servers with severely degraded performance if alternatives exist
        # Only apply this filter when we have multiple servers and enough data
        if len(servers) > 1:
            reliable_servers = [
                s for s in servers
                if (s.success_count + s.error_count) >= self.success_rate_window
            ]

            if reliable_servers:
                # Calculate average success rate among reliable servers
                avg_success_rate = sum(s.success_rate for s in reliable_servers) / len(reliable_servers)

                # Identify underperforming servers (below threshold AND below average)
                underperforming = []
                acceptable = []

                for s in servers:
                    total_requests = s.success_count + s.error_count
                    if total_requests >= self.success_rate_window:
                        # We have reliable data for this server
                        if s.success_rate < self.success_rate_threshold and s.success_rate < avg_success_rate * 0.8:
                            underperforming.append(s)
                        else:
                            acceptable.append(s)
                    else:
                        # Not enough data yet, treat as acceptable
                        acceptable.append(s)

                # If we have acceptable servers, prefer them
                candidates = acceptable if acceptable else servers
            else:
                # Not enough reliable data yet, use all servers
                candidates = servers
        else:
            candidates = servers

        # Select server with minimum effective load
        selected = min(candidates, key=lambda s: s.effective_load)

        # Log when we're avoiding a server due to poor performance
        if len(candidates) < len(servers):
            avoided = [s for s in servers if s not in candidates]
            self.logger.debug(
                f"[LoadBalancer] Avoiding {len(avoided)} underperforming server(s) due to low success rate. "
                f"Selected {selected.name} with {selected.active_requests} active requests, "
                f"success rate: {selected.success_rate:.1%}"
            )

        return selected

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
            server_stats = []
            for s in self.servers:
                server_stats.append({
                    'name': s.name,
                    'healthy': s.healthy,
                    'active_requests': s.active_requests,
                    'success_count': s.success_count,
                    'error_count': s.error_count,
                    'success_rate': f"{s.success_rate:.2%}",
                    'effective_load': s.effective_load
                })

            return {
                'total_servers': len(self.servers),
                'healthy_servers': healthy_count,
                'unhealthy_servers': len(self.servers) - healthy_count,
                'strategy': self.strategy,
                'allow_fallback': self.allow_fallback,
                'success_rate_threshold': f"{self.success_rate_threshold:.2%}",
                'success_rate_window': self.success_rate_window,
                'total_requests': sum(s.request_count for s in self.servers),
                'servers': server_stats
            }
