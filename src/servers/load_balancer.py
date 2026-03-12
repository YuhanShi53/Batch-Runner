"""
Load balancer module for distributing requests across vLLM servers.
"""
import math
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
    - adaptive_round_robin: Round-robin with intelligent skipping for congested/underperforming servers
    - load_aware_round_robin: Round-robin with dynamic per-server load balancing
    - random: Distribute randomly

    Features:
    - Dynamic server list updates from health check callbacks
    - Automatic failover to healthy servers
    - Configurable fallback behavior when no healthy servers available
    - Success rate tracking for least_connections and adaptive_round_robin strategies

    Configuration:
        servers: List of VLLMServer objects
        strategy: Load balancing strategy (default: "round_robin")
        allow_fallback: Allow routing to unhealthy servers if no healthy ones (default: False)
        success_rate_threshold: Minimum success rate (0.0-1.0) for least_connections and adaptive_round_robin (default: 0.5)
        success_rate_window: Number of requests needed to trust success rate (default: 10)
        max_active_requests: Maximum active requests before server is considered congested (default: 50)
    """

    def __init__(
        self,
        servers: List[VLLMServer],
        strategy: str = 'round_robin',
        allow_fallback: bool = False,
        success_rate_threshold: float = 0.5,
        success_rate_window: int = 10,
        max_active_requests: int = 50
    ):
        """
        Initialize load balancer.

        Args:
            servers: List of VLLMServer objects
            strategy: Load balancing strategy ('round_robin', 'least_connections', 'adaptive_round_robin', 'random')
            allow_fallback: If True, allows routing to unhealthy servers when no healthy ones available
            success_rate_threshold: Minimum success rate for a server to be preferred
            success_rate_window: Minimum requests before success rate is considered reliable
            max_active_requests: Maximum active requests before server is considered congested (for adaptive_round_robin)
        """
        self.servers = servers
        self.strategy = strategy
        self.allow_fallback = allow_fallback
        self.success_rate_threshold = success_rate_threshold
        self.success_rate_window = success_rate_window
        self.max_active_requests = max_active_requests
        self._lock = threading.Lock()
        self._round_robin_index = 0
        self.logger = logging.getLogger(__name__)

        # Validate strategy
        valid_strategies = [
            'round_robin',
            'least_connections',
            'adaptive_round_robin',
            'load_aware_round_robin',
            'random'
        ]
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
            elif self.strategy == 'adaptive_round_robin':
                return self._adaptive_round_robin(healthy_servers)
            elif self.strategy == 'load_aware_round_robin':
                return self._load_aware_round_robin(healthy_servers)
            elif self.strategy == 'random':
                return self._random(healthy_servers)
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

    def _round_robin(self, servers: List[VLLMServer]) -> VLLMServer:
        """Round-robin load balancing."""
        index = self._round_robin_index % len(servers)
        server = servers[index]
        self._round_robin_index = (index + 1) % len(servers)
        return server

    def _is_underperforming(self, server: VLLMServer, avg_success_rate: Optional[float]) -> bool:
        """Return True when a server has enough history and is clearly underperforming."""
        if avg_success_rate is None:
            return False

        total_requests = server.success_count + server.error_count
        if total_requests < self.success_rate_window:
            return False

        return (
            server.success_rate < self.success_rate_threshold
            and server.success_rate < avg_success_rate * 0.8
        )

    def _get_average_success_rate(self, servers: List[VLLMServer]) -> Optional[float]:
        """Return average success rate for servers with enough samples, if any."""
        reliable_servers = [
            s for s in servers
            if (s.success_count + s.error_count) >= self.success_rate_window
        ]

        if not reliable_servers:
            return None

        return sum(s.success_rate for s in reliable_servers) / len(reliable_servers)

    def _filter_underperforming_servers(self, servers: List[VLLMServer]) -> List[VLLMServer]:
        """Prefer healthy servers that are not clearly underperforming."""
        if len(servers) <= 1:
            return servers

        avg_success_rate = self._get_average_success_rate(servers)
        if avg_success_rate is None:
            return servers

        acceptable = [
            s for s in servers
            if not self._is_underperforming(s, avg_success_rate)
        ]
        return acceptable if acceptable else servers

    def _select_first_in_round_robin_order(
        self,
        servers: List[VLLMServer],
        candidates: List[VLLMServer]
    ) -> Optional[VLLMServer]:
        """Pick the first candidate encountered in the current round-robin order."""
        if not candidates:
            return None

        candidate_ids = {id(server) for server in candidates}
        start_index = self._round_robin_index % len(servers)

        for offset in range(len(servers)):
            server = servers[(start_index + offset) % len(servers)]
            if id(server) in candidate_ids:
                return server

        return candidates[0]

    def _select_min_with_round_robin_tie_break(self, servers: List[VLLMServer], score_fn) -> Optional[VLLMServer]:
        """Select the minimum-scored server and break ties using round-robin order."""
        if not servers:
            return None

        best_score = min(score_fn(server) for server in servers)
        candidates = [server for server in servers if score_fn(server) == best_score]
        return self._select_first_in_round_robin_order(servers, candidates)

    def _advance_round_robin_index(self, servers: List[VLLMServer], selected: VLLMServer) -> None:
        """Advance round-robin index to the slot after the selected server."""
        for i, server in enumerate(servers):
            if server == selected:
                self._round_robin_index = (i + 1) % len(servers)
                return

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

        candidates = self._filter_underperforming_servers(servers)

        # Select server with minimum effective load
        selected = self._select_min_with_round_robin_tie_break(
            candidates,
            lambda s: (s.effective_load, s.active_requests)
        )

        # Log when we're avoiding a server due to poor performance
        if len(candidates) < len(servers):
            avoided = [s for s in servers if s not in candidates]
            self.logger.debug(
                f"[LoadBalancer] Avoiding {len(avoided)} underperforming server(s) due to low success rate. "
                f"Selected {selected.name} with {selected.active_requests} active requests, "
                f"success rate: {selected.success_rate:.1%}"
            )

        return selected

    def _adaptive_round_robin(self, servers: List[VLLMServer]) -> VLLMServer:
        """
        Adaptive round-robin load balancing with intelligent server skipping.

        This strategy extends round-robin by skipping servers that are:
        1. Congested: active_requests >= max_active_requests (indicating blocked/queued requests)
        2. Underperforming: low success rate below threshold AND below average

        Selection logic:
        - Start from current round-robin index
        - Check each server sequentially
        - Skip servers that are congested or severely underperforming
        - Select the first server that meets criteria
        - If no server meets criteria, select the one with lowest active_requests

        This provides round-robin fairness while avoiding congested servers
        that are likely to cause delays or timeouts.
        """
        if not servers:
            return None

        n = len(servers)
        original_index = self._round_robin_index % n
        avg_success_rate = self._get_average_success_rate(servers)

        # Track candidates and reasons for skipping
        skipped_congested = []
        skipped_low_success = []
        candidates = []

        # Check each server in round-robin order
        for offset in range(n):
            index = (original_index + offset) % n
            server = servers[index]

            # Check if server is congested (too many active requests)
            if server.active_requests >= self.max_active_requests:
                skipped_congested.append(server)
                continue

            if self._is_underperforming(server, avg_success_rate):
                skipped_low_success.append(server)
                continue

            # Server passes all checks
            candidates.append(server)

        # Select best candidate
        if candidates:
            # Prefer the earliest candidate (maintains round-robin order)
            selected = candidates[0]
        else:
            # All servers are either congested or underperforming
            # Fall back to selecting the one with lowest active_requests
            selected = min(servers, key=lambda s: s.active_requests)
            self.logger.warning(
                f"[LoadBalancer] All servers congested or underperforming. "
                f"Selected {selected.name} with {selected.active_requests} active requests as fallback"
            )

        self._advance_round_robin_index(servers, selected)

        # Log skipped servers for visibility
        if skipped_congested:
            self.logger.debug(
                f"[LoadBalancer] Skipped {len(skipped_congested)} congested server(s) "
                f"(active_requests >= {self.max_active_requests})"
            )
        if skipped_low_success:
            self.logger.debug(
                f"[LoadBalancer] Skipped {len(skipped_low_success)} low success rate server(s) "
                f"(success_rate < {self.success_rate_threshold:.2%})"
            )

        return selected

    def _load_aware_round_robin(self, servers: List[VLLMServer]) -> VLLMServer:
        """
        Round-robin with dynamic load awareness for large server pools.

        This keeps round-robin fairness when loads are similar, but skips servers
        that are already materially above the cluster's current fair-share load.
        That makes it a better fit than plain round-robin when concurrency is high
        and server latency is uneven.
        """
        if not servers:
            return None

        n = len(servers)
        original_index = self._round_robin_index % n
        avg_success_rate = self._get_average_success_rate(servers)
        total_active = sum(server.active_requests for server in servers)
        min_active = min(server.active_requests for server in servers)

        # Allow a server to stay within one request of the cluster's fair share.
        fair_share_limit = max(min_active + 1, math.ceil((total_active + 1) / n))
        hard_limit = self.max_active_requests

        skipped_overloaded = []
        skipped_low_success = []
        candidates = []

        for offset in range(n):
            index = (original_index + offset) % n
            server = servers[index]

            if hard_limit > 0 and server.active_requests >= hard_limit:
                skipped_overloaded.append(server)
                continue

            if server.active_requests > fair_share_limit:
                skipped_overloaded.append(server)
                continue

            if self._is_underperforming(server, avg_success_rate):
                skipped_low_success.append(server)
                continue

            candidates.append(server)

        if candidates:
            selected = candidates[0]
        else:
            fallback_servers = self._filter_underperforming_servers(servers)
            selected = self._select_min_with_round_robin_tie_break(
                fallback_servers,
                lambda s: (s.effective_load, s.active_requests)
            )
            self.logger.warning(
                f"[LoadBalancer] No server met dynamic load target (fair_share_limit={fair_share_limit}). "
                f"Falling back to {selected.name} with {selected.active_requests} active requests"
            )

        self._advance_round_robin_index(servers, selected)

        if skipped_overloaded:
            self.logger.debug(
                f"[LoadBalancer] Skipped {len(skipped_overloaded)} overloaded server(s) "
                f"(dynamic fair_share_limit={fair_share_limit}, hard_limit={hard_limit})"
            )
        if skipped_low_success:
            self.logger.debug(
                f"[LoadBalancer] Skipped {len(skipped_low_success)} low success rate server(s) "
                f"(success_rate < {self.success_rate_threshold:.2%})"
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
                'max_active_requests': self.max_active_requests,
                'total_requests': sum(s.request_count for s in self.servers),
                'servers': server_stats
            }
