"""
Tests for load balancer functionality.

Run with: python -m pytest tests/test_load_balancer.py -v
"""
import pytest
from src.servers.load_balancer import LoadBalancer
from src.servers.manager import VLLMServer


class TestLoadBalancer:
    """Test LoadBalancer functionality."""

    def test_round_robin_strategy(self):
        """Test basic round-robin distribution."""
        servers = [
            VLLMServer(name="server_1", ip="127.0.0.1", port=8000),
            VLLMServer(name="server_2", ip="127.0.0.1", port=8001),
            VLLMServer(name="server_3", ip="127.0.0.1", port=8002),
        ]

        lb = LoadBalancer(servers, strategy='round_robin')

        # Should cycle through servers
        assert lb.get_server().name == "server_1"
        assert lb.get_server().name == "server_2"
        assert lb.get_server().name == "server_3"
        assert lb.get_server().name == "server_1"  # Back to start

    def test_least_connections_strategy(self):
        """Test least_connections strategy."""
        servers = [
            VLLMServer(name="server_1", ip="127.0.0.1", port=8000),
            VLLMServer(name="server_2", ip="127.0.0.1", port=8001),
            VLLMServer(name="server_3", ip="127.0.0.1", port=8002),
        ]

        lb = LoadBalancer(servers, strategy='least_connections')

        # Simulate different loads
        servers[0].increment_active()  # server_1: 1 active
        servers[0].increment_active()  # server_1: 2 active
        servers[1].increment_active()  # server_2: 1 active
        # server_3: 0 active

        # Should select server with least active requests
        selected = lb.get_server()
        assert selected.name == "server_3"

    def test_adaptive_round_robin_skips_congested(self):
        """Test adaptive_round_robin skips congested servers."""
        servers = [
            VLLMServer(name="server_1", ip="127.0.0.1", port=8000),
            VLLMServer(name="server_2", ip="127.0.0.1", port=8001),
            VLLMServer(name="server_3", ip="127.0.0.1", port=8002),
        ]

        lb = LoadBalancer(
            servers,
            strategy='adaptive_round_robin',
            max_active_requests=10  # Low threshold for testing
        )

        # Make server_1 congested
        for _ in range(15):
            servers[0].increment_active()

        # Should skip server_1 and select server_2
        selected = lb.get_server()
        assert selected.name == "server_2"
        assert servers[0].active_requests == 15  # Unchanged

    def test_adaptive_round_robin_all_congested_fallback(self):
        """Test adaptive_round_robin fallback when all servers are congested."""
        servers = [
            VLLMServer(name="server_1", ip="127.0.0.1", port=8000),
            VLLMServer(name="server_2", ip="127.0.0.1", port=8001),
        ]

        lb = LoadBalancer(
            servers,
            strategy='adaptive_round_robin',
            max_active_requests=10
        )

        # Make both servers congested
        for _ in range(20):
            servers[0].increment_active()
        for _ in range(15):
            servers[1].increment_active()

        # Should select the one with lowest active requests (server_2)
        selected = lb.get_server()
        assert selected.name == "server_2"

    def test_adaptive_round_robin_skips_low_success_rate(self):
        """Test adaptive_round_robin skips servers with low success rate."""
        servers = [
            VLLMServer(name="server_1", ip="127.0.0.1", port=8000),
            VLLMServer(name="server_2", ip="127.0.0.1", port=8001),
            VLLMServer(name="server_3", ip="127.0.0.1", port=8002),
        ]

        lb = LoadBalancer(
            servers,
            strategy='adaptive_round_robin',
            success_rate_threshold=0.7,
            success_rate_window=5
        )

        # Make server_1 have low success rate
        for _ in range(10):
            servers[0].record_error()
        for _ in range(2):
            servers[0].record_success()

        # server_1: 2 success, 10 errors = 16.7% success rate
        assert servers[0].success_rate < 0.2

        # Give server_2 and server_3 good success rates
        for _ in range(10):
            servers[1].record_success()
        for _ in range(10):
            servers[2].record_success()

        # Should skip server_1 and select server_2
        selected = lb.get_server()
        assert selected.name == "server_2"

    def test_adaptive_round_robin_maintains_fairness(self):
        """Test that adaptive_round_robin maintains round-robin fairness when no skipping."""
        servers = [
            VLLMServer(name="server_1", ip="127.0.0.1", port=8000),
            VLLMServer(name="server_2", ip="127.0.0.1", port=8001),
            VLLMServer(name="server_3", ip="127.0.0.1", port=8002),
        ]

        lb = LoadBalancer(
            servers,
            strategy='adaptive_round_robin',
            max_active_requests=50
        )

        # No congestion, should behave like regular round-robin
        selections = [lb.get_server().name for _ in range(6)]
        assert selections == ["server_1", "server_2", "server_3", "server_1", "server_2", "server_3"]

    def test_load_aware_round_robin_maintains_fairness(self):
        """Test that load_aware_round_robin behaves like round-robin when loads are even."""
        servers = [
            VLLMServer(name="server_1", ip="127.0.0.1", port=8000),
            VLLMServer(name="server_2", ip="127.0.0.1", port=8001),
            VLLMServer(name="server_3", ip="127.0.0.1", port=8002),
        ]

        lb = LoadBalancer(servers, strategy='load_aware_round_robin')

        selections = [lb.get_server().name for _ in range(6)]
        assert selections == ["server_1", "server_2", "server_3", "server_1", "server_2", "server_3"]

    def test_load_aware_round_robin_skips_overloaded_server(self):
        """Test that load_aware_round_robin skips servers far above the fair-share load."""
        servers = [
            VLLMServer(name=f"server_{i}", ip="127.0.0.1", port=8000 + i)
            for i in range(200)
        ]

        for index, server in enumerate(servers):
            server.active_requests = 20
            if index == 0:
                server.active_requests = 80

        lb = LoadBalancer(servers, strategy='load_aware_round_robin', max_active_requests=512)

        selected = lb.get_server()
        assert selected.name == "server_1"

    def test_load_aware_round_robin_skips_low_success_rate(self):
        """Test that load_aware_round_robin avoids servers with sustained poor success rate."""
        servers = [
            VLLMServer(name="server_1", ip="127.0.0.1", port=8000),
            VLLMServer(name="server_2", ip="127.0.0.1", port=8001),
            VLLMServer(name="server_3", ip="127.0.0.1", port=8002),
        ]

        for _ in range(10):
            servers[0].record_error()
            servers[1].record_success()
            servers[2].record_success()

        lb = LoadBalancer(
            servers,
            strategy='load_aware_round_robin',
            success_rate_threshold=0.7,
            success_rate_window=5
        )

        selected = lb.get_server()
        assert selected.name == "server_2"

    def test_random_strategy(self):
        """Test random strategy."""
        servers = [
            VLLMServer(name="server_1", ip="127.0.0.1", port=8000),
            VLLMServer(name="server_2", ip="127.0.0.1", port=8001),
        ]

        lb = LoadBalancer(servers, strategy='random')

        # Just verify it returns a valid server
        for _ in range(10):
            server = lb.get_server()
            assert server.name in ["server_1", "server_2"]

    def test_invalid_strategy(self):
        """Test that invalid strategy raises error."""
        servers = [
            VLLMServer(name="server_1", ip="127.0.0.1", port=8000),
        ]

        with pytest.raises(ValueError, match="Unknown strategy"):
            LoadBalancer(servers, strategy='invalid_strategy')

    def test_unhealthy_server_filtered(self):
        """Test that unhealthy servers are filtered out."""
        servers = [
            VLLMServer(name="server_1", ip="127.0.0.1", port=8000),
            VLLMServer(name="server_2", ip="127.0.0.1", port=8001, healthy=False),
            VLLMServer(name="server_3", ip="127.0.0.1", port=8002),
        ]

        lb = LoadBalancer(servers, strategy='round_robin', allow_fallback=False)

        # Should only select from healthy servers
        for _ in range(5):
            server = lb.get_server()
            assert server.name != "server_2"
            assert server.healthy is True

    def test_fallback_to_unhealthy(self):
        """Test fallback to unhealthy servers when no healthy ones available."""
        servers = [
            VLLMServer(name="server_1", ip="127.0.0.1", port=8000, healthy=False),
            VLLMServer(name="server_2", ip="127.0.0.1", port=8001, healthy=False),
        ]

        lb = LoadBalancer(servers, strategy='round_robin', allow_fallback=True)

        # Should fallback to unhealthy server
        server = lb.get_server()
        assert server is not None
        assert server.healthy is False

    def test_no_servers_returns_none(self):
        """Test that None is returned when no servers available."""
        lb = LoadBalancer([], strategy='round_robin')
        assert lb.get_server() is None

    def test_update_servers(self):
        """Test updating server list."""
        servers = [
            VLLMServer(name="server_1", ip="127.0.0.1", port=8000),
        ]

        lb = LoadBalancer(servers, strategy='round_robin')

        # Update with new servers
        new_servers = [
            VLLMServer(name="server_1", ip="127.0.0.1", port=8000),
            VLLMServer(name="server_2", ip="127.0.0.1", port=8001),
        ]

        lb.update_servers(new_servers)

        # Should now select from both servers
        selections = set()
        for _ in range(10):
            selections.add(lb.get_server().name)

        assert "server_1" in selections
        assert "server_2" in selections

    def test_get_stats(self):
        """Test getting load balancer statistics."""
        servers = [
            VLLMServer(name="server_1", ip="127.0.0.1", port=8000),
            VLLMServer(name="server_2", ip="127.0.0.1", port=8001, healthy=False),
        ]

        # Add some activity
        servers[0].increment_active()
        servers[0].record_success()
        servers[0].record_error()

        lb = LoadBalancer(
            servers,
            strategy='adaptive_round_robin',
            max_active_requests=50,
            success_rate_threshold=0.5
        )

        stats = lb.get_stats()

        assert stats['total_servers'] == 2
        assert stats['healthy_servers'] == 1
        assert stats['unhealthy_servers'] == 1
        assert stats['strategy'] == 'adaptive_round_robin'
        assert stats['max_active_requests'] == 50
        assert len(stats['servers']) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
