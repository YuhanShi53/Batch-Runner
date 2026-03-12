# High Concurrency Performance Optimization

## Problem Analysis

With 4096 concurrent requests across 160 vLLM servers, the original implementation had several critical bottlenecks:

### 1. Event Loop Creation Overhead
**Issue**: `asyncio.run()` was called for every request, creating a new event loop each time.
```python
# OLD CODE - SLOW
response = asyncio.run(_send_http_request_async(...))  # New loop per request!
```

**Impact**: Creating/destroying event loops is expensive and causes significant CPU overhead at high concurrency.

### 2. HTTP Client Not Reused
**Issue**: A new `httpx.AsyncClient` was created for each request.
```python
# OLD CODE - SLOW
async with httpx.AsyncClient(...) as client:  # New client per request!
```

**Impact**: Connection pooling was ineffective. Each request required:
- New TCP connection establishment
- TLS handshake (if HTTPS)
- No HTTP/2 connection reuse

### 3. Fixed Connection Pool Limits
**Issue**: Hardcoded limits were suboptimal for high concurrency.
```python
limits = httpx.Limits(
    max_connections=10000,      # Too large, wastes resources
    max_keepalive_connections=1000
)
```

## Solution

### Thread-Local Async Context

Implemented `AsyncClientContext` that maintains:
- A persistent event loop per worker thread
- A reusable `httpx.AsyncClient` with proper connection pooling
- Thread-safe initialization and cleanup

```python
class AsyncClientContext:
    """Manages persistent event loop and HTTP client per thread."""

    def _get_or_create_loop(self):
        """Get or create event loop for this thread."""
        if self._loop is None or self._loop.is_closed():
            with self._lock:
                if self._loop is None or self._loop.is_closed():
                    self._loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(self._loop)
        return self._loop

    def _get_or_create_client(self):
        """Get or create httpx client for this thread."""
        if self._client is None or self._client.is_closed:
            with self._lock:
                if self._client is None or self._client.is_closed:
                    limits = httpx.Limits(
                        max_connections=self.max_connections,
                        max_keepalive_connections=self.max_keepalive_connections
                    )
                    self._client = httpx.AsyncClient(
                        limits=limits,
                        timeout=self.timeout,
                        http2=self.http2
                    )
        return self._client
```

### Request Processing Flow

```python
# NEW CODE - FAST
def _send_request_with_retry(self, server, payload: dict):
    # Get the persistent async client context for this thread
    context = self._get_async_client_context()

    # Send request using the persistent client
    response = context.run_coroutine(
        _send_http_request_with_client(
            client=context._get_or_create_client(),
            url=url,
            payload=payload,
            max_retries=self.config.max_retries,
            base_delay=self.config.retry_delay
        )
    )
    return response
```

## Configuration Recommendations

### For 4096 Concurrency with 160 Servers

```yaml
runner:
  max_concurrency: 4096
  streaming: true
  stream_queue_size: 2048  # Larger queue for better throughput

  # HTTP client settings
  http_max_connections: 256  # Per-thread connections
  http_max_keepalive_connections: 50  # ~20% keepalive

  # Load balancing
  load_balancing_strategy: load_aware_round_robin  # Best when many servers have uneven response times
  max_active_requests: 512  # Per-server limit

  # Timeout and retries
  request_timeout: 300  # May need longer with queuing
  max_retries: 1  # Reduce to avoid cascading failures
  retry_delay: 0.5
```

### Parameter Tuning Guide

#### `http_max_connections`
- **Purpose**: Maximum concurrent connections per worker thread
- **Formula**: Each thread handles 1 request at a time, so `1-2` is sufficient
- **Example**: With `max_concurrency: 4096`, you have 4096 threads, each needs 1-2 connections
- **Too low**: May cause connection starvation if retries are needed
- **Too high**: Wastes memory and file descriptors (4096 threads × high limit = system FD exhaustion)

#### `http_max_keepalive_connections`
- **Purpose**: Maximum idle connections to keep open per thread
- **Formula**: `1` is sufficient for single-request-per-thread workloads
- **Example**: 1 keepalive per thread = 4096 total keepalive connections
- **Benefit**: Reuses TCP connections to the same server, avoids TLS handshakes

#### `max_active_requests`
- **Purpose**: Per-server concurrent request limit
- **Formula**: `max_concurrency / num_servers * safety_factor`
- **Example**: 4096 / 160 * 2 = 51.2 → 512 (generous headroom)
- **Too low**: Load balancer will skip healthy servers
- **Too high**: No effective load balancing

## Performance Monitoring

Added timing instrumentation:

```python
# Track slow requests
elapsed = time.time() - start_time
if elapsed > 5.0:
    self.logger.warning(
        f"Request to {server.name} took {elapsed:.2f}s "
        f"(slow, may indicate server congestion)"
    )
```

## Expected Improvements

### Before Optimization
- Event loop creation per request: ~5-10ms overhead
- New TCP connection per request: ~50-100ms overhead
- TLS handshake per request: ~100-200ms overhead
- **Total overhead per request**: ~155-310ms

### After Optimization
- Persistent event loop: 0ms overhead
- Connection reuse: ~0-1ms overhead (if cached)
- **Total overhead per request**: ~0-1ms

### Throughput Improvement
- **Theoretical speedup**: 10-100x reduction in overhead
- **Expected throughput**: Should now be limited by vLLM server capacity, not client

## Verification

Monitor these metrics to verify the optimization:

1. **GPU Utilization**: Should approach 100% on all servers
2. **Throughput (req/s)**: Should match vLLM server capacity
3. **Request Latency**: Should be consistent (not increasing with concurrency)
4. **Connection Reuse**: Check logs for "Created new AsyncClientContext" messages
   - Should see one message per worker thread at startup
   - NOT one message per request

## Troubleshooting

### Still seeing low GPU utilization?

1. **Check vLLM server logs**:
   ```bash
   # On each server
   tail -f /path/to/vllm.log
   ```
   Look for:
   - Request queue depth
   - Time spent processing each request
   - Any errors or timeouts

2. **Check network**:
   ```bash
   # Test latency from client to servers
   for server in $(cat server_list.txt); do
       ping -c 10 $server
   done
   ```

3. **Check client resources**:
   ```bash
   # Monitor client CPU/memory
   top -p $(pgrep -f "python -m src.cli")
   ```

4. **Enable debug logging**:
   ```bash
   python -m src.cli --config configs/high_concurrency_config.yaml --log-level DEBUG
   ```

### Seeing too many connection errors?

Reduce `http_max_connections` per thread:
```yaml
runner:
  http_max_connections: 128  # Try half the previous value
```

### Seeing timeouts?

Increase `request_timeout`:
```yaml
runner:
  request_timeout: 600  # 10 minutes
```

Or reduce concurrency per server by increasing `max_active_requests` threshold.
