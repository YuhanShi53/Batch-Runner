"""
Batch runner module - main orchestrator for batch inference.

Coordinates data loading, server management, request processing, and result saving.
"""
import time
import asyncio
from typing import Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import logging

try:
    import httpx
except ImportError:
    httpx = None

from .loaders.base import DataLoader, LoadResult
from .savers.base import ResultSaver, SaveResult
from .servers.manager import VLLMServerManager
from .servers.load_balancer import LoadBalancer
from .utils.progress import ProgressTracker
from .utils.retry import retry_with_backoff
from .adapters.base import ModelAdapter


@dataclass
class BatchConfig:
    """Configuration for batch inference."""
    # Concurrency settings
    max_concurrency: int = 10
    max_retries: int = 3
    retry_delay: float = 1.0
    request_timeout: int = 120

    # HTTP client settings
    http_max_connections: int = 4096  # Maximum concurrent connections (shared)
    http_max_keepalive_connections: int = 1000  # Maximum keepalive connections (shared)
    http2: bool = True  # Enable HTTP/2

    # Model settings
    model_name: str = "default"
    temperature: float = 0.7
    max_tokens: int = 1000
    system_prompt: str = ""

    # Server settings
    servers_dir: str = "."
    load_balancing_strategy: str = "round_robin"
    health_check_interval: int = 30
    max_failures: int = 5
    allow_unhealthy_fallback: bool = False
    success_rate_threshold: float = 0.5
    success_rate_window: int = 10
    max_active_requests: int = 50

    # Progress settings
    progress_report_interval: int = 10

    # Optional sampling parameters
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    # Adapter settings
    adapter_class: str = "OpenAIAdapter"
    adapter: ModelAdapter = None

    # Streaming settings
    streaming: bool = True
    stream_queue_size: int = 100  # Max items in buffer queue for backpressure

    # Resume settings
    resume: bool = True  # Enable resuming from existing output

    def get(self, key: str, default=None):
        """Get config value with default fallback."""
        return getattr(self, key, default)


@dataclass
class BatchStats:
    """Statistics for batch processing."""
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    retried_requests: int = 0
    total_tokens: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    # Remove lock for better performance - accept small race conditions
    # For monitoring purposes, 100% accuracy is not critical

    @property
    def duration(self) -> float:
        """Get elapsed duration in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    @property
    def success_rate(self) -> float:
        """Get success rate (0.0 to 1.0)."""
        if self.total_requests == 0:
            return 0.0
        return self.completed_requests / self.total_requests

    async def increment_completed(self):
        """Increment completed requests count."""
        self.completed_requests += 1

    async def increment_failed(self):
        """Increment failed requests count."""
        self.failed_requests += 1

    async def increment_retried(self):
        """Increment retried requests count."""
        self.retried_requests += 1

    async def add_tokens(self, count: int):
        """Add to total tokens count."""
        self.total_tokens += count

    # Keep sync versions for compatibility
    def increment_completed_sync(self):
        """Increment completed requests count (sync version)."""
        self.completed_requests += 1

    def increment_failed_sync(self):
        """Increment failed requests count (sync version)."""
        self.failed_requests += 1

    def increment_retried_sync(self):
        """Increment retried requests count (sync version)."""
        self.retried_requests += 1

    def add_tokens_sync(self, count: int):
        """Add to total tokens count (sync version)."""
        self.total_tokens += count


class BatchRunner:
    """
    Main batch inference runner with async support.

    Orchestrates:
    - Data loading
    - Server management and load balancing
    - Concurrent request processing (async)
    - Result saving
    - Progress tracking and reporting

    Attributes:
        config: BatchConfig instance
        loader: DataLoader instance
        saver: ResultSaver instance
    """

    def __init__(self, config: BatchConfig, loader: DataLoader, saver: ResultSaver):
        """
        Initialize batch runner.

        Args:
            config: BatchConfig instance
            loader: DataLoader instance
            saver: ResultSaver instance
        """
        self.config = config
        self.loader = loader
        self.saver = saver

        self.logger = logging.getLogger(__name__)
        self.stats = BatchStats()

        # Shared async HTTP client
        self._http_client = None

        # Thread pool for blocking I/O operations (saver, logging)
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="saver_")

        # Initialize server manager and load balancer
        server_config = {
            'servers_dir': config.servers_dir,
            'request_timeout': config.request_timeout,
            'health_check_interval': config.get('health_check_interval', 30),
            'max_failures': config.get('max_failures', 5)
        }
        self.server_manager = VLLMServerManager(server_config)

        # Register callback to update load balancer when server health changes
        def on_server_state_change(_server, _is_healthy):
            """Callback when server health state changes."""
            self.load_balancer.update_servers(self.server_manager.get_all_servers())

        self.server_manager.register_state_change_callback(on_server_state_change)

        # Initialize load balancer with current server list
        allow_fallback = config.get('allow_unhealthy_fallback', False)
        success_rate_threshold = config.get('success_rate_threshold', 0.5)
        success_rate_window = config.get('success_rate_window', 10)
        max_active_requests = config.get('max_active_requests', 50)
        self.load_balancer = LoadBalancer(
            self.server_manager.get_all_servers(),
            strategy=config.load_balancing_strategy,
            allow_fallback=allow_fallback,
            success_rate_threshold=success_rate_threshold,
            success_rate_window=success_rate_window,
            max_active_requests=max_active_requests
        )

        # Progress tracking
        self.progress_tracker = ProgressTracker(
            total_items=self._estimate_total_items(),
            report_interval=config.progress_report_interval,
            stats=self.stats
        )

    async def _get_http_client(self):
        """
        Get or create shared async HTTP client.

        Returns:
            httpx.AsyncClient instance
        """
        if self._http_client is None or self._http_client.is_closed:
            limits = httpx.Limits(
                max_connections=self.config.http_max_connections,
                max_keepalive_connections=self.config.http_max_keepalive_connections
            )
            self._http_client = httpx.AsyncClient(
                limits=limits,
                timeout=self.config.request_timeout,
                http2=self.config.http2
            )
            self.logger.info(
                f"Created shared AsyncClient (max_connections={self.config.http_max_connections}, "
                f"max_keepalive={self.config.http_max_keepalive_connections})"
            )
        return self._http_client

    async def close(self):
        """Clean up resources."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
            self.logger.info("Closed shared HTTP client")

        # Shutdown thread pool
        self._executor.shutdown(wait=True)
        self.logger.info("Shutdown saver thread pool")

    def _estimate_total_items(self) -> int:
        """Estimate total number of requests to process."""
        try:
            return len(self.loader)
        except NotImplementedError:
            return 0

    def run(self):
        """
        Execute the batch inference process (sync wrapper for async).

        This method is kept for backward compatibility.
        """
        asyncio.run(self.run_async())

    async def run_async(self):
        """Execute the batch inference process with async."""
        self.logger.info(f"Starting batch inference with {self.config.max_concurrency} concurrent requests")
        self.logger.info(f"Healthy servers: {self.server_manager.get_server_count()}")

        streaming_mode = self.config.get('streaming', True)
        try:
            if streaming_mode:
                self.logger.info("Running in STREAMING mode (async pipeline processing)")
                await self._run_streaming_async()
            else:
                self.logger.info("Running in BATCH mode (async processing)")
                await self._run_batch_async()
        finally:
            await self.close()
            self._finalize_stats()

    async def _run_batch_async(self):
        """Batch mode: load all data first, then process with async."""
        # Load all data into list
        all_items = []
        for item in self.loader:
            all_items.append(item)

        total_items = len(all_items)
        self.logger.info(f"Loaded {total_items} items")

        # Filter out completed requests if resume is enabled
        if self.config.resume:
            original_count = len(all_items)
            all_items = [
                item for item in all_items
                if not self.saver.is_completed(item.request_id)
            ]
            skipped_count = original_count - len(all_items)
            if skipped_count > 0:
                self.logger.info(f"Resume: skipping {skipped_count} already completed requests")
                self.logger.info(f"Resume: {len(all_items)} requests remaining to process")

        total_requests = len(all_items)
        self.stats.total_requests = total_requests
        self.logger.info(f"Total requests to process: {total_requests}")

        # Get shared client
        client = await self._get_http_client()

        # Process all requests concurrently
        tasks = [
            self._process_request_async(req, client)
            for req in all_items
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Finalize
        await self._finalize_batch_async(total_requests)

    async def _run_streaming_async(self):
        """Streaming mode: producer-consumer pipeline with bounded queue."""
        queue_size = self.config.get('stream_queue_size', 100)
        request_queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)

        producer_exception = []
        total_requests = [0]

        async def producer():
            """Producer coroutine: stream data from loader into queue."""
            try:
                skipped_count = 0
                for item in self.loader:
                    # Skip if already completed (resume mode)
                    if self.config.resume and self.saver.is_completed(item.request_id):
                        skipped_count += 1
                        if skipped_count == 1:
                            self.logger.info("Resume mode: skipping already completed requests")
                        continue

                    # Block if queue is full (backpressure)
                    await request_queue.put(item)
                    total_requests[0] += 1

                # Log skip statistics at end
                if skipped_count > 0:
                    self.logger.info(f"Resume: skipped {skipped_count} already completed requests")

                self.logger.info(f"Producer finished: {total_requests[0]} requests queued")
            except Exception as e:
                self.logger.error(f"Producer error: {e}")
                producer_exception.append(e)
            finally:
                # Signal end of data
                await request_queue.put(None)  # Sentinel value

        async def consumer():
            """Consumer coroutine: process requests from queue."""
            client = await self._get_http_client()
            while True:
                # Get next request (may block)
                request = await request_queue.get()

                # Check for sentinel (end of stream)
                if request is None:
                    # Re-put sentinel for other consumers
                    await request_queue.put(None)
                    break

                # Process request
                try:
                    await self._process_request_async(request, client)
                except Exception as e:
                    self.logger.error(f"Request processing failed: {e}")
                    await self.stats.increment_failed()
                finally:
                    request_queue.task_done()

        # Start producer and consumers
        producer_task = asyncio.create_task(producer())

        # Create consumer tasks
        consumers = []
        for _ in range(self.config.max_concurrency):
            consumer_task = asyncio.create_task(consumer())
            consumers.append(consumer_task)

        # Wait for producer to finish
        await producer_task

        # Wait for all consumers to finish
        await asyncio.gather(*consumers, return_exceptions=True)

        # Check for producer exception
        if producer_exception:
            raise producer_exception[0]

        # Update total count now that we know it
        self.stats.total_requests = total_requests[0]

        # Finalize
        await self._finalize_batch_async(total_requests[0])

    async def _finalize_batch_async(self, total_requests):
        """Common finalization logic for both batch and streaming modes."""
        self.stats.end_time = time.time()

        self.saver.cleanup()
        self.progress_tracker.finalize()
        self.server_manager.shutdown()
        self._print_summary()

    async def _process_request_async(self, request: LoadResult, client):
        """
        Process a single inference request asynchronously.

        Implements server-switching retry logic:
        - First attempt: Use load balancer to select server
        - On failure: Select different server for retry (avoid unhealthy servers)
        - Clear logging: Shows which server used and why/when switching

        Args:
            request: LoadResult containing messages and metadata
            client: Shared httpx.AsyncClient
        """
        # Prepare messages once (outside retry loop)
        messages = list(request.messages)
        if self.config.system_prompt:
            messages = [{"role": "system", "content": self.config.system_prompt}] + messages

        # Build request payload once (outside retry loop)
        payload = self.config.adapter.build_request(
            model_name=self.config.model_name,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            frequency_penalty=self.config.frequency_penalty,
            presence_penalty=self.config.presence_penalty,
        )

        # Log messages at DEBUG level for inspection
        self.logger.debug(f"Sending request {request.request_id} with messages:\n{messages}")

        last_server = None

        for attempt in range(self.config.max_retries + 1):
            server = self.load_balancer.get_server()
            if not server:
                self.logger.warning(f"[{request.request_id}] No healthy servers available")
                await self.stats.increment_failed()
                return

            # Skip if same server as last failed attempt (avoid retrying on unhealthy server)
            if last_server is not None and server.name == last_server.name:
                self.logger.debug(
                    f"[{request.request_id}] Skipping same server {server.name} for retry attempt {attempt + 1}"
                )
                # Try to get a different server by marking current as temporarily avoided
                # The load balancer's healthy filter will skip unhealthy servers
                server.decrement_active()
                continue

            # Increment active request count for load balancing
            server.increment_active()
            last_server = server

            try:
                # Log which server we're using
                self.logger.debug(
                    f"[{request.request_id}] Attempt {attempt + 1}/{self.config.max_retries + 1} "
                    f"using server {server.name} ({server.ip}:{server.port})"
                )

                # Send request async (no retry here - we handle retry at higher level)
                url = self.config.adapter.get_chat_url(server.base_url)
                start_time = time.time()

                response = await self._send_request_async_no_retry(
                    client=client,
                    url=url,
                    payload=payload
                )

                # Log timing for performance monitoring
                elapsed = time.time() - start_time
                self.logger.debug(
                    f"[{request.request_id}] Request to {server.name} took {elapsed:.2f}s"
                )

                # Parse response using adapter
                model_output = self.config.adapter.parse_response(response) if response else {}

                # Save result (run in thread pool to avoid blocking event loop)
                save_result = SaveResult(
                    request_id=request.request_id,
                    model_output=model_output,
                    additional_data=request.additional_data
                )
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(self._executor, self.saver.save, save_result)

                # Update statistics
                await self.stats.increment_completed()
                tokens = model_output.get('usage', {}).get('total_tokens', 0)
                await self.stats.add_tokens(tokens)

                # Record successful request on server
                server.record_success()

                # Update progress (run in thread pool to avoid blocking)
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(self._executor, self.progress_tracker.update, 1)

                # Success! Break out of retry loop
                break

            except Exception as e:
                # Record failed request on server
                server.record_error()

                # Extract exception type for better logging
                exc_type = type(e).__name__
                exc_msg = str(e)

                if attempt < self.config.max_retries:
                    # Retry with different server
                    delay = self.config.retry_delay * (2 ** attempt)
                    self.logger.warning(
                        f"[{request.request_id}] Attempt {attempt + 1}/{self.config.max_retries + 1} "
                        f"failed on {server.name} ({server.ip}:{server.port}): "
                        f"[{exc_type}] {exc_msg}. "
                        f"Retrying on different server in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    await self.stats.increment_retried()
                else:
                    # All retries exhausted
                    self.logger.error(
                        f"[{request.request_id}] All {self.config.max_retries + 1} attempts failed. "
                        f"Last server: {server.name} ({server.ip}:{server.port}), "
                        f"Error type: {exc_type}, Error: {exc_msg}"
                    )
                    await self.stats.increment_failed()
                    raise

    async def _send_request_async_no_retry(
        self,
        client,
        url: str,
        payload: dict
    ):
        """
        Send async HTTP request without retry logic.

        Retry logic is handled at a higher level in _process_request_async
        to enable server switching between retries.

        Args:
            client: Shared httpx.AsyncClient
            url: Request URL
            payload: JSON payload

        Returns:
            Response JSON

        Raises:
            httpx.HTTPError: If request fails
        """
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    def _finalize_stats(self):
        """Finalize statistics."""
        self.stats.end_time = time.time()

    def _print_summary(self):
        """Print processing summary."""
        self.logger.info("=" * 60)
        self.logger.info("Batch Inference Summary")
        self.logger.info("=" * 60)
        self.logger.info(f"Total Requests:     {self.stats.total_requests}")
        self.logger.info(f"Completed:          {self.stats.completed_requests}")
        self.logger.info(f"Failed:             {self.stats.failed_requests}")
        self.logger.info(f"Retried:            {self.stats.retried_requests}")
        self.logger.info(f"Success Rate:       {self.stats.success_rate:.2%}")
        self.logger.info(f"Total Tokens:       {self.stats.total_tokens}")
        self.logger.info(f"Duration:           {self.stats.duration:.2f}s")
        self.logger.info(f"Throughput:         {self.stats.completed_requests / self.stats.duration:.2f} req/s")
        self.logger.info("=" * 60)

        # Print server stats
        lb_stats = self.load_balancer.get_stats()
        self.logger.info(f"Server Stats: {lb_stats}")
