"""
Batch runner module - main orchestrator for batch inference.

Coordinates data loading, server management, request processing, and result saving.
"""
import time
import asyncio
import inspect
import threading
from typing import Dict, Any, Optional
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
from .utils.json_codec import json_codec
from .utils.resume import BitmapResumeStore, HybridResumeStore, LegacyResumeStore
from .adapters.base import ModelAdapter
from .adapters.openai_adapter import OpenAIAdapter


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
    rollout_n: int = 1
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
    producer_prefetch: int = 100

    # Resume settings
    resume: bool = True  # Enable resuming from existing output
    resume_backend: str = "legacy_output_scan"

    # Writer pipeline settings
    writer_queue_size: int = 1000
    writer_batch_size: int = 100
    writer_flush_interval_ms: int = 100
    writer_workers: int = 1

    # Load balancing / admission settings
    selection_sample_size: int = 2
    max_inflight_cost: float = 0.0

    # Multimodal runtime settings
    image_encode_workers: int = 4

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

    _REQUEST_SENTINEL = object()
    _SAVE_SENTINEL = object()

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
        self._validate_rollout_configuration()

        self.logger = logging.getLogger(__name__)
        self.stats = BatchStats()

        # Shared async HTTP client
        self._http_client = None

        # Dedicated writer executor to keep disk I/O off the event loop.
        self._writer_executor = ThreadPoolExecutor(
            max_workers=max(1, self.config.writer_workers),
            thread_name_prefix="writer_",
        )
        self._writer_failure = None
        self._stop_requested = threading.Event()

        # Initialize server manager and load balancer
        server_config = {
            'servers_dir': config.servers_dir,
            'request_timeout': config.request_timeout,
            'health_check_interval': config.get('health_check_interval', 30),
            'max_failures': config.get('max_failures', 5),
            'health_check_timeout': 5,
            'http2': config.http2,
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
            max_active_requests=max_active_requests,
            selection_sample_size=config.get('selection_sample_size', 2),
            max_inflight_cost=config.get('max_inflight_cost', 0.0),
        )

        # Progress tracking
        self.progress_tracker = ProgressTracker(
            total_items=self._estimate_total_items(),
            report_interval=config.progress_report_interval,
            stats=self.stats
        )
        self.resume_store = self._create_resume_store()

    def _validate_rollout_configuration(self):
        """Validate rollout-related settings before starting runtime state."""
        if self.config.rollout_n < 1:
            raise ValueError(f"runner.rollout_n must be >= 1, got {self.config.rollout_n}")

        if self.config.rollout_n > 1 and not isinstance(self.config.adapter, OpenAIAdapter):
            adapter_name = type(self.config.adapter).__name__ if self.config.adapter else "None"
            raise ValueError(
                "runner.rollout_n > 1 is only supported with OpenAIAdapter for "
                f"OpenAI-compatible chat completions. Got {adapter_name}."
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
                "Created shared AsyncClient (max_connections=%s, max_keepalive=%s, json_backend=%s)",
                self.config.http_max_connections,
                self.config.http_max_keepalive_connections,
                json_codec.backend_name,
            )
        return self._http_client

    async def close(self):
        """Clean up resources."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
            self.logger.info("Closed shared HTTP client")

        await self.server_manager.aclose()
        if self.resume_store:
            self.resume_store.close()
        self.loader.cleanup()
        self._writer_executor.shutdown(wait=True)
        self.logger.info("Shutdown writer executor")

    def _estimate_total_items(self) -> int:
        """Estimate total number of requests to process."""
        try:
            return len(self.loader)
        except (NotImplementedError, TypeError):
            return 0

    def run(self):
        """
        Execute the batch inference process (sync wrapper for async).

        This method is kept for backward compatibility.
        """
        asyncio.run(self.run_async())

    async def run_async(self):
        """Execute the batch inference process with async."""
        self.logger.info("Starting batch inference with %s concurrent requests", self.config.max_concurrency)
        self.logger.info("Healthy servers: %s", self.server_manager.get_server_count())
        if self.config.rollout_n > 1:
            self.logger.info(
                "rollout_n=%s enabled: requesting %s choices per input via a single API call",
                self.config.rollout_n,
                self.config.rollout_n,
            )
        await self.server_manager.start_async()

        streaming_mode = self.config.get('streaming', True)
        try:
            if streaming_mode:
                self.logger.info("Running in STREAMING mode (async pipeline processing)")
                await self._run_streaming_async()
            else:
                self.logger.info("Running in BATCH mode (load-all + fixed-concurrency async pipeline)")
                await self._run_batch_async()
        finally:
            await self.close()
            self._finalize_stats()

    async def _run_batch_async(self):
        """Batch mode now shares the same high-throughput scheduler as streaming mode."""
        await self._run_pipeline_async()

    async def _run_streaming_async(self):
        """Streaming mode shares the same high-throughput scheduler as batch mode."""
        await self._run_pipeline_async()

    async def _run_pipeline_async(self):
        """Run a producer/scheduler/writer pipeline with bounded in-flight requests."""
        self._ensure_runtime_state()
        request_queue: asyncio.Queue = asyncio.Queue(maxsize=self._get_producer_prefetch())
        completion_queue: asyncio.Queue = asyncio.Queue(maxsize=max(1, self.config.writer_queue_size))
        loop = asyncio.get_running_loop()
        self._writer_failure = loop.create_future()
        self._stop_requested.clear()

        producer_task = asyncio.create_task(
            asyncio.to_thread(self._produce_requests, request_queue, loop)
        )
        writer_tasks = [
            asyncio.create_task(self._writer_worker(completion_queue, index))
            for index in range(max(1, self.config.writer_workers))
        ]

        client = await self._get_http_client()
        pending_tasks = set()
        producer_finished = False
        writers_closed = False

        try:
            while True:
                self._raise_if_writer_failed()
                producer_finished = await self._fill_pending_from_queue(
                    request_queue=request_queue,
                    pending_tasks=pending_tasks,
                    client=client,
                    producer_finished=producer_finished,
                    completion_queue=completion_queue,
                )

                if not pending_tasks:
                    if producer_finished:
                        break
                    continue

                done, pending_tasks = await asyncio.wait(
                    pending_tasks,
                    return_when=asyncio.FIRST_COMPLETED
                )

                for task in done:
                    try:
                        task.result()
                    except Exception as exc:
                        self.logger.error("Request processing failed: %s", exc)

            producer_result = await producer_task
            self.stats.total_requests = producer_result['queued']

            for _ in writer_tasks:
                await completion_queue.put(self._SAVE_SENTINEL)
            writers_closed = True

            await asyncio.gather(*writer_tasks)
            self._raise_if_writer_failed()
            await self._finalize_batch_async(self.stats.total_requests)
        finally:
            self._stop_requested.set()
            if not producer_task.done():
                producer_task.cancel()
                await asyncio.gather(producer_task, return_exceptions=True)
            if not writers_closed:
                for _ in writer_tasks:
                    await completion_queue.put(self._SAVE_SENTINEL)
            await asyncio.gather(*writer_tasks, return_exceptions=True)

    async def _finalize_batch_async(self, total_requests):
        """Common finalization logic for both batch and streaming modes."""
        self.stats.end_time = time.time()

        self.saver.cleanup()
        self.progress_tracker.finalize()
        self._print_summary()

    async def _process_request_async(self, request: LoadResult, client, completion_queue: asyncio.Queue):
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
        self._raise_if_writer_failed()

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
            rollout_n=self.config.rollout_n,
            top_p=self.config.top_p,
            frequency_penalty=self.config.frequency_penalty,
            presence_penalty=self.config.presence_penalty,
        )
        payload_bytes = json_codec.dumps_bytes(payload)
        dispatch_cost = request.dispatch_cost or self._estimate_request_cost(
            messages=messages,
            additional_data=request.additional_data,
        )

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("Sending request %s with messages:\n%s", request.request_id, messages)

        excluded_servers = set()

        for attempt in range(self.config.max_retries + 1):
            server = self.load_balancer.get_server(excluded_names=excluded_servers or None)
            if not server:
                self.logger.warning("[%s] No healthy servers available", request.request_id)
                self.stats.increment_failed_sync()
                return

            # Increment active request count for load balancing
            server.increment_active(cost=dispatch_cost)

            try:
                # Log which server we're using
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(
                        "[%s] Attempt %s/%s using server %s (%s:%s)",
                        request.request_id,
                        attempt + 1,
                        self.config.max_retries + 1,
                        server.name,
                        server.ip,
                        server.port,
                    )

                # Send request async (no retry here - we handle retry at higher level)
                url = self.config.adapter.get_chat_url(server.base_url)
                start_time = time.time()

                response = await self._send_request_async_no_retry(
                    client=client,
                    url=url,
                    payload_bytes=payload_bytes
                )

                # Log timing for performance monitoring
                elapsed = time.time() - start_time
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(
                        "[%s] Request to %s took %.2fs",
                        request.request_id,
                        server.name,
                        elapsed,
                    )

                # Parse response using adapter
                model_output = self.config.adapter.parse_response(response) if response else {}
                server.record_success(cost=dispatch_cost)

                save_result = SaveResult(
                    request_id=request.request_id,
                    model_output=model_output,
                    additional_data=request.additional_data,
                    resume_key=request.resume_key,
                )
                self._raise_if_writer_failed()
                await completion_queue.put(save_result)

                # Success! Break out of retry loop
                break

            except Exception as e:
                # Record failed request on server
                server.record_error(cost=dispatch_cost)

                # Extract exception type for better logging
                exc_type = type(e).__name__
                exc_msg = str(e)

                if attempt < self.config.max_retries:
                    # Retry with different server
                    delay = self.config.retry_delay * (2 ** attempt)
                    excluded_servers.add(server.name)
                    self.logger.warning(
                        "[%s] Attempt %s/%s failed on %s (%s:%s): [%s] %s. Retrying in %.1fs",
                        request.request_id,
                        attempt + 1,
                        self.config.max_retries + 1,
                        server.name,
                        server.ip,
                        server.port,
                        exc_type,
                        exc_msg,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    self.stats.increment_retried_sync()
                else:
                    # All retries exhausted
                    self.logger.error(
                        "[%s] All %s attempts failed. Last server: %s (%s:%s), Error type: %s, Error: %s",
                        request.request_id,
                        self.config.max_retries + 1,
                        server.name,
                        server.ip,
                        server.port,
                        exc_type,
                        exc_msg,
                    )
                    self.stats.increment_failed_sync()
                    raise

    async def _send_request_async_no_retry(
        self,
        client,
        url: str,
        payload_bytes: bytes
    ):
        """
        Send async HTTP request without retry logic.

        Retry logic is handled at a higher level in _process_request_async
        to enable server switching between retries.

        Args:
            client: Shared httpx.AsyncClient
            url: Request URL
            payload_bytes: Serialized JSON payload

        Returns:
            Response JSON

        Raises:
            httpx.HTTPError: If request fails
        """
        response = await client.post(
            url,
            content=payload_bytes,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        return json_codec.loads(response.content)

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

    def _create_resume_store(self):
        """Create the configured resume backend with compatibility fallback."""
        if not self.config.resume:
            return None

        legacy_store = LegacyResumeStore(self.saver)
        backend = self.config.get('resume_backend', 'legacy_output_scan')
        if backend != 'bitmap':
            return legacy_store

        resume_path = self.saver.get_resume_store_path()
        if not resume_path:
            self.logger.warning("Bitmap resume requested but saver does not expose a resume path. Falling back to legacy output scan.")
            return legacy_store

        return HybridResumeStore(
            primary=BitmapResumeStore(base_dir=resume_path),
            fallback=legacy_store,
        )

    def _get_producer_prefetch(self) -> int:
        """Return the bounded request queue size used by the background producer."""
        return max(
            1,
            int(
                self.config.get(
                    'producer_prefetch',
                    self.config.get('stream_queue_size', 100),
                )
            ),
        )

    def _produce_requests(self, request_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> Dict[str, int]:
        """Load requests on a background thread and feed the async request queue."""
        queued = 0
        skipped = 0

        try:
            for item in self.loader:
                if self._stop_requested.is_set():
                    break
                if self.resume_store and self.resume_store.contains(
                    request_id=item.request_id,
                    resume_key=item.resume_key,
                ):
                    skipped += 1
                    continue

                if not self._threadsafe_queue_put(request_queue, item, loop):
                    break
                queued += 1
        finally:
            self._threadsafe_queue_put(request_queue, self._REQUEST_SENTINEL, loop)

        if skipped > 0:
            self.logger.info("Resume: skipped %s already completed requests", skipped)
        self.logger.info("Producer finished: %s requests queued", queued)
        return {'queued': queued, 'skipped': skipped}

    def _threadsafe_queue_put(self, queue: asyncio.Queue, item, loop: asyncio.AbstractEventLoop) -> bool:
        """Put an item into an asyncio.Queue from a background thread."""
        while not self._stop_requested.is_set():
            future = asyncio.run_coroutine_threadsafe(
                asyncio.wait_for(queue.put(item), timeout=0.5),
                loop,
            )
            try:
                future.result()
                return True
            except asyncio.TimeoutError:
                continue
            except Exception:
                if self._stop_requested.is_set():
                    return False
                raise

        return False

    async def _fill_pending_from_queue(
        self,
        request_queue: asyncio.Queue,
        pending_tasks: set,
        client,
        producer_finished: bool,
        completion_queue: asyncio.Queue,
    ) -> bool:
        """Refill request slots from the producer queue while capacity is available."""
        max_in_flight = max(1, self.config.max_concurrency)

        while len(pending_tasks) < max_in_flight and not producer_finished:
            request = await request_queue.get()
            if request is self._REQUEST_SENTINEL:
                producer_finished = True
                break

            pending_tasks.add(
                self._create_request_task(request, client, completion_queue)
            )

            while len(pending_tasks) < max_in_flight:
                try:
                    request = request_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                if request is self._REQUEST_SENTINEL:
                    producer_finished = True
                    break

                pending_tasks.add(
                    self._create_request_task(request, client, completion_queue)
                )

        return producer_finished

    async def _writer_worker(self, completion_queue: asyncio.Queue, worker_index: int):
        """Drain completion queue, batch results, and persist them off the event loop."""
        batch = []
        flush_interval = max(0.001, self.config.writer_flush_interval_ms / 1000.0)

        try:
            while True:
                try:
                    if batch:
                        item = await asyncio.wait_for(completion_queue.get(), timeout=flush_interval)
                    else:
                        item = await completion_queue.get()
                except asyncio.TimeoutError:
                    await self._flush_completion_batch(batch)
                    batch = []
                    continue

                if item is self._SAVE_SENTINEL:
                    break

                batch.append(item)
                if len(batch) >= max(1, self.config.writer_batch_size):
                    await self._flush_completion_batch(batch)
                    batch = []

            if batch:
                await self._flush_completion_batch(batch)
        except Exception as exc:
            if self._writer_failure and not self._writer_failure.done():
                self._writer_failure.set_exception(exc)
            self.logger.error("Writer worker %s failed: %s", worker_index, exc)
            raise

    async def _flush_completion_batch(self, batch):
        """Persist a completion batch and update post-save statistics."""
        if not batch:
            return

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._writer_executor, self.saver.save_batch, list(batch))

        if self.resume_store:
            self.resume_store.mark_many(
                (result.request_id, result.resume_key)
                for result in batch
            )

        token_count = sum(
            result.model_output.get('usage', {}).get('total_tokens', 0)
            for result in batch
            if result.model_output
        )
        self.stats.completed_requests += len(batch)
        self.stats.total_tokens += token_count
        self.progress_tracker.update(len(batch))

    def _raise_if_writer_failed(self):
        """Raise the first writer exception if the background writer has failed."""
        if self._writer_failure and self._writer_failure.done():
            exception = self._writer_failure.exception()
            if exception is not None:
                raise exception

    def _estimate_request_cost(self, messages, additional_data=None) -> float:
        """Estimate a lightweight request cost for routing decisions."""
        additional_data = additional_data or {}
        for key in ("dispatch_cost", "estimated_tokens", "input_tokens", "prompt_tokens"):
            value = additional_data.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return float(value)

        total_chars = 0
        image_count = 0
        for message in messages:
            content = message.get('content')
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get('type') == 'text':
                        total_chars += len(part.get('text', ''))
                    elif part.get('type') == 'image_url':
                        image_count += 1

        return float(max(1, total_chars // 4) + (image_count * 256))

    def _create_request_task(self, request: LoadResult, client, completion_queue: asyncio.Queue):
        """Create a request task while tolerating older two-argument test doubles."""
        process_request = self._process_request_async
        parameter_count = len(inspect.signature(process_request).parameters)
        if parameter_count <= 2:
            return asyncio.create_task(process_request(request, client))
        return asyncio.create_task(process_request(request, client, completion_queue))

    def _ensure_runtime_state(self):
        """Initialize lazily-created runtime state for tests and direct construction."""
        if not hasattr(self, '_stop_requested') or self._stop_requested is None:
            self._stop_requested = threading.Event()
        if not hasattr(self, '_writer_executor') or self._writer_executor is None:
            self._writer_executor = ThreadPoolExecutor(
                max_workers=max(1, self.config.writer_workers),
                thread_name_prefix="writer_",
            )
        if not hasattr(self, 'resume_store'):
            self.resume_store = None
