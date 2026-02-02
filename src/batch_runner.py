"""
Batch runner module - main orchestrator for batch inference.

Coordinates data loading, server management, request processing, and result saving.
"""
import time
import threading
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
import logging

try:
    import requests
except ImportError:
    requests = None

from .loaders.base import DataLoader, LoadResult
from .savers.base import ResultSaver, SaveResult
from .servers.manager import VLLMServerManager
from .servers.load_balancer import LoadBalancer
from .utils.progress import ProgressTracker
from .utils.retry import retry_with_backoff
from .utils.checkpoint import CheckpointManager
from .adapters.base import ModelAdapter


@dataclass
class BatchConfig:
    """Configuration for batch inference."""
    # Concurrency settings
    max_concurrency: int = 10
    max_retries: int = 3
    retry_delay: float = 1.0
    request_timeout: int = 120

    # Rollout settings
    num_rollouts: int = 1

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

    # Progress settings
    progress_report_interval: int = 10

    # Optional sampling parameters
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    # Adapter settings
    adapter_class: str = "OpenAIAdapter"
    adapter: ModelAdapter = None

    # Checkpoint settings
    enable_checkpoint: bool = False
    checkpoint_path: str = "checkpoints/batch_checkpoint.json"
    checkpoint_interval: int = 10

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
    _lock: threading.Lock = field(default_factory=threading.Lock)

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

    def increment_completed(self):
        """Increment completed requests count."""
        with self._lock:
            self.completed_requests += 1

    def increment_failed(self):
        """Increment failed requests count."""
        with self._lock:
            self.failed_requests += 1

    def increment_retried(self):
        """Increment retried requests count."""
        with self._lock:
            self.retried_requests += 1

    def add_tokens(self, count: int):
        """Add to total tokens count."""
        with self._lock:
            self.total_tokens += count


class BatchRunner:
    """
    Main batch inference runner.

    Orchestrates:
    - Data loading
    - Server management and load balancing
    - Concurrent request processing
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
            # Update the load balancer with the latest server list
            self.load_balancer.update_servers(self.server_manager.get_all_servers())

        self.server_manager.register_state_change_callback(on_server_state_change)

        # Initialize load balancer with current server list
        allow_fallback = config.get('allow_unhealthy_fallback', False)
        self.load_balancer = LoadBalancer(
            self.server_manager.get_all_servers(),
            strategy=config.load_balancing_strategy,
            allow_fallback=allow_fallback
        )

        # Initialize checkpoint manager if enabled
        self.checkpoint_manager: Optional[CheckpointManager] = None
        if config.enable_checkpoint:
            self.checkpoint_manager = CheckpointManager(
                checkpoint_path=config.checkpoint_path,
                save_interval=config.checkpoint_interval
            )

        # Progress tracking
        self.progress_tracker = ProgressTracker(
            total_items=self._estimate_total_items(),
            report_interval=config.progress_report_interval
        )

        # Request queue
        self._request_queue: List[LoadResult] = []
        self._lock = threading.Lock()

    def _estimate_total_items(self) -> int:
        """Estimate total number of requests to process."""
        try:
            base_count = len(self.loader)
            return base_count * self.config.num_rollouts
        except NotImplementedError:
            return 0

    def run(self):
        """Execute the batch inference process."""
        self.logger.info(f"Starting batch inference with {self.config.max_concurrency} workers")
        self.logger.info(f"Rollouts per sample: {self.config.num_rollouts}")
        self.logger.info(f"Healthy servers: {self.server_manager.get_server_count()}")

        # Load all data into queue
        for item in self.loader:
            for rollout_idx in range(self.config.num_rollouts):
                rollout_result = LoadResult(
                    messages=item.messages,
                    request_id=f"{item.request_id}_rollout_{rollout_idx}",
                    additional_data=item.additional_data
                )
                self._request_queue.append(rollout_result)

        total_requests = len(self._request_queue)
        self.stats.total_requests = total_requests
        self.logger.info(f"Total requests to process: {total_requests}")

        # Load checkpoint and filter completed requests
        if self.checkpoint_manager:
            checkpoint_data = self.checkpoint_manager.load_or_create(total_requests)
            completed_count = checkpoint_data.completed_count

            if completed_count > 0:
                self.logger.info(f"Resuming from checkpoint: {completed_count}/{total_requests} already completed")
                # Restore stats from checkpoint
                self.stats.completed_requests = checkpoint_data.completed_count
                self.stats.failed_requests = checkpoint_data.failed_count
                self.stats.retried_requests = checkpoint_data.retried_count
                self.stats.total_tokens = checkpoint_data.total_tokens

                # Filter out completed requests
                pending_requests = [
                    req for req in self._request_queue
                    if not self.checkpoint_manager.is_completed(req.request_id)
                ]
                self._request_queue = pending_requests
                self.logger.info(f"Pending requests to process: {len(self._request_queue)}")

                # Update progress tracker to account for completed items
                for _ in range(completed_count):
                    self.progress_tracker.update(1)

            if not self._request_queue:
                self.logger.info("All requests already completed!")
                self.stats.end_time = time.time()
                self._print_summary()
                return

        # Process with thread pool
        with ThreadPoolExecutor(max_workers=self.config.max_concurrency) as executor:
            futures = {
                executor.submit(self._process_request, request): request
                for request in self._request_queue
            }

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Request processing failed: {e}")
                    self.stats.increment_failed()
                    if self.checkpoint_manager:
                        self.checkpoint_manager.mark_failed()

        # Finalize
        self.stats.end_time = time.time()
        self.saver.cleanup()
        self.progress_tracker.finalize()
        self.server_manager.shutdown()

        # Handle checkpoint after completion
        if self.checkpoint_manager:
            if self.stats.completed_requests == total_requests:
                self.logger.info("All requests completed, deleting checkpoint")
                self.checkpoint_manager.delete()
            else:
                self.checkpoint_manager.save()

        self._print_summary()

    def _process_request(self, request: LoadResult):
        """
        Process a single inference request.

        Args:
            request: LoadResult containing messages and metadata
        """
        # Check if already completed (for checkpoint resume)
        if self.checkpoint_manager and self.checkpoint_manager.is_completed(request.request_id):
            self.logger.debug(f"Skipping already completed request: {request.request_id}")
            return

        server = self.load_balancer.get_server()
        if not server:
            raise RuntimeError("No healthy servers available")

        # Prepare messages
        messages = list(request.messages)
        # Prepend system prompt if configured
        if self.config.system_prompt:
            messages = [{"role": "system", "content": self.config.system_prompt}] + messages

        # Build request payload using adapter
        payload = self.config.adapter.build_request(
            model_name=self.config.model_name,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            frequency_penalty=self.config.frequency_penalty,
            presence_penalty=self.config.presence_penalty,
        )

        # Send request with retry (tracks if retried)
        was_retried, response = self._send_request_with_retry(server, payload)

        # Parse response using adapter
        model_output = self.config.adapter.parse_response(response) if response else {}

        # Save result
        save_result = SaveResult(
            request_id=request.request_id,
            model_output=model_output,
            additional_data=request.additional_data
        )
        self.saver.save(save_result)

        # Update statistics
        self.stats.increment_completed()
        tokens = model_output.get('usage', {}).get('total_tokens', 0)
        self.stats.add_tokens(tokens)
        server.request_count += 1

        # Update checkpoint
        if self.checkpoint_manager:
            self.checkpoint_manager.mark_completed(
                request_id=request.request_id,
                tokens=tokens,
                retried=was_retried
            )
            self.checkpoint_manager.maybe_save()

        # Update progress
        self.progress_tracker.update(1)

    def _send_request_with_retry(self, server, payload: dict):
        """
        Send request to server with retry logic.

        Args:
            server: VLLMServer instance
            payload: Request payload

        Returns:
            Tuple of (was_retried: bool, response: Response object)
        """
        was_retried = False

        @retry_with_backoff(
            max_retries=self.config.max_retries,
            base_delay=self.config.retry_delay,
            exceptions=(requests.exceptions.RequestException, requests.exceptions.Timeout)
        )
        def _send():
            nonlocal was_retried
            # Use adapter's get_chat_url to build the endpoint
            url = self.config.adapter.get_chat_url(server.base_url)
            response = requests.post(
                url,
                json=payload,
                timeout=self.config.request_timeout
            )
            response.raise_for_status()
            was_retried = False  # Success, no retry needed
            return response

        try:
            result = _send()
            return (was_retried, result)
        except Exception as e:
            was_retried = True
            self.stats.increment_retried()
            raise

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
