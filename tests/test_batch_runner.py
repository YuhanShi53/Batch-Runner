"""
Tests for batch runner concurrency behavior.
"""
import asyncio
import logging

from src.batch_runner import BatchRunner, BatchConfig, BatchStats
from src.loaders.base import LoadResult
from src.savers.base import SaveResult


class StubLoader:
    """Simple iterable loader for batch runner tests."""

    def __init__(self, items):
        self.items = items

    def __iter__(self):
        return iter(self.items)


class StreamingLenUnsupportedLoader:
    """Loader stub that matches streaming loaders which reject len()."""

    def __len__(self):
        raise TypeError("Streaming mode does not support len()")


class StubSaver:
    """Saver stub with no completed requests."""

    def is_completed(self, request_id):
        return False

    def cleanup(self):
        pass


def test_estimate_total_items_returns_zero_when_loader_len_raises_type_error():
    """Streaming loaders that reject len() should not break runner initialization."""
    runner = object.__new__(BatchRunner)
    runner.loader = StreamingLenUnsupportedLoader()

    assert runner._estimate_total_items() == 0


def test_run_batch_async_refills_slots_when_streaming_disabled():
    """Batch mode should keep a fixed number of requests in flight and refill slots as they free up."""
    items = [
        LoadResult(messages=[{"role": "user", "content": "a"}], request_id="req0"),
        LoadResult(messages=[{"role": "user", "content": "b"}], request_id="req1"),
        LoadResult(messages=[{"role": "user", "content": "c"}], request_id="req2"),
        LoadResult(messages=[{"role": "user", "content": "d"}], request_id="req3"),
    ]

    async def exercise_runner():
        runner = object.__new__(BatchRunner)
        runner.config = BatchConfig(max_concurrency=2, streaming=False, resume=False)
        runner.loader = StubLoader(items)
        runner.saver = StubSaver()
        runner.logger = logging.getLogger("tests.test_batch_runner")
        runner.stats = BatchStats()

        finalized = []
        finish_order = []
        refill_happened = False
        active_requests = 0
        max_active_requests = 0
        durations = {
            "req0": 0.01,
            "req1": 0.05,
            "req2": 0.01,
            "req3": 0.01,
        }

        async def fake_get_http_client():
            return object()

        async def fake_finalize(total_requests):
            finalized.append(total_requests)

        async def fake_process_request(request, client):
            nonlocal active_requests, max_active_requests, refill_happened
            del client
            active_requests += 1
            max_active_requests = max(max_active_requests, active_requests)

            if request.request_id == "req2" and "req1" not in finish_order:
                refill_happened = True

            await asyncio.sleep(durations[request.request_id])
            finish_order.append(request.request_id)
            active_requests -= 1

        runner._get_http_client = fake_get_http_client
        runner._finalize_batch_async = fake_finalize
        runner._process_request_async = fake_process_request

        await runner._run_batch_async()

        assert finalized == [len(items)]
        assert max_active_requests == runner.config.max_concurrency
        assert refill_happened is True

    asyncio.run(exercise_runner())


def test_flush_completion_batch_updates_stats_and_resume():
    """Writer batch flush should persist results, update stats, and mark resume state."""

    class StubBatchSaver:
        def __init__(self):
            self.saved_batches = []

        def save_batch(self, results):
            self.saved_batches.append([result.request_id for result in results])

    class StubResumeStore:
        def __init__(self):
            self.marked = []

        def mark_many(self, items):
            self.marked.extend(list(items))

    class StubProgress:
        def __init__(self):
            self.counts = []

        def update(self, count):
            self.counts.append(count)

    async def exercise():
        runner = object.__new__(BatchRunner)
        runner.config = BatchConfig(writer_workers=1)
        runner.saver = StubBatchSaver()
        runner.resume_store = StubResumeStore()
        runner.progress_tracker = StubProgress()
        runner.stats = BatchStats()
        runner._writer_executor = None
        runner._ensure_runtime_state()

        results = [
            SaveResult(
                request_id="req-1",
                model_output={"usage": {"total_tokens": 7}},
                resume_key=("source", 1, 0),
            ),
            SaveResult(
                request_id="req-2",
                model_output={"usage": {"total_tokens": 3}},
                resume_key=("source", 2, 0),
            ),
        ]

        await runner._flush_completion_batch(results)

        assert runner.saver.saved_batches == [["req-1", "req-2"]]
        assert runner.resume_store.marked == [
            ("req-1", ("source", 1, 0)),
            ("req-2", ("source", 2, 0)),
        ]
        assert runner.progress_tracker.counts == [2]
        assert runner.stats.completed_requests == 2
        assert runner.stats.total_tokens == 10

        runner._writer_executor.shutdown(wait=True)

    asyncio.run(exercise())
