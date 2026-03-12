"""
Tests for batch runner concurrency behavior.
"""
import asyncio
import logging

from src.batch_runner import BatchRunner, BatchConfig, BatchStats
from src.loaders.base import LoadResult


class StubLoader:
    """Simple iterable loader for batch runner tests."""

    def __init__(self, items):
        self.items = items

    def __iter__(self):
        return iter(self.items)


class StubSaver:
    """Saver stub with no completed requests."""

    def is_completed(self, request_id):
        return False

    def cleanup(self):
        pass


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
