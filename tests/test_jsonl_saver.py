"""
Tests for JSONL saver functionality.
"""
from pathlib import Path

from src.savers.base import SaveResult
from src.savers.jsonl_saver import JSONLResultSaver
from src.utils.json_codec import json_codec


def test_jsonl_saver_minimal_projection(tmp_path):
    """Minimal projection should avoid writing the full model output payload."""
    output_path = tmp_path / "results.jsonl"
    saver = JSONLResultSaver(
        {
            "output_path": str(output_path),
            "output_projection": "minimal",
            "immediate_flush": True,
        }
    )

    saver.save(
        SaveResult(
            request_id="req-1",
            model_output={
                "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
                "usage": {"total_tokens": 5},
            },
            additional_data={"source": "unit"},
        )
    )
    saver.cleanup()

    payload = json_codec.loads(output_path.read_text(encoding="utf-8").strip())
    assert payload == {
        "request_id": "req-1",
        "content": "hello",
        "finish_reason": "stop",
        "usage": {"total_tokens": 5},
        "additional_data": {"source": "unit"},
    }

    assert saver.get_resume_store_path().endswith(".resume")
