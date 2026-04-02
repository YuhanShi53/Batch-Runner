"""
Tests for JSONL saver functionality.
"""
import csv
import json

from src.savers.console_saver import ConsoleResultSaver
from src.savers.csv_saver import CSVResultSaver
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
        "contents": ["hello"],
        "finish_reason": "stop",
        "finish_reasons": ["stop"],
        "usage": {"total_tokens": 5},
        "additional_data": {"source": "unit"},
    }

    assert saver.get_resume_store_path().endswith(".resume")


def test_jsonl_saver_minimal_projection_preserves_multiple_choices(tmp_path):
    """Minimal projection should keep first-choice compatibility and multi-choice fields."""
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
            request_id="req-2",
            model_output={
                "choices": [
                    {"message": {"content": "hello"}, "finish_reason": "stop"},
                    {"message": {"content": "bonjour"}, "finish_reason": "length"},
                ],
                "usage": {"total_tokens": 8},
            },
        )
    )
    saver.cleanup()

    payload = json_codec.loads(output_path.read_text(encoding="utf-8").strip())
    assert payload["content"] == "hello"
    assert payload["contents"] == ["hello", "bonjour"]
    assert payload["finish_reason"] == "stop"
    assert payload["finish_reasons"] == ["stop", "length"]


def test_csv_saver_writes_multi_choice_columns(tmp_path):
    """CSV saver should keep first-choice compatibility while persisting all contents."""
    output_path = tmp_path / "results.csv"
    saver = CSVResultSaver({"output_path": str(output_path)})

    saver.save(
        SaveResult(
            request_id="req-csv",
            model_output={
                "choices": [
                    {"message": {"content": "alpha"}, "finish_reason": "stop"},
                    {"message": {"content": "beta"}, "finish_reason": "stop"},
                ],
                "usage": {"total_tokens": 12},
            },
        )
    )
    saver.cleanup()

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        row_map = next(csv.DictReader(handle))

    assert row_map["content"] == "alpha"
    assert json.loads(row_map["contents"]) == ["alpha", "beta"]
    assert row_map["num_choices"] == "2"


def test_console_saver_reports_choice_count(capsys):
    """Console saver should show the first response and mention multi-choice counts."""
    saver = ConsoleResultSaver({})
    saver.save(
        SaveResult(
            request_id="req-console",
            model_output={
                "choices": [
                    {"message": {"content": "first"}, "finish_reason": "stop"},
                    {"message": {"content": "second"}, "finish_reason": "stop"},
                ],
                "usage": {"total_tokens": 4},
            },
        )
    )

    output = capsys.readouterr().out
    assert "Response: first" in output
    assert "Choices: 2" in output
