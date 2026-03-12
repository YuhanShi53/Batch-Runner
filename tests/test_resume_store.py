"""
Tests for resume backends.
"""
from pathlib import Path

from src.utils.resume import BitmapResumeStore


def test_bitmap_resume_store_persists_across_reopen(tmp_path):
    """Bitmap resume should persist completed keys on disk."""
    store_path = tmp_path / "resume"
    store = BitmapResumeStore(store_path)

    keys = [
        ("source_a.jsonl", 1, 0),
        ("source_a.jsonl", 8, 0),
        ("source_a.jsonl", 8, 1),
        ("source_b.jsonl", 3, 0),
    ]

    for key in keys:
        assert store.contains(resume_key=key) is False
        store.mark_completed(resume_key=key)
        assert store.contains(resume_key=key) is True

    assert store.contains(resume_key=("source_a.jsonl", 2, 0)) is False
    store.close()

    reopened = BitmapResumeStore(store_path)
    for key in keys:
        assert reopened.contains(resume_key=key) is True
    assert reopened.contains(resume_key=("source_a.jsonl", 2, 0)) is False
    reopened.close()
