"""
Resume backends for large-scale batch processing.
"""
from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from threading import RLock
from typing import Iterable, Optional, Tuple


logger = logging.getLogger(__name__)

ResumeKey = Tuple[str, int, int]


class ResumeStore(ABC):
    """Abstract base class for resume state backends."""

    @abstractmethod
    def contains(self, request_id: Optional[str] = None, resume_key: Optional[ResumeKey] = None) -> bool:
        """Return True when the request has already been completed."""

    def mark_completed(self, request_id: Optional[str] = None, resume_key: Optional[ResumeKey] = None) -> None:
        """Mark a single request as completed."""
        self.mark_many([(request_id, resume_key)])

    def mark_many(self, items: Iterable[Tuple[Optional[str], Optional[ResumeKey]]]) -> None:
        """Mark many requests as completed."""
        for request_id, resume_key in items:
            self._mark_one(request_id=request_id, resume_key=resume_key)

    @abstractmethod
    def _mark_one(self, request_id: Optional[str], resume_key: Optional[ResumeKey]) -> None:
        """Backend-specific single-item update."""

    def close(self) -> None:
        """Flush and release any backend resources."""


class LegacyResumeStore(ResumeStore):
    """Compatibility backend that delegates to saver.is_completed()."""

    def __init__(self, saver):
        self._saver = saver

    def contains(self, request_id: Optional[str] = None, resume_key: Optional[ResumeKey] = None) -> bool:
        del resume_key
        if request_id is None:
            return False
        return self._saver.is_completed(request_id)

    def _mark_one(self, request_id: Optional[str], resume_key: Optional[ResumeKey]) -> None:
        del request_id, resume_key


class HybridResumeStore(ResumeStore):
    """Use bitmap resume when a resume_key is available and legacy lookup otherwise."""

    def __init__(self, primary: ResumeStore, fallback: ResumeStore):
        self.primary = primary
        self.fallback = fallback

    def contains(self, request_id: Optional[str] = None, resume_key: Optional[ResumeKey] = None) -> bool:
        if resume_key is not None:
            return self.primary.contains(request_id=request_id, resume_key=resume_key)
        return self.fallback.contains(request_id=request_id, resume_key=resume_key)

    def _mark_one(self, request_id: Optional[str], resume_key: Optional[ResumeKey]) -> None:
        if resume_key is not None:
            self.primary.mark_completed(request_id=request_id, resume_key=resume_key)
        else:
            self.fallback.mark_completed(request_id=request_id, resume_key=resume_key)

    def close(self) -> None:
        self.primary.close()
        self.fallback.close()


class _BitmapBucket:
    """In-memory mutable bitmap for a specific (source, item_idx)."""

    def __init__(self, path: Path):
        self.path = path
        self.buffer = bytearray(path.read_bytes()) if path.exists() else bytearray()
        self.dirty = False

    def contains(self, line_number: int) -> bool:
        if line_number < 1:
            return False

        bit_index = line_number - 1
        byte_index = bit_index // 8
        if byte_index >= len(self.buffer):
            return False

        bit_mask = 1 << (bit_index % 8)
        return bool(self.buffer[byte_index] & bit_mask)

    def mark(self, line_number: int) -> None:
        if line_number < 1:
            return

        bit_index = line_number - 1
        byte_index = bit_index // 8
        bit_mask = 1 << (bit_index % 8)

        if byte_index >= len(self.buffer):
            self.buffer.extend(b"\x00" * (byte_index + 1 - len(self.buffer)))

        if not self.buffer[byte_index] & bit_mask:
            self.buffer[byte_index] |= bit_mask
            self.dirty = True

    def flush(self) -> None:
        if not self.dirty:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(self.buffer)
        self.dirty = False


class BitmapResumeStore(ResumeStore):
    """
    Exact bitmap-based resume backend keyed by (source_file, line_num, item_idx).

    Each source file and split index gets its own compact bitmap, keeping memory
    bounded while allowing exact resume checks for sequential JSONL-style inputs.
    """

    def __init__(self, base_dir: Path, max_open_bitmaps: int = 8):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.max_open_bitmaps = max(1, max_open_bitmaps)
        self._lock = RLock()
        self._buckets: "OrderedDict[tuple[str, int], _BitmapBucket]" = OrderedDict()

    def contains(self, request_id: Optional[str] = None, resume_key: Optional[ResumeKey] = None) -> bool:
        del request_id
        if resume_key is None:
            return False

        source_file, line_num, item_idx = resume_key
        with self._lock:
            bucket = self._get_bucket(source_file, item_idx)
            return bucket.contains(line_num)

    def _mark_one(self, request_id: Optional[str], resume_key: Optional[ResumeKey]) -> None:
        del request_id
        if resume_key is None:
            return

        source_file, line_num, item_idx = resume_key
        with self._lock:
            bucket = self._get_bucket(source_file, item_idx)
            bucket.mark(line_num)

    def close(self) -> None:
        with self._lock:
            for bucket in self._buckets.values():
                bucket.flush()
            self._buckets.clear()

    def _get_bucket(self, source_file: str, item_idx: int) -> _BitmapBucket:
        cache_key = (source_file, item_idx)
        bucket = self._buckets.get(cache_key)
        if bucket is not None:
            self._buckets.move_to_end(cache_key)
            return bucket

        bucket = _BitmapBucket(self._bucket_path(source_file, item_idx))
        self._buckets[cache_key] = bucket
        self._evict_if_needed()
        return bucket

    def _bucket_path(self, source_file: str, item_idx: int) -> Path:
        source_hash = hashlib.sha1(source_file.encode("utf-8")).hexdigest()
        return self.base_dir / source_hash[:2] / f"{source_hash}_idx{item_idx}.bitmap"

    def _evict_if_needed(self) -> None:
        while len(self._buckets) > self.max_open_bitmaps:
            _, bucket = self._buckets.popitem(last=False)
            bucket.flush()
