"""
Checkpoint module for resumable batch processing.

Allows batch jobs to resume from where they left off after interruption.
"""
import json
import threading
from pathlib import Path
from typing import Set, Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class CheckpointData:
    """Data stored in checkpoint."""
    completed_request_ids: Set[str]
    total_requests: int
    completed_count: int
    failed_count: int
    retried_count: int
    total_tokens: int
    last_update_time: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'completed_request_ids': list(self.completed_request_ids),
            'total_requests': self.total_requests,
            'completed_count': self.completed_count,
            'failed_count': self.failed_count,
            'retried_count': self.retried_count,
            'total_tokens': self.total_tokens,
            'last_update_time': self.last_update_time,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointData':
        """Create from dictionary."""
        return cls(
            completed_request_ids=set(data.get('completed_request_ids', [])),
            total_requests=data.get('total_requests', 0),
            completed_count=data.get('completed_count', 0),
            failed_count=data.get('failed_count', 0),
            retried_count=data.get('retried_count', 0),
            total_tokens=data.get('total_tokens', 0),
            last_update_time=data.get('last_update_time', 0),
        )


class CheckpointManager:
    """
    Manages checkpoint saving and loading for resumable batch processing.

    Usage:
        checkpoint = CheckpointManager('checkpoints/run1.json', interval=10)
        checkpoint.initialize(total_requests=100)

        # During processing
        if checkpoint.is_completed(request_id):
            continue  # Skip already completed
        # ... process request ...
        checkpoint.mark_completed(request_id, stats)

        # Save periodically
        checkpoint.maybe_save()
    """

    def __init__(self, checkpoint_path: str, save_interval: int = 10):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_path: Path to checkpoint file
            save_interval: Save checkpoint every N completions (default: 10)
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.save_interval = save_interval
        self._lock = threading.Lock()

        self.data: Optional[CheckpointData] = None
        self._pending_count = 0

        # Create parent directory if needed
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def load_or_create(self, total_requests: int) -> CheckpointData:
        """
        Load existing checkpoint or create new one.

        Args:
            total_requests: Total number of requests to process

        Returns:
            CheckpointData instance
        """
        with self._lock:
            if self.checkpoint_path.exists():
                try:
                    with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    self.data = CheckpointData.from_dict(data)

                    logger.info(f"Loaded checkpoint: {self.data.completed_count}/{self.data.total_requests} completed")
                    return self.data
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint, starting fresh: {e}")

            # Create new checkpoint
            self.data = CheckpointData(
                completed_request_ids=set(),
                total_requests=total_requests,
                completed_count=0,
                failed_count=0,
                retried_count=0,
                total_tokens=0,
                last_update_time=0,
            )
            self.save()
            return self.data

    def is_completed(self, request_id: str) -> bool:
        """
        Check if a request has been completed.

        Args:
            request_id: Request identifier

        Returns:
            True if request was already completed
        """
        with self._lock:
            if self.data is None:
                return False
            return request_id in self.data.completed_request_ids

    def mark_completed(
        self,
        request_id: str,
        tokens: int = 0,
        retried: bool = False
    ):
        """
        Mark a request as completed.

        Args:
            request_id: Request identifier
            tokens: Number of tokens used
            retried: Whether this request was retried
        """
        with self._lock:
            if self.data is None:
                return

            self.data.completed_request_ids.add(request_id)
            self.data.completed_count += 1
            self.data.total_tokens += tokens
            if retried:
                self.data.retried_count += 1

            self._pending_count += 1
            self.data.last_update_time = self._pending_count

    def mark_failed(self):
        """Increment failed count."""
        with self._lock:
            if self.data is None:
                return
            self.data.failed_count += 1

    def maybe_save(self) -> bool:
        """
        Save checkpoint if enough completions accumulated.

        Returns:
            True if checkpoint was saved
        """
        with self._lock:
            if self._pending_count >= self.save_interval:
                self.save()
                self._pending_count = 0
                return True
        return False

    def save(self):
        """Force save checkpoint to disk."""
        with self._lock:
            if self.data is None:
                return

            try:
                with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
                    json.dump(self.data.to_dict(), f, indent=2)
                logger.debug(f"Checkpoint saved: {self.checkpoint_path}")
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")

    def delete(self):
        """Delete checkpoint file."""
        with self._lock:
            if self.checkpoint_path.exists():
                self.checkpoint_path.unlink()
                logger.info(f"Checkpoint deleted: {self.checkpoint_path}")

    def get_progress(self) -> Dict[str, Any]:
        """Get current progress from checkpoint."""
        with self._lock:
            if self.data is None:
                return {}

            return {
                'completed': self.data.completed_count,
                'total': self.data.total_requests,
                'failed': self.data.failed_count,
                'retried': self.data.retried_count,
                'tokens': self.data.total_tokens,
            }
