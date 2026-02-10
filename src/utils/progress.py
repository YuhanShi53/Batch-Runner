"""
Progress tracking module.

Provides progress tracking and reporting for batch operations.
"""
import time
import logging
from typing import Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ProgressTracker:
    """
    Track and report progress for long-running operations.

    Attributes:
        total_items: Total number of items to process (0 if unknown)
        completed_items: Number of items completed
        report_interval: Minimum seconds between progress reports
        stats: Optional BatchStats for detailed reporting
    """
    total_items: int = 0
    report_interval: int = 10
    completed_items: int = field(default=0)
    start_time: float = field(default_factory=time.time)
    last_report_time: float = field(default_factory=time.time)
    # Remove lock for better performance - accept small timing race conditions
    stats: Optional[Any] = field(default=None)

    def update(self, count: int = 1):
        """
        Update progress.

        Args:
            count: Number of items completed since last update
        """
        self.completed_items += count
        current_time = time.time()

        # Report if interval has passed
        if current_time - self.last_report_time >= self.report_interval:
            self._report()
            self.last_report_time = current_time

    def _report(self):
        """Print progress report."""
        elapsed = time.time() - self.start_time
        rate = self.completed_items / elapsed if elapsed > 0 else 0

        if self.total_items > 0:
            percentage = (self.completed_items / self.total_items) * 100
            eta = (self.total_items - self.completed_items) / rate if rate > 0 else 0

            logger.info(
                f"[Progress] {self.completed_items}/{self.total_items} "
                f"({percentage:.1f}%) | "
                f"Rate: {rate:.2f} items/sec | "
                f"ETA: {eta:.0f}s"
            )
        else:
            logger.info(
                f"[Progress] {self.completed_items} completed | "
                f"Rate: {rate:.2f} items/sec"
            )

        # Print detailed stats if available
        if self.stats:
            self._report_stats()

    def get_progress(self) -> dict:
        """
        Get current progress statistics.

        Returns:
            Dictionary with progress statistics
        """
        elapsed = time.time() - self.start_time
        rate = self.completed_items / elapsed if elapsed > 0 else 0

        return {
            'completed': self.completed_items,
            'total': self.total_items,
            'percentage': (self.completed_items / self.total_items * 100) if self.total_items > 0 else None,
            'elapsed_time': elapsed,
            'rate': rate,
            'eta': (self.total_items - self.completed_items) / rate if rate > 0 and self.total_items > 0 else None
        }

    def finalize(self):
        """Print final progress report."""
        elapsed = time.time() - self.start_time
        rate = self.completed_items / elapsed if elapsed > 0 else 0

        if self.total_items > 0:
            percentage = (self.completed_items / self.total_items) * 100
            logger.info(
                f"[Progress] Complete: {self.completed_items}/{self.total_items} "
                f"({percentage:.1f}%) | "
                f"Total time: {elapsed:.2f}s | "
                f"Avg rate: {rate:.2f} items/sec"
            )
        else:
            logger.info(
                f"[Progress] Complete: {self.completed_items} items | "
                f"Total time: {elapsed:.2f}s | "
                f"Avg rate: {rate:.2f} items/sec"
            )

        # Print detailed stats if available
        if self.stats:
            self._report_stats()

    def _report_stats(self):
        """Print detailed statistics from BatchStats."""
        if not self.stats:
            return

        # Access stats directly without lock for performance
        failed = self.stats.failed_requests
        retried = self.stats.retried_requests
        tokens = self.stats.total_tokens
        completed = self.stats.completed_requests

        # Calculate success rate based on actual results
        total = self.stats.total_requests
        if total > 0:
            success_rate = (completed / total) * 100 if total > 0 else 0.0
        elif completed > 0:
            success_count = completed - failed
            success_rate = (success_count / completed) * 100 if completed > 0 else 0.0
        else:
            success_rate = 0.0

        parts = []
        if failed > 0:
            parts.append(f"Failed: {failed}")
        if retried > 0:
            parts.append(f"Retried: {retried}")
        if tokens > 0:
            parts.append(f"Tokens: {tokens:,}")
        parts.append(f"Success: {success_rate:.1f}%")

        stats_str = " | ".join(parts)
        logger.info(f"[Stats] {stats_str}")
