"""
Progress tracking module.

Provides progress tracking and reporting for batch operations.
"""
import time
import threading
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class ProgressTracker:
    """
    Track and report progress for long-running operations.

    Attributes:
        total_items: Total number of items to process (0 if unknown)
        completed_items: Number of items completed
        report_interval: Minimum seconds between progress reports
    """
    total_items: int = 0
    report_interval: int = 10
    completed_items: int = field(default=0)
    start_time: float = field(default_factory=time.time)
    last_report_time: float = field(default_factory=time.time)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, count: int = 1):
        """
        Update progress.

        Args:
            count: Number of items completed since last update
        """
        with self._lock:
            self.completed_items += count
            current_time = time.time()

            # Report if interval has passed
            if current_time - self.last_report_time >= self.report_interval:
                self._report()
                self.last_report_time = current_time

    def _report(self):
        """Print progress report."""
        if self.total_items > 0:
            percentage = (self.completed_items / self.total_items) * 100
            elapsed = time.time() - self.start_time
            rate = self.completed_items / elapsed if elapsed > 0 else 0
            eta = (self.total_items - self.completed_items) / rate if rate > 0 else 0

            print(
                f"[Progress] {self.completed_items}/{self.total_items} "
                f"({percentage:.1f}%) | "
                f"Rate: {rate:.2f} items/sec | "
                f"ETA: {eta:.0f}s"
            )
        else:
            elapsed = time.time() - self.start_time
            rate = self.completed_items / elapsed if elapsed > 0 else 0

            print(
                f"[Progress] {self.completed_items} completed | "
                f"Rate: {rate:.2f} items/sec"
            )

    def get_progress(self) -> dict:
        """
        Get current progress statistics.

        Returns:
            Dictionary with progress statistics
        """
        with self._lock:
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
        with self._lock:
            elapsed = time.time() - self.start_time
            rate = self.completed_items / elapsed if elapsed > 0 else 0

            if self.total_items > 0:
                percentage = (self.completed_items / self.total_items) * 100
                print(
                    f"[Progress] Complete: {self.completed_items}/{self.total_items} "
                    f"({percentage:.1f}%) | "
                    f"Total time: {elapsed:.2f}s | "
                    f"Avg rate: {rate:.2f} items/sec"
                )
            else:
                print(
                    f"[Progress] Complete: {self.completed_items} items | "
                    f"Total time: {elapsed:.2f}s | "
                    f"Avg rate: {rate:.2f} items/sec"
                )
