"""
JSON file result saver implementation.

Saves inference results to a JSON file with batch writing support.
"""
import json
import threading
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime

from .base import ResultSaver, SaveResult


class JSONResultSaver(ResultSaver):
    """
    Save inference results to a JSON file.

    Results are buffered in memory and written to disk in batches for efficiency.

    Configuration:
        output_path: Path to output JSON file
        batch_size: Number of results to buffer before writing (default: 100)
        pretty_print: Whether to format JSON with indentation (default: true)
    """

    def _initialize(self):
        """Initialize JSON file saver."""
        self.output_path = Path(self.config['output_path'])
        self.batch_size = self.config.get('batch_size', 100)
        self.pretty = self.config.get('pretty_print', True)

        # Create output directory if needed
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.results = []
        self._lock = threading.Lock()
        self._write_count = 0

    def save(self, result: SaveResult):
        """
        Save a single result to memory buffer.

        Results are written to disk in batches for efficiency.
        Thread-safe for concurrent writes.
        """
        output_data = {
            'request_id': result.request_id,
            'model_output': result.model_output,
            'additional_data': result.additional_data,
            'timestamp': datetime.now().isoformat()
        }

        if result.error:
            output_data['error'] = result.error

        with self._lock:
            self.results.append(output_data)

            # Write to disk if batch size reached
            if len(self.results) >= self.batch_size:
                self._flush()

    def _flush(self):
        """Write buffered results to disk."""
        if not self.results:
            return

        # Read existing data if file exists
        existing_data = []
        if self.output_path.exists():
            with open(self.output_path, 'r', encoding='utf-8') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = []

        # Append new results
        existing_data.extend(self.results)

        # Write back to file
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2 if self.pretty else None, ensure_ascii=False)

        self.results = []
        self._write_count += 1

    def cleanup(self):
        """Flush any remaining results and close file."""
        with self._lock:
            self._flush()
