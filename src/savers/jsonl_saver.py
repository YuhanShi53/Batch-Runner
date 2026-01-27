"""
JSONL file result saver implementation.

Saves inference results to a JSONL file.
Each result is written as a separate JSON object on its own line.
"""
import json
import threading
from typing import Dict, Any
from pathlib import Path
from datetime import datetime

from .base import ResultSaver, SaveResult


class JSONLResultSaver(ResultSaver):
    """
    Save inference results to a JSONL file.

    Each result is written as a separate JSON object on one line.
    Thread-safe for concurrent writes.

    Configuration:
        output_path: Path to output JSONL file
        append: Whether to append to existing file (default: true)
    """

    def _initialize(self):
        """Initialize JSONL file saver."""
        self.output_path = Path(self.config['output_path'])
        self.append = self.config.get('append', True)

        # Create output directory if needed
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Open file in append mode
        mode = 'a' if self.append else 'w'
        self.file = open(self.output_path, mode, encoding='utf-8')
        self._lock = threading.Lock()

    def save(self, result: SaveResult):
        """
        Save a single result to JSONL file.

        Each result is written as a JSON object on a separate line.
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

        # Write as a single line JSON
        line = json.dumps(output_data, ensure_ascii=False)

        with self._lock:
            self.file.write(line + '\n')
            self.file.flush()  # Ensure data is written immediately

    def cleanup(self):
        """Close the file."""
        with self._lock:
            if hasattr(self, 'file') and not self.file.closed:
                self.file.close()
