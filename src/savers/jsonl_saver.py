"""
JSONL file result saver implementation.

Saves inference results to a JSONL file.
Each result is written as a separate JSON object on its own line.
"""
import threading
from pathlib import Path

from .base import ResultSaver, SaveResult
from .jsonl_mixin import JSONLSaverMixin


class JSONLResultSaver(JSONLSaverMixin, ResultSaver):
    """
    Save inference results to a JSONL file.

    Each result is written as a separate JSON object on one line.
    Thread-safe for concurrent writes.

    Configuration:
        output_path: Path to output JSONL file
        append: Whether to append to existing file (default: true)

    Customization:
        Override methods from JSONLSaverMixin to customize output:
        - format_result(): Customize the output dictionary structure
        - serialize_output(): Customize JSON serialization

    Example:
        class MySaver(JSONLResultSaver):
            def format_result(self, result):
                content = result.model_output['choices'][0]['message']['content']
                return {
                    "id": result.request_id,
                    "response": content
                }
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
        Uses the mixin's process_result_to_line template method.
        """
        # Use the mixin's template method for processing
        line = self.process_result_to_line(result)

        with self._lock:
            self.file.write(line + '\n')
            self.file.flush()  # Ensure data is written immediately

    def cleanup(self):
        """Close the file."""
        with self._lock:
            if hasattr(self, 'file') and not self.file.closed:
                self.file.close()
