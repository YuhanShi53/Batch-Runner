"""
JSONL file result saver implementation.

Saves inference results to a JSONL file.
Each result is written as a separate JSON object on its own line.
Supports streaming mode with immediate write-back.
"""
import threading
from pathlib import Path
import logging

from .base import ResultSaver, SaveResult
from .jsonl_mixin import JSONLSaverMixin


logger = logging.getLogger(__name__)


class JSONLResultSaver(JSONLSaverMixin, ResultSaver):
    """
    Save inference results to a JSONL file.

    Each result is written as a separate JSON object on one line.
    Thread-safe for concurrent writes. Streaming mode is enabled by default
    for immediate write-back and fault tolerance.

    Configuration:
        output_path: Path to output JSONL file
        append: Whether to append to existing file (default: true)
        streaming: Enable streaming mode (default: true)
        immediate_flush: Flush to disk after each write (default: true)

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
        self.streaming = self.config.get('streaming', True)
        self.immediate_flush = self.config.get('immediate_flush', True)

        # Create output directory if needed
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Open file in append mode
        mode = 'a' if self.append else 'w'
        self.file = open(self.output_path, mode, encoding='utf-8')
        self._lock = threading.Lock()

        if self.streaming:
            logger.info(f"Streaming mode enabled for {self.output_path}")

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
            if self.immediate_flush:
                self.file.flush()  # Ensure data is written immediately

    def cleanup(self):
        """Close the file."""
        with self._lock:
            if hasattr(self, 'file') and not self.file.closed:
                self.file.close()

    def _load_completed_ids(self) -> set:
        """
        Load completed request_ids from existing JSONL output file.

        Reads the output file line by line and extracts all request_ids.
        Handles corrupted lines gracefully by skipping them.

        Returns:
            Set of base request_id strings (without _rollout_N suffix)
        """
        import json

        completed_ids = set()

        # Check if output file exists
        if not self.output_path.exists():
            logger.info(f"Output file {self.output_path} does not exist, starting fresh")
            return completed_ids

        logger.info(f"Loading completed request_ids from {self.output_path}")

        try:
            with open(self.output_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue

                    try:
                        data = json.loads(line)
                        if 'request_id' in data:
                            # Strip rollout suffix to get base ID
                            base_id = data['request_id'].split('_rollout_')[0]
                            completed_ids.add(base_id)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                        continue

            logger.info(f"Loaded {len(completed_ids)} completed request_ids from {self.output_path}")

        except Exception as e:
            logger.error(f"Error loading completed IDs: {e}")
            # Return empty set on error - fail open to allow processing

        return completed_ids
