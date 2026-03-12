"""
JSON file result saver implementation.

Saves inference results to a JSON file with batch writing support.
Supports chunked output via ChunkedSaverMixin.
"""
import json
import logging
from typing import Dict, Any
from pathlib import Path

from .base import ResultSaver, SaveResult
from .streaming_mixin import BatchWriterMixin, OutputFormatterMixin
from .chunked_mixin import ChunkedSaverMixin


logger = logging.getLogger(__name__)


class JSONResultSaver(ChunkedSaverMixin, BatchWriterMixin, OutputFormatterMixin, ResultSaver):
    """
    Save inference results to a JSON file.

    Results are buffered in memory and written to disk in batches for efficiency.

    Configuration:
        output_path: Path to output JSON file
        batch_size: Number of results to buffer before writing (default: 100)
        pretty_print: Whether to format JSON with indentation (default: true)
        num_chunks: Total number of chunks (default: 1)
        chunk_index: Which chunk to process (0-indexed, default: 0)
        add_chunk_suffix: Auto-add chunk suffix to output path (default: true)

    Chunked Output:
        When num_chunks > 1, automatically appends _chunk_{index} to output filename.
        Example: output_path: results/data.json -> results/data_chunk_0.json

    Customization:
        Override methods from OutputFormatterMixin to customize output:
        - format_output(): Customize the output dictionary structure
        - extract_content(): Extract main content from model output
        - extract_usage(): Extract token usage information
    """

    def _initialize(self):
        """Initialize JSON file saver."""
        # Initialize chunking from ChunkedSaverMixin
        self._initialize_chunking()

        # Get output path with chunk suffix automatically applied
        self.output_path = self._get_chunked_output_path()

        self.pretty = self.config.get('pretty_print', True)
        self.output_projection = self.config.get('output_projection', 'full')
        self.output_fields = self.config.get('output_fields')
        self.include_timestamp = self.config.get(
            'include_timestamp',
            self.output_projection == 'full'
        )

        # Create output directory if needed
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize batch writing from BatchWriterMixin
        self._initialize_batch(default_batch_size=100)

    def _format_result(self, result: SaveResult) -> Dict[str, Any]:
        """
        Format a result for output.

        Uses the OutputFormatterMixin's format_output method.

        Args:
            result: The SaveResult to format

        Returns:
            Dictionary ready for JSON serialization
        """
        return self.format_output(result)

    def save(self, result: SaveResult):
        """
        Save a single result to memory buffer.

        Results are written to disk in batches for efficiency.
        Thread-safe for concurrent writes.
        """
        formatted_data = self._format_result(result)
        self._add_to_batch(formatted_data)

    def save_batch(self, results):
        """Buffer multiple formatted results at once."""
        if not results:
            return

        formatted_batch = [self._format_result(result) for result in results]
        should_flush = False
        with self._batch_lock:
            self._batch_buffer.extend(formatted_batch)
            should_flush = len(self._batch_buffer) >= self.batch_size

        if should_flush:
            self._flush_batch()

    def _flush_batch(self) -> None:
        """
        Flush buffered batch data to storage.

        Reads existing data, appends new results, and writes back to file.
        Thread-safe.
        """
        batch = self._get_batch()

        if not batch:
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
        existing_data.extend(batch)

        # Write back to file
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2 if self.pretty else None, ensure_ascii=False)

    def cleanup(self):
        """Flush any remaining results and close file."""
        self._flush_batch()

    def _load_completed_ids(self) -> set:
        """
        Load completed request_ids from existing JSON output file.

        Returns:
            Set of request_id strings
        """
        completed_ids = set()

        if not self.output_path.exists():
            logger.info(f"Output file {self.output_path} does not exist, starting fresh")
            return completed_ids

        logger.info(f"Loading completed request_ids from {self.output_path}")

        try:
            with open(self.output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, list):
                for item in data:
                    if 'request_id' in item:
                        completed_ids.add(item['request_id'])

            logger.info(f"Loaded {len(completed_ids)} completed request_ids")

        except Exception as e:
            logger.error(f"Error loading completed IDs: {e}")

        return completed_ids

    def get_resume_store_path(self) -> str:
        """Return the directory used for bitmap resume state."""
        return str(Path(f"{self.output_path}.resume"))
