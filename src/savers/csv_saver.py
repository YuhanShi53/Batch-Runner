"""
CSV file result saver implementation.

Saves inference results to a CSV file.
"""
import csv
import logging
import threading
from typing import Dict, Any
from pathlib import Path

from .base import ResultSaver, SaveResult


logger = logging.getLogger(__name__)


class CSVResultSaver(ResultSaver):
    """
    Save inference results to a CSV file.

    Configuration:
        output_path: Path to output CSV file
        fields: Fields to extract from model output (default: extracts common fields)
        encoding: File encoding (default: "utf-8")
    """

    def _initialize(self):
        """Initialize CSV file saver."""
        self.output_path = Path(self.config['output_path'])
        self.encoding = self.config.get('encoding', 'utf-8')
        self.fields = self.config.get('fields', None)

        # Create output directory if needed
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._file = None
        self._writer = None
        self._initialized = False

    def _ensure_initialized(self):
        """Ensure file and writer are initialized (lazy initialization)."""
        if self._initialized:
            return

        # Determine fields from model output structure
        default_fields = ['request_id', 'content', 'finish_reason', 'total_tokens']
        fieldnames = self.fields if self.fields else default_fields

        self._file = open(self.output_path, 'w', encoding=self.encoding, newline='')
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()
        self._initialized = True

    def save(self, result: SaveResult):
        """
        Save a single result to CSV file.

        Thread-safe for concurrent writes.
        """
        with self._lock:
            self._ensure_initialized()

            # Extract content from model output
            content = ''
            if result.model_output:
                choices = result.model_output.get('choices', [])
                if choices and len(choices) > 0:
                    message = choices[0].get('message', {})
                    content = message.get('content', '')

            # Extract usage info
            usage = result.model_output.get('usage', {}) if result.model_output else {}

            # Build row data
            row = {
                'request_id': result.request_id,
                'content': content,
                'finish_reason': choices[0].get('finish_reason', '') if choices else '',
                'total_tokens': usage.get('total_tokens', ''),
            }

            # Add custom fields if specified
            if self.fields:
                row_data = {}
                for field in self.fields:
                    if field == 'request_id':
                        row_data[field] = result.request_id
                    elif field == 'content':
                        row_data[field] = content
                    else:
                        # Try to extract from model_output
                        row_data[field] = str(result.model_output.get(field, '') if result.model_output else '')
                row = row_data

            self._writer.writerow(row)
            self._file.flush()

    def cleanup(self):
        """Close file."""
        with self._lock:
            if self._file:
                self._file.close()
                self._file = None
                self._initialized = False

    def _load_completed_ids(self) -> set:
        """
        Load completed request_ids from existing CSV output file.

        Returns:
            Set of request_id strings
        """
        completed_ids = set()

        if not self.output_path.exists():
            logger.info(f"Output file {self.output_path} does not exist, starting fresh")
            return completed_ids

        logger.info(f"Loading completed request_ids from {self.output_path}")

        try:
            with open(self.output_path, 'r', encoding=self.encoding) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'request_id' in row:
                        completed_ids.add(row['request_id'])

            logger.info(f"Loaded {len(completed_ids)} completed request_ids")

        except Exception as e:
            logger.error(f"Error loading completed IDs: {e}")

        return completed_ids
