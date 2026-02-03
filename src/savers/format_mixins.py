"""
Format-specific streaming mixins for common file formats.

These mixins provide ready-to-use streaming implementations for JSON,
JSONL, CSV, and directory-based output. They integrate with the
base streaming mixins to provide a complete solution for each format.
"""
import json
import csv
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
import threading
from datetime import datetime

from .base import SaveResult
from .streaming_mixin import (
    StreamingSaverMixin,
    OutputFormatterMixin,
    MultimodalOutputMixin,
)
from .jsonl_mixin import JSONLSaverMixin


logger = logging.getLogger(__name__)


# ===== JSON Streaming Mixin =====

class JSONStreamingMixin(StreamingSaverMixin, OutputFormatterMixin):
    """
    Streaming mixin for JSON file output.

    Writes results to a JSON file with immediate or batched flushing.

    Configuration:
        output_path: Path to output JSON file
        streaming: Enable streaming mode (default: True)
        batch_size: Number of results to buffer before writing (default: 100)
        pretty_print: Format JSON with indentation (default: True)

    Usage:
        class MyJSONSaver(JSONStreamingMixin, ResultSaver):
            pass  # Inherits all streaming logic
    """

    def _initialize_streaming(self):
        """Initialize JSON-specific streaming configuration."""
        super()._initialize_streaming()
        self.output_path = Path(self.config['output_path'])
        self.batch_size = self.config.get('batch_size', 100)
        self.pretty = self.config.get('pretty_print', True)

        # Create output directory
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # For batch mode
        self._batch_buffer = []

    def _get_output_path(self, result: SaveResult) -> Path:
        """Return the output file path."""
        return self.output_path

    def _format_result(self, result: SaveResult) -> Dict[str, Any]:
        """Format result using OutputFormatterMixin."""
        return self.format_output(result)

    def _write_result(self, output_path: Path, formatted_data: Dict[str, Any]) -> None:
        """Write result to file."""
        with self._lock:
            self._batch_buffer.append(formatted_data)

            # Write to disk if batch size reached
            if len(self._batch_buffer) >= self.batch_size:
                self._flush_to_disk()

    def _flush_to_disk(self) -> None:
        """Flush buffered results to disk."""
        if not self._batch_buffer:
            return

        # Read existing data if file exists
        existing_data = []
        if self.output_path.exists():
            try:
                with open(self.output_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                existing_data = []

        # Append new results
        existing_data.extend(self._batch_buffer)

        # Write back to file
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(
                existing_data,
                f,
                indent=2 if self.pretty else None,
                ensure_ascii=False
            )

        self._batch_buffer = []

    def cleanup(self):
        """Flush any remaining results."""
        with self._lock:
            self._flush_to_disk()


# ===== JSONL Streaming Mixin =====

class JSONLStreamingMixin(JSONLSaverMixin, StreamingSaverMixin):
    """
    Streaming mixin for JSONL file output.

    Writes results to a JSONL file with immediate line-by-line output.

    Configuration:
        output_path: Path to output JSONL file
        streaming: Enable streaming mode (default: True)
        append: Append to existing file (default: True)
        immediate_flush: Flush after each write (default: True)

    Usage:
        class MyJSONLSaver(JSONLStreamingMixin, ResultSaver):
            pass  # Inherits all streaming and JSONL logic
    """

    def _initialize_streaming(self):
        """Initialize JSONL-specific streaming configuration."""
        super()._initialize_streaming()
        self.output_path = Path(self.config['output_path'])
        self.append = self.config.get('append', True)

        # Create output directory
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Open file in append mode
        mode = 'a' if self.append else 'w'
        self.file = open(self.output_path, mode, encoding='utf-8')

    def _get_output_path(self, result: SaveResult) -> Path:
        """Return the output file path."""
        return self.output_path

    def _format_result(self, result: SaveResult) -> Dict[str, Any]:
        """Format result using JSONLSaverMixin."""
        # Use the mixin's format_result method
        return self.format_result(result)

    def _write_result(self, output_path: Path, formatted_data: Dict[str, Any]) -> None:
        """Write result as a JSONL line."""
        # Use the mixin's serialize_output method
        line = self.serialize_output(formatted_data)

        with self._lock:
            self.file.write(line + '\n')
            if self.immediate_flush:
                self.file.flush()

    def cleanup(self):
        """Close the file."""
        with self._lock:
            if hasattr(self, 'file') and not self.file.closed:
                self.file.close()


# ===== CSV Streaming Mixin =====

class CSVStreamingMixin(StreamingSaverMixin, OutputFormatterMixin):
    """
    Streaming mixin for CSV file output.

    Writes results to a CSV file with automatic field detection.

    Configuration:
        output_path: Path to output CSV file
        streaming: Enable streaming mode (default: True)
        fields: List of field names to output (auto-detected if not specified)
        include_header: Include header row (default: True)

    Usage:
        class MyCSVSaver(CSVStreamingMixin, ResultSaver):
            pass  # Inherits all streaming and CSV logic
    """

    def _initialize_streaming(self):
        """Initialize CSV-specific streaming configuration."""
        super()._initialize_streaming()
        self.output_path = Path(self.config['output_path'])
        self.fields = self.config.get('fields', None)
        self.include_header = self.config.get('include_header', True)
        self._header_written = False

        # Create output directory
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_output_path(self, result: SaveResult) -> Path:
        """Return the output file path."""
        return self.output_path

    def _format_result(self, result: SaveResult) -> Dict[str, Any]:
        """Format result for CSV output."""
        # If fields specified, extract only those fields
        if self.fields:
            formatted = self.format_output(result)
            return {k: formatted.get(k, '') for k in self.fields}
        return self.format_output(result)

    def _write_result(self, output_path: Path, formatted_data: Dict[str, Any]) -> None:
        """Write result as a CSV row."""
        with self._lock:
            # Auto-detect fields from first result if not specified
            if self.fields is None:
                self.fields = list(formatted_data.keys())

            # Write header if needed
            if self.include_header and not self._header_written:
                with open(output_path, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self.fields)
                    writer.writeheader()
                self._header_written = True

            # Write data row
            with open(output_path, 'a', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fields)
                writer.writerow(formatted_data)


# ===== Directory Streaming Mixin =====

class DirectoryStreamingMixin(StreamingSaverMixin):
    """
    Streaming mixin for directory-based output.

    Organizes output files to mirror the input directory structure.

    Configuration:
        output_dir: Root output directory
        streaming: Enable streaming mode (default: True)
        output_file_pattern: Name pattern for output files (default: "result.jsonl")
        preserve_structure: Mirror input directory structure (default: True)

    Usage:
        class MyDirectorySaver(DirectoryStreamingMixin, JSONLStreamingMixin, ResultSaver):
            pass  # Inherits directory organization and JSONL writing
    """

    def _initialize_streaming(self):
        """Initialize directory-specific streaming configuration."""
        super()._initialize_streaming()
        self.output_dir = Path(self.config['output_dir'])
        self.output_file_pattern = self.config.get('output_file_pattern', 'result.jsonl')
        self.preserve_structure = self.config.get('preserve_structure', True)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track open files: {output_path: file_handle}
        self._files = {}

    def _get_output_path(self, result: SaveResult) -> Path:
        """
        Determine output path based on source information.

        Uses _source_file from additional_data to mirror input structure.
        """
        additional_data = result.additional_data or {}
        source_file = additional_data.get('_source_file', '')
        source_dir = additional_data.get('_source_dir', '')

        if not source_file or not self.preserve_structure:
            # Default to root output file
            return self.output_dir / self.output_file_pattern

        # Mirror the directory structure
        if source_dir:
            output_path = self.output_dir / source_dir / self.output_file_pattern
        else:
            output_path = self.output_dir / self.output_file_pattern

        return output_path

    def _get_file_handle(self, output_path: Path):
        """
        Get or create a file handle for the given output path.

        Thread-safe file handle management.
        """
        # Check if file is already open
        if output_path in self._files:
            return self._files[output_path]

        # Create parent directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Open file in append mode
        file_handle = open(output_path, 'a', encoding='utf-8')
        self._files[output_path] = file_handle

        return file_handle

    def _format_result(self, result: SaveResult) -> Dict[str, Any]:
        """
        Format result for output.

        This is a placeholder - subclasses should mix in a format-specific
        mixin (JSONLStreamingMixin, JSONStreamingMixin, etc.) that provides
        the actual _format_result implementation.
        """
        # If the class also has JSONLStreamingMixin, use its format_result
        if hasattr(self, 'format_result'):
            return self.format_result(result)
        return {
            'request_id': result.request_id,
            'model_output': result.model_output,
            'additional_data': result.additional_data,
            'timestamp': datetime.now().isoformat()
        }

    def _write_result(self, output_path: Path, formatted_data: Any) -> None:
        """
        Write formatted data to the appropriate file.

        Handles file management for directory-based output.
        """
        # If the class also has JSONLStreamingMixin, serialize as JSONL
        if hasattr(self, 'serialize_output'):
            line = self.serialize_output(formatted_data)
            file_handle = self._get_file_handle(output_path)
            with self._lock:
                file_handle.write(line + '\n')
                if self.immediate_flush:
                    file_handle.flush()
        else:
            # Default: write as JSON
            file_handle = self._get_file_handle(output_path)
            with self._lock:
                file_handle.write(json.dumps(formatted_data, ensure_ascii=False) + '\n')
                if self.immediate_flush:
                    file_handle.flush()

    def cleanup(self):
        """Close all open files."""
        with self._lock:
            for file_handle in self._files.values():
                if not file_handle.closed:
                    file_handle.close()
            self._files.clear()


# ===== Convenience classes combining mixins =====

class StreamingJSONLSaver(JSONLStreamingMixin, ResultSaver):
    """
    Complete streaming JSONL saver implementation.

    This class combines all necessary mixins for a fully functional
    streaming JSONL saver. Subclass this to customize behavior.

    Usage:
        class MyCustomSaver(StreamingJSONLSaver):
            def format_output(self, result):
                # Custom output format
                content = result.model_output['choices'][0]['message']['content']
                return {"id": result.request_id, "response": content}
    """
    pass


class StreamingDirectoryJSONLSaver(DirectoryStreamingMixin, JSONLStreamingMixin, ResultSaver):
    """
    Complete streaming directory JSONL saver implementation.

    This class combines all necessary mixins for a fully functional
    streaming saver that organizes output to mirror input directory structure.
    Subclass this to customize behavior.

    Usage:
        class MyCustomSaver(StreamingDirectoryJSONLSaver):
            def format_output(self, result):
                # Custom output format
                return self.format_output(result)
    """
    pass


class StreamingCSVSaver(CSVStreamingMixin, ResultSaver):
    """
    Complete streaming CSV saver implementation.

    This class combines all necessary mixins for a fully functional
    streaming CSV saver. Subclass this to customize behavior.

    Usage:
        class MyCustomSaver(StreamingCSVSaver):
            def format_output(self, result):
                # Flatten nested structure for CSV
                content = result.model_output['choices'][0]['message']['content']
                return {
                    "id": result.request_id,
                    "response": content,
                    "tokens": result.model_output.get('usage', {}).get('total_tokens', 0)
                }
    """
    pass
