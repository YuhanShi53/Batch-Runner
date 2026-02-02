"""
Directory JSONL result saver implementation.

Saves inference results to output files while preserving the input
directory structure. Results from the same source file are grouped
into corresponding output files.
"""
import json
import threading
from typing import Dict, Any
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from .base import ResultSaver, SaveResult


class DirectoryJSONLResultSaver(ResultSaver):
    """
    Save inference results to JSONL files, preserving input directory structure.

    Each result is written to an output file that mirrors the input directory
    structure. Results from the same source conv.jsonl file are grouped together
    in the corresponding output location.

    Configuration:
        output_dir: Root output directory
        output_file_pattern: Name pattern for output files (default: "result.jsonl")
        preserve_structure: Whether to mirror input directory structure (default: true)
        output_filename: Optional custom filename (overrides output_file_pattern)
                        Use "{source}" to reference source filename without extension

    Output filename behavior:
        - Default: Creates "result.jsonl" in each directory
        - With output_filename="{source}.out": Converts "conv.jsonl" to "conv.jsonl.out"
        - With output_filename="results_{source}.jsonl": Converts to "results_conv.jsonl"
    """

    def _initialize(self):
        """Initialize directory JSONL saver."""
        self.output_dir = Path(self.config['output_dir'])
        self.output_file_pattern = self.config.get('output_file_pattern', 'result.jsonl')
        self.preserve_structure = self.config.get('preserve_structure', True)
        self.output_filename = self.config.get('output_filename', None)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track open files: {output_path: file_handle}
        self._files = {}
        self._lock = threading.Lock()

    def _get_output_path(self, result: SaveResult) -> Path:
        """
        Determine the output file path for a result.

        Uses _source_file from additional_data to mirror input structure.
        """
        # Extract source information
        additional_data = result.additional_data or {}
        source_file = additional_data.get('_source_file', '')
        source_dir = additional_data.get('_source_dir', '')

        if not source_file or not self.preserve_structure:
            # Default to root output file
            return self.output_dir / self.output_file_pattern

        # Determine output filename
        if self.output_filename:
            # Custom filename with optional {source} placeholder
            source_name = Path(source_file).stem  # filename without extension
            output_filename = self.output_filename.format(source=source_name)
        else:
            # Use default pattern
            output_filename = self.output_file_pattern

        # Mirror the directory structure
        if source_dir:
            output_path = self.output_dir / source_dir / output_filename
        else:
            output_path = self.output_dir / output_filename

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

    def save(self, result: SaveResult):
        """
        Save a single result to the appropriate JSONL file.

        Thread-safe for concurrent writes. Results from the same source
        file are grouped together in the corresponding output location.
        """
        output_path = self._get_output_path(result)

        # Prepare output data
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
            file_handle = self._get_file_handle(output_path)
            file_handle.write(line + '\n')
            file_handle.flush()  # Ensure data is written immediately

    def cleanup(self):
        """Close all open files."""
        with self._lock:
            for file_handle in self._files.values():
                if not file_handle.closed:
                    file_handle.close()
            self._files.clear()
