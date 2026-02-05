"""
Directory JSONL result saver implementation.

Saves inference results to output files while preserving the input
directory structure. Results from the same source file are grouped
into corresponding output files.
"""
import json
import logging
from pathlib import Path

from .base import ResultSaver, SaveResult
from .jsonl_mixin import JSONLSaverMixin
from .streaming_mixin import StreamingSaverMixin


logger = logging.getLogger(__name__)


class DirectoryJSONLResultSaver(StreamingSaverMixin, JSONLSaverMixin, ResultSaver):
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

    Customization:
        Override methods from JSONLSaverMixin to customize output:
        - format_result(): Customize the output dictionary structure
        - serialize_output(): Customize JSON serialization
    """

    def _initialize(self):
        """Initialize directory JSONL saver."""
        self.output_dir = Path(self.config['output_dir'])
        self.output_file_pattern = self.config.get('output_file_pattern', 'result.jsonl')
        self.preserve_structure = self.config.get('preserve_structure', True)
        self.output_filename = self.config.get('output_filename', None)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize streaming configuration from StreamingSaverMixin
        self._initialize_streaming()

        # Track open files: {output_path: file_handle}
        self._files = {}

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

    def _format_result(self, result: SaveResult) -> str:
        """
        Format a result for JSONL output.

        Uses the JSONLSaverMixin's process_result_to_line template method.

        Args:
            result: The SaveResult to format

        Returns:
            JSONL line string (without newline)
        """
        return self.process_result_to_line(result)

    def _write_result(self, output_path: Path, formatted_data: str) -> None:
        """
        Write formatted data to the output path.

        Thread-safe file handle management with file pooling.

        Args:
            output_path: Path where data should be written
            formatted_data: JSONL line string to write
        """
        # Get or create file handle
        if output_path not in self._files:
            # Create parent directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # Open file in append mode
            self._files[output_path] = open(output_path, 'a', encoding='utf-8')

        file_handle = self._files[output_path]
        file_handle.write(formatted_data + '\n')

    def _flush(self, output_path: Path):
        """
        Flush output to disk.

        Args:
            output_path: Path that needs flushing
        """
        if output_path in self._files:
            self._files[output_path].flush()

    def save(self, result: SaveResult):
        """
        Save a single result to the appropriate JSONL file.

        Thread-safe for concurrent writes. Results from the same source
        file are grouped together in the corresponding output location.
        Uses the StreamingSaverMixin template method.
        """
        self._stream_save(result)

    def cleanup(self):
        """Close all open files."""
        with self._lock:
            for file_handle in self._files.values():
                if not file_handle.closed:
                    file_handle.close()
            self._files.clear()

    def _load_completed_ids(self) -> set:
        """
        Load completed request_ids from all JSONL files in the output directory.

        Scans the output directory recursively and loads request_ids from
        all .jsonl files. Handles the distributed file structure properly.

        Returns:
            Set of base request_id strings (without _rollout_N suffix)
        """
        completed_ids = set()

        # Check if output directory exists
        if not self.output_dir.exists():
            logger.info(f"Output directory {self.output_dir} does not exist, starting fresh")
            return completed_ids

        logger.info(f"Loading completed request_ids from {self.output_dir}")

        try:
            # Find all .jsonl files in the output directory
            jsonl_files = list(self.output_dir.rglob('*.jsonl'))

            if not jsonl_files:
                logger.info(f"No .jsonl files found in {self.output_dir}")
                return completed_ids

            for jsonl_file in jsonl_files:
                logger.debug(f"Scanning {jsonl_file}")
                try:
                    with open(jsonl_file, 'r', encoding='utf-8') as f:
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
                                logger.warning(f"Skipping invalid JSON in {jsonl_file}:{line_num}: {e}")
                                continue

                except Exception as e:
                    logger.warning(f"Error reading {jsonl_file}: {e}")
                    continue

            logger.info(f"Loaded {len(completed_ids)} completed request_ids from {len(jsonl_files)} files")

        except Exception as e:
            logger.error(f"Error loading completed IDs: {e}")

        return completed_ids
