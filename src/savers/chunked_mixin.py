"""
Chunked saver mixin for distributed processing.

This mixin enables result savers to automatically append chunk information
to output filenames when processing distributed data.

This ensures that different processes/chunks write to different files,
avoiding file write conflicts.

Configuration:
    num_chunks: Total number of chunks (default: 1)
    chunk_index: Which chunk this process is handling (default: 0)
    add_chunk_suffix: Whether to automatically add chunk suffix to output path (default: true)

Example usage:
    saver:
        class: JSONLResultSaver
        params:
            output_path: results/output.jsonl
            num_chunks: 4
            chunk_index: 1
            # Actual output will be: results/output_chunk_1.jsonl
"""
from pathlib import Path
import logging


logger = logging.getLogger(__name__)


class ChunkedSaverMixin:
    """
    Mixin to automatically add chunk suffix to output paths.

    This mixin modifies the output_path to include chunk information,
    preventing file write conflicts when multiple processes run in parallel.

    The suffix format is: _chunk_{index} before the file extension.
    For example:
    - results/output.jsonl -> results/output_chunk_0.jsonl
    - data/results.json -> data/results_chunk_1.json

    Usage:
        class MySaver(ChunkedSaverMixin, ResultSaver):
            def _initialize(self):
                # Get the chunked output path
                self.output_path = self._get_chunked_output_path()

    Configuration:
        num_chunks: Total number of chunks (default: 1, meaning no chunking)
        chunk_index: Which chunk to process (0-indexed, default: 0)
        add_chunk_suffix: Whether to add suffix (default: True)
                          Set to False to disable automatic suffix addition
    """

    def _initialize_chunking(self):
        """
        Initialize chunking configuration from config.

        Stores configuration for later use by _get_chunked_output_path().
        """
        self.num_chunks = self.config.get('num_chunks', 1)
        self.chunk_index = self.config.get('chunk_index', 0)
        self.add_chunk_suffix = self.config.get('add_chunk_suffix', True)

        # Validate configuration
        if self.num_chunks < 1:
            raise ValueError(f"num_chunks must be >= 1, got {self.num_chunks}")

        if self.chunk_index < 0 or self.chunk_index >= self.num_chunks:
            raise ValueError(
                f"chunk_index must be in range [0, {self.num_chunks - 1}], "
                f"got {self.chunk_index}"
            )

        # Log chunking info
        if self.num_chunks > 1 and self.add_chunk_suffix:
            logger.info(
                f"Chunked output: chunk {self.chunk_index + 1}/{self.num_chunks}"
            )

    def _get_chunked_output_path(self, base_path: str = None) -> Path:
        """
        Get the output path with chunk suffix added.

        If chunking is enabled (num_chunks > 1) and add_chunk_suffix is True,
        automatically adds _chunk_{index} suffix before the file extension.

        Args:
            base_path: Original output path. If None, uses config['output_path']

        Returns:
            Path object with chunk suffix added if chunking is enabled

        Examples:
            # With num_chunks=4, chunk_index=1:
            _get_chunked_output_path("results/output.jsonl")
            # Returns: Path("results/output_chunk_1.jsonl")

            # With num_chunks=1 (no chunking):
            _get_chunked_output_path("results/output.jsonl")
            # Returns: Path("results/output.jsonl")

            # With add_chunk_suffix=False:
            _get_chunked_output_path("results/output.jsonl")
            # Returns: Path("results/output.jsonl") (unchanged)
        """
        if base_path is None:
            base_path = self.config.get('output_path', 'output.jsonl')

        path = Path(base_path)

        # Don't add suffix if:
        # - Chunking is disabled (num_chunks <= 1)
        # - Suffix addition is disabled
        if self.num_chunks <= 1 or not self.add_chunk_suffix:
            return path

        # Add chunk suffix: insert _chunk_{index} before extension
        stem = path.stem
        suffix = path.suffix
        chunked_stem = f"{stem}_chunk_{self.chunk_index}"
        chunked_path = path.with_name(f"{chunked_stem}{suffix}")

        logger.info(f"Output path with chunk suffix: {chunked_path}")

        return chunked_path
