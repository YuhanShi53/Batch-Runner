"""
Chunked loader mixin for distributed processing.

This mixin enables data loaders to process only a specific chunk of data,
allowing distributed processing across multiple machines to reduce CPU load.

Configuration:
    num_chunks: Total number of chunks to split the data into (default: 1)
    chunk_index: Which chunk this process should handle (0-indexed, default: 0)

Example usage:
    # Split data into 4 chunks, process chunk 1 (second chunk)
    loader:
        class: JSONLDataLoader
        params:
            file_path: data.jsonl
            num_chunks: 4
            chunk_index: 1

    # On different machines:
    # Machine 1: chunk_index: 0
    # Machine 2: chunk_index: 1
    # Machine 3: chunk_index: 2
    # Machine 4: chunk_index: 3
"""
import math
import logging
from typing import Iterator, Any


logger = logging.getLogger(__name__)


class ChunkedLoaderMixin:
    """
    Mixin to enable chunked/distributed processing of data.

    This mixin allows a loader to process only a specific portion of the data
    by calculating which items belong to the current chunk based on:
    - num_chunks: Total number of chunks to split data into
    - chunk_index: Which chunk to process (0-indexed)

    The chunking is done deterministically using modulo arithmetic:
    - item_index % num_chunks == chunk_index

    This ensures that:
    - All chunks process roughly equal amounts of data
    - No item is processed by more than one chunk
    - The distribution is deterministic across runs

    Usage:
        class MyLoader(ChunkedLoaderMixin, DataLoader):
            def __init__(self, config):
                super().__init__(config)
                self._initialize_chunking()

            def load(self):
                for idx, item in enumerate(self.data):
                    if self._should_process_item(idx):
                        yield self.process_item(item)

    Configuration:
        num_chunks: Total number of chunks (default: 1, meaning no chunking)
        chunk_index: Which chunk to process, 0-indexed (default: 0)
                    Must be less than num_chunks
    """

    def _initialize_chunking(self):
        """
        Initialize chunking configuration from config.

        Validates that chunk_index is within valid range.
        Logs chunking information.
        """
        self.num_chunks = self.config.get('num_chunks', 1)
        self.chunk_index = self.config.get('chunk_index', 0)

        # Validate configuration
        if self.num_chunks < 1:
            raise ValueError(f"num_chunks must be >= 1, got {self.num_chunks}")

        if self.chunk_index < 0 or self.chunk_index >= self.num_chunks:
            raise ValueError(
                f"chunk_index must be in range [0, {self.num_chunks - 1}], "
                f"got {self.chunk_index}"
            )

        # Log chunking info
        if self.num_chunks > 1:
            logger.info(
                f"Chunked processing enabled: "
                f"processing chunk {self.chunk_index + 1}/{self.num_chunks} "
                f"(approximately {100.0 / self.num_chunks:.1f}% of data)"
            )

    def _should_process_item(self, item_index: int) -> bool:
        """
        Determine if an item should be processed based on chunk configuration.

        Uses modulo arithmetic for deterministic distribution:
        - Items are assigned to chunks using: item_index % num_chunks
        - This ensures even distribution across chunks
        - Same item always goes to same chunk

        Args:
            item_index: Zero-based index of the item in the data stream

        Returns:
            True if this item belongs to the current chunk, False otherwise

        Examples:
            # With num_chunks=3, chunk_index=1:
            # item_index=0 -> 0%3=0 -> skip
            # item_index=1 -> 1%3=1 -> process
            # item_index=2 -> 2%3=2 -> skip
            # item_index=3 -> 3%3=0 -> skip
            # item_index=4 -> 4%3=1 -> process
        """
        if self.num_chunks <= 1:
            # No chunking, process all items
            return True

        return item_index % self.num_chunks == self.chunk_index

    def _get_chunk_slice(self, total_items: int) -> tuple:
        """
        Calculate the start and end indices for this chunk.

        This is useful when you need to know the range of items
        that will be processed (e.g., for progress reporting).

        Args:
            total_items: Total number of items in the dataset

        Returns:
            Tuple of (start_index, end_index) for this chunk

        Note:
            This uses ceiling division to ensure all items are assigned
            to some chunk, even if total_items is not evenly divisible
            by num_chunks.
        """
        if self.num_chunks <= 1:
            return (0, total_items)

        # Calculate chunk size (round up to include all items)
        chunk_size = math.ceil(total_items / self.num_chunks)

        start = self.chunk_index * chunk_size
        end = min(start + chunk_size, total_items)

        return (start, end)

    def _estimate_chunk_size(self, total_items: int) -> int:
        """
        Estimate the number of items this chunk will process.

        Useful for logging and progress tracking.

        Args:
            total_items: Total number of items in the dataset

        Returns:
            Estimated number of items for this chunk
        """
        if self.num_chunks <= 1:
            return total_items

        # Base chunk size
        base_size = total_items // self.num_chunks

        # First (total_items % num_chunks) chunks get one extra item
        remainder = total_items % self.num_chunks

        if self.chunk_index < remainder:
            return base_size + 1
        else:
            return base_size
