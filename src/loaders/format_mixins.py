"""
Format-specific streaming mixins for common file formats.

These mixins provide ready-to-use streaming implementations for JSON,
JSONL, CSV, and directory-based data sources. They integrate with the
base streaming mixins to provide a complete solution for each format.
"""
import json
import csv
from typing import Iterator, Dict, Any, Optional, List
from pathlib import Path
import logging

from .base import LoadResult
from .streaming_mixin import (
    StreamingLoaderMixin,
    MessagesBuilderMixin,
    PromptExtractorMixin,
)
from .jsonl_mixin import JSONLLoaderMixin


logger = logging.getLogger(__name__)


# ===== JSON Streaming Mixin =====

class JSONStreamingMixin(StreamingLoaderMixin, MessagesBuilderMixin):
    """
    Streaming mixin for JSON file data sources.

    Processes JSON files in a streaming fashion, yielding items one at a time.

    Configuration:
        file_path: Path to JSON file
        streaming: Enable streaming mode (default: True)

    The JSON file should contain an array of objects at the root level.

    Usage:
        class MyJSONLoader(JSONStreamingMixin, DataLoader):
            pass  # Inherits all streaming logic
    """

    def _initialize_streaming(self):
        """Initialize JSON-specific streaming configuration."""
        super()._initialize_streaming()
        self.file_path = Path(self.config['file_path'])
        self.prompt_field = self.config.get('prompt_field', 'prompt')
        self.id_field = self.config.get('id_field', 'id')

        if not self.file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {self.file_path}")

    def _discover_sources(self) -> List[Any]:
        """Return the JSON file as the single source."""
        return [self.file_path]

    def _process_source(self, source: Path) -> Iterator[LoadResult]:
        """
        Process the JSON file and yield LoadResult objects.

        Args:
            source: Path to JSON file
        """
        with open(source, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"JSON root must be a list of objects in {source}")

        for idx, item in enumerate(data, 1):
            prompt = self.extract_prompt_from_item(item)
            if prompt is None:
                continue

            request_id = str(item.get(self.id_field, f"req_{idx}"))

            # Extract additional data
            additional_data = {
                k: v for k, v in item.items()
                if k not in [self.prompt_field, self.id_field]
            }

            # Build messages
            messages = self.build_messages(prompt, additional_data)

            yield LoadResult(
                messages=messages,
                request_id=request_id,
                additional_data=additional_data or None
            )

    def extract_prompt_from_item(self, item: Dict[str, Any]) -> Optional[str]:
        """
        Extract prompt from a JSON item.

        Override this method for custom prompt extraction.

        Args:
            item: Dictionary representing a JSON object

        Returns:
            Prompt string or None
        """
        return item.get(self.prompt_field)


# ===== JSONL Streaming Mixin =====

class JSONLStreamingMixin(JSONLLoaderMixin, StreamingLoaderMixin):
    """
    Streaming mixin for JSONL file data sources.

    Processes JSONL files in a streaming fashion, yielding items line by line.

    Configuration:
        file_path: Path to JSONL file
        streaming: Enable streaming mode (default: True)
        prompt_field: Field name containing the prompt (default: "prompt")
        id_field: Field name containing the ID (default: "id")

    Usage:
        class MyJSONLLoader(JSONLStreamingMixin, DataLoader):
            pass  # Inherits all streaming and JSONL logic
    """

    def _initialize_streaming(self):
        """Initialize JSONL-specific streaming configuration."""
        super()._initialize_streaming()
        self.file_path = Path(self.config['file_path'])
        self.prompt_field = self.config.get('prompt_field', 'prompt')
        self.id_field = self.config.get('id_field', 'id')

        if not self.file_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {self.file_path}")

    def _discover_sources(self) -> List[Any]:
        """Return the JSONL file as the single source."""
        return [self.file_path]

    def _process_source(self, source: Path) -> Iterator[LoadResult]:
        """
        Process the JSONL file and yield LoadResult objects.

        Args:
            source: Path to JSONL file
        """
        with open(source, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                result = self.process_line_to_load_result(
                    line=line,
                    line_num=line_num,
                    source=str(source),
                    default_id=f"{source.name}:{line_num}"
                )

                if result is not None:
                    yield result


# ===== CSV Streaming Mixin =====

class CSVStreamingMixin(StreamingLoaderMixin, MessagesBuilderMixin):
    """
    Streaming mixin for CSV file data sources.

    Processes CSV files in a streaming fashion, yielding items row by row.

    Configuration:
        file_path: Path to CSV file
        streaming: Enable streaming mode (default: True)
        prompt_column: Column name containing the prompt (default: "prompt")
        id_column: Column name containing the ID (default: "id")

    Usage:
        class MyCSVLoader(CSVStreamingMixin, DataLoader):
            pass  # Inherits all streaming and CSV logic
    """

    def _initialize_streaming(self):
        """Initialize CSV-specific streaming configuration."""
        super()._initialize_streaming()
        self.file_path = Path(self.config['file_path'])
        self.prompt_column = self.config.get('prompt_column', 'prompt')
        self.id_column = self.config.get('id_column', 'id')

        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")

    def _discover_sources(self) -> List[Any]:
        """Return the CSV file as the single source."""
        return [self.file_path]

    def _process_source(self, source: Path) -> Iterator[LoadResult]:
        """
        Process the CSV file and yield LoadResult objects.

        Args:
            source: Path to CSV file
        """
        with open(source, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, 1):
                prompt = row.get(self.prompt_column)
                if prompt is None:
                    continue

                request_id = str(row.get(self.id_column, f"req_{idx}"))

                # Extract additional data (all columns except prompt and id)
                additional_data = {
                    k: v for k, v in row.items()
                    if k not in [self.prompt_column, self.id_column]
                }

                # Build messages
                messages = self.build_messages(prompt, additional_data)

                yield LoadResult(
                    messages=messages,
                    request_id=request_id,
                    additional_data=additional_data or None
                )


# ===== Directory Streaming Mixin =====

class DirectoryStreamingMixin(StreamingLoaderMixin):
    """
    Streaming mixin for directory-based data sources.

    Recursively discovers files matching a pattern and processes them
    in a streaming fashion.

    Configuration:
        input_dir: Root directory to search
        file_pattern: Glob pattern for files to load (default: "*.jsonl")
        recursive: Whether to search subdirectories (default: True)
        streaming: Enable streaming mode (default: True)

    Usage:
        class MyDirectoryLoader(DirectoryStreamingMixin, JSONLStreamingMixin, DataLoader):
            pass  # Inherits directory discovery and JSONL processing
    """

    def _initialize_streaming(self):
        """Initialize directory-specific streaming configuration."""
        super()._initialize_streaming()
        self.input_dir = Path(self.config['input_dir'])
        self.file_pattern = self.config.get('file_pattern', '*.jsonl')
        self.recursive = self.config.get('recursive', True)

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")

        if not self.input_dir.is_dir():
            raise ValueError(f"Input path is not a directory: {self.input_dir}")

    def _discover_sources(self) -> List[Path]:
        """
        Discover all files matching the pattern.

        Returns:
            Sorted list of file paths
        """
        if self.recursive:
            files = sorted(self.input_dir.rglob(self.file_pattern))
        else:
            files = sorted(self.input_dir.glob(self.file_pattern))

        if not files:
            raise ValueError(
                f"No files matching '{self.file_pattern}' found in {self.input_dir}"
            )

        logger.info(f"Discovered {len(files)} files matching '{self.file_pattern}'")
        return files

    def _process_source(self, source: Path) -> Iterator[LoadResult]:
        """
        Process a single file.

        This is a placeholder - subclasses should mix in a format-specific
        mixin (JSONLStreamingMixin, JSONStreamingMixin, etc.) that provides
        the actual _process_source implementation.

        Args:
            source: Path to file
        """
        # If the class also has JSONLStreamingMixin, use its logic
        if hasattr(self, '_read_jsonl_lines'):
            for line_num, line in self._read_jsonl_lines(source):
                result = self.process_line_to_load_result(
                    line=line,
                    line_num=line_num,
                    source=str(source.relative_to(self.input_dir)),
                    default_id=f"{source.name}:{line_num}"
                )
                if result is not None:
                    # Add source metadata
                    if result.additional_data is None:
                        result.additional_data = {}
                    result.additional_data['_source_file'] = str(source.relative_to(self.input_dir))
                    result.additional_data['_source_dir'] = str(source.parent.relative_to(self.input_dir))
                    yield result
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__} must be mixed with a format-specific "
                f"mixin (JSONLStreamingMixin, JSONStreamingMixin, etc.)"
            )

    def _on_source_start(self, source: Path):
        """Log when starting to process a file."""
        logger.debug(f"Processing file: {source.relative_to(self.input_dir)}")

    def _on_source_complete(self, source: Path, item_count: int):
        """Log when file processing completes."""
        logger.debug(f"Completed {source.relative_to(self.input_dir)}: {item_count} items")


# ===== Convenience classes combining mixins =====

class StreamingJSONLLoader(JSONLStreamingMixin, DataLoader):
    """
    Complete streaming JSONL loader implementation.

    This class combines all necessary mixins for a fully functional
    streaming JSONL loader. Subclass this to customize behavior.

    Usage:
        class MyCustomLoader(StreamingJSONLLoader):
            def extract_prompt(self, item):
                # Custom prompt extraction
                return item.get('custom_field')
    """
    pass


class StreamingCSVLoader(CSVStreamingMixin, DataLoader):
    """
    Complete streaming CSV loader implementation.

    This class combines all necessary mixins for a fully functional
    streaming CSV loader. Subclass this to customize behavior.

    Usage:
        class MyCustomLoader(StreamingCSVLoader):
            def build_messages(self, prompt, additional_data=None):
                # Add system prompt
                return [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
    """
    pass


class StreamingDirectoryJSONLLoader(DirectoryStreamingMixin, JSONLStreamingMixin, DataLoader):
    """
    Complete streaming directory JSONL loader implementation.

    This class combines all necessary mixins for a fully functional
    streaming loader that processes JSONL files from a directory tree.
    Subclass this to customize behavior.

    Usage:
        class MyCustomLoader(StreamingDirectoryJSONLLoader):
            def should_skip_source(self, source):
                # Skip temporary files
                return source.name.startswith('.')
    """
    pass
