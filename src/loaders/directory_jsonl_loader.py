"""
Directory JSONL data loader implementation.

Recursively loads conv.jsonl files from a directory tree.
Preserves directory structure information for each loaded item.

Supports both text-only and multimodal (text + images) data.
Supports chunked/distributed processing via ChunkedLoaderMixin.
"""
import json
from typing import Iterator, Dict, Any, Optional, List, Tuple
from pathlib import Path
import logging

from .base import DataLoader, LoadResult
from .jsonl_mixin import JSONLLoaderMixin
from .streaming_mixin import StreamingLoaderMixin, MessagesBuilderMixin
from .multimodal_base import MultimodalDataLoader, MultimodalLoadResult
from .chunked_mixin import ChunkedLoaderMixin


logger = logging.getLogger(__name__)


class DirectoryJSONLDataLoader(ChunkedLoaderMixin, StreamingLoaderMixin, MessagesBuilderMixin, JSONLLoaderMixin, DataLoader):
    """
    Load inference requests from conv.jsonl files in a directory tree (text-only mode).

    Recursively searches for files named 'conv.jsonl' in the input directory
    and loads all JSON objects from them. Each loaded item includes directory
    structure information in additional_data for later reconstruction.

    Expected conv.jsonl format (one JSON object per line):
    {"id": "1", "prompt": "What is AI?", "category": "tech"}
    {"id": "2", "prompt": "Explain quantum computing"}

    Configuration:
        input_dir: Root directory to search for conv.jsonl files
        file_pattern: Glob pattern for files to load (default: "conv.jsonl")
        prompt_field: Field name containing the prompt (default: "prompt")
        id_field: Field name containing the ID (default: "id")
        recursive: Whether to search subdirectories (default: true)
        streaming: Enable streaming mode (default: True for efficiency)
        num_chunks: Total number of chunks (default: 1)
        chunk_index: Which chunk to process, 0-indexed (default: 0)

    Additional data included for each item:
        _source_file: Relative path from input_dir to the source conv.jsonl file
        _source_dir: Relative directory path from input_dir
    """

    def _initialize(self):
        """Initialize directory JSONL loader."""
        self.input_dir = Path(self.config['input_dir'])
        self.file_pattern = self.config.get('file_pattern', 'conv.jsonl')
        self.prompt_field = self.config.get('prompt_field', 'prompt')
        self.id_field = self.config.get('id_field', 'id')
        self.recursive = self.config.get('recursive', True)

        # Initialize streaming configuration from StreamingLoaderMixin
        self._initialize_streaming()

        # Initialize chunking from ChunkedLoaderMixin
        self._initialize_chunking()

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")

        if not self.input_dir.is_dir():
            raise ValueError(f"Input path is not a directory: {self.input_dir}")

        # Find all conv.jsonl files
        if self.recursive:
            self.files = sorted(self.input_dir.rglob(self.file_pattern))
        else:
            self.files = sorted(self.input_dir.glob(self.file_pattern))

        if not self.files:
            raise ValueError(
                f"No files matching '{self.file_pattern}' found in {self.input_dir}"
            )

        logger.info(f"Found {len(self.files)} {self.file_pattern} files in {self.input_dir}")

        # For backwards compatibility, support non-streaming mode
        if not self.streaming:
            logger.info("Non-streaming mode: loading all data into memory")
            self.data = []
            for file_path in self.files:
                rel_path = file_path.relative_to(self.input_dir)
                rel_dir = rel_path.parent

                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            # Use the mixin's parse_line method for extensibility
                            obj = self.parse_line(line, line_num, str(rel_path))
                            if obj is not None:
                                # Add source information
                                obj['_source_file'] = str(rel_path)
                                obj['_source_dir'] = str(rel_dir) if rel_dir != Path('.') else ''
                                self.data.append(obj)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON in {rel_path}:{line_num}: {e}")
                            continue

            if not self.data:
                raise ValueError(f"No valid JSON objects found in {self.input_dir}")

            logger.info(f"Loaded {len(self.data)} items from {len(self.files)} files into memory")

            if self.num_chunks > 1:
                estimated = self._estimate_chunk_size(len(self.data))
                logger.info(
                    f"Dataset has {len(self.data)} total items, "
                    f"this chunk will process ~{estimated} items"
                )

    def _discover_sources(self) -> List[Tuple[Path, Path, Path]]:
        """
        Discover all JSONL files along with their metadata.

        Returns:
            List of tuples (file_path, rel_path, rel_dir) for each file
        """
        sources = []
        for file_path in self.files:
            rel_path = file_path.relative_to(self.input_dir)
            rel_dir = rel_path.parent
            sources.append((file_path, rel_path, rel_dir))
        return sources

    def _process_source(self, source: Tuple[Path, Path, Path]) -> Iterator[LoadResult]:
        """
        Process a single JSONL file.

        Args:
            source: Tuple of (file_path, rel_path, rel_dir)

        Yields:
            LoadResult objects from the file
        """
        file_path, rel_path, rel_dir = source

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    # Use the mixin's process_line_to_load_result template method
                    result = self.process_line_to_load_result(
                        line=line,
                        line_num=line_num,
                        source=str(rel_path),
                        default_id=f"{rel_path}:{line_num}"
                    )

                    if result is None:
                        # Line was skipped by the mixin
                        continue

                    # Add source information to additional_data
                    if result.additional_data is None:
                        result.additional_data = {}
                    result.additional_data['_source_file'] = str(rel_path)
                    result.additional_data['_source_dir'] = str(rel_dir) if rel_dir != Path('.') else ''

                    yield result

                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in {rel_path}:{line_num}: {e}")
                    continue

    def _on_source_start(self, source: Tuple[Path, Path, Path]):
        """Log when starting to process a file."""
        file_path, rel_path, _ = source
        logger.debug(f"Processing file: {rel_path}")

    def _on_source_complete(self, source: Tuple[Path, Path, Path], item_count: int):
        """Log when file processing is complete."""
        file_path, rel_path, _ = source
        logger.debug(f"Completed {rel_path}: {item_count} items")

    def _on_source_error(self, source: Tuple[Path, Path, Path], error: Exception):
        """Handle file processing errors."""
        file_path, rel_path, _ = source
        logger.error(f"Error reading file {rel_path}: {error}")

    def load(self) -> Iterator[LoadResult]:
        """Yield LoadResult objects from directory JSONL data."""
        if self.streaming:
            # Use chunked streaming template method
            yield from self._chunked_stream_load()
        else:
            # Batch mode: iterate over pre-loaded data
            for idx, item in enumerate(self.data):
                try:
                    # Check if this item belongs to the current chunk
                    if not self._should_process_item(idx):
                        continue

                    prompt = self.extract_prompt(item)
                    if prompt is None:
                        logger.debug(
                            f"Skipping item: no '{self.prompt_field}' field"
                        )
                        continue

                    request_id = self.extract_request_id(item, f"req_{id(item)}")
                    additional_data = self.extract_additional_data(item)

                    # Build messages using MessagesBuilderMixin
                    messages = self.build_messages(prompt, additional_data)

                    yield LoadResult(
                        messages=messages,
                        request_id=str(request_id),
                        additional_data=additional_data or None
                    )
                except Exception as e:
                    logger.error(
                        f"Unexpected error processing item at index {idx}: {e}"
                    )
                    continue

    def _chunked_stream_load(self) -> Iterator[LoadResult]:
        """
        Streaming template method with chunking support.

        This overrides the default StreamingLoaderMixin._stream_load to
        add global item indexing for chunked processing across multiple files.
        """
        sources = self._discover_sources()
        logger.info(f"Streaming mode: discovered {len(sources)} sources")

        if self.num_chunks > 1:
            logger.info(
                f"Chunked streaming: processing chunk {self.chunk_index + 1}/"
                f"{self.num_chunks} (approximately {100.0 / self.num_chunks:.1f}% of data)"
            )

        global_idx = 0
        for source in sources:
            # Check if we should skip this source
            if self._should_skip_source(source):
                logger.debug(f"Skipping source: {source}")
                continue

            # Call pre-processing hook
            self._on_source_start(source)

            item_count = 0
            try:
                # Process this source
                for result in self._process_source_with_index(source, global_idx):
                    item_count += 1
                    global_idx += 1
                    yield result

                # Call post-processing hook
                self._on_source_complete(source, item_count)

            except Exception as e:
                # Call error handling hook
                self._on_source_error(source, e)

    def _process_source_with_index(
        self, source: Tuple[Path, Path, Path], start_global_idx: int
    ) -> Iterator[LoadResult]:
        """
        Process a single JSONL file with global indexing for chunking.

        Args:
            source: Tuple of (file_path, rel_path, rel_dir)
            start_global_idx: Starting global index for this file

        Yields:
            LoadResult objects from the file
        """
        file_path, rel_path, rel_dir = source

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                # Calculate global index for this line
                global_idx = start_global_idx + line_idx

                # Check if this line belongs to the current chunk
                if not self._should_process_item(global_idx):
                    continue

                line_num = line_idx + 1  # 1-indexed for error messages
                try:
                    # Use the mixin's process_line_to_load_result template method
                    result = self.process_line_to_load_result(
                        line=line,
                        line_num=line_num,
                        source=str(rel_path),
                        default_id=f"{rel_path}:{line_num}"
                    )

                    if result is None:
                        # Line was skipped by the mixin
                        continue

                    # Add source information to additional_data
                    if result.additional_data is None:
                        result.additional_data = {}
                    result.additional_data['_source_file'] = str(rel_path)
                    result.additional_data['_source_dir'] = str(rel_dir) if rel_dir != Path('.') else ''

                    yield result

                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in {rel_path}:{line_num}: {e}")
                    continue

    def __len__(self):
        if not self.streaming:
            return len(self.data)
        # In streaming mode, we don't know the total count upfront
        raise NotImplementedError("Cannot get length in streaming mode")


class MultimodalDirectoryJSONLDataLoader(ChunkedLoaderMixin, StreamingLoaderMixin, JSONLLoaderMixin, MultimodalDataLoader):
    """
    Load multimodal inference requests from conv.jsonl files in a directory tree.

    Supports text + image data in OpenAI vision API format.
    Images are resolved relative to the directory containing each conv.jsonl file.

    Expected conv.jsonl format (one JSON object per line):

    Text-only:
    {"id": "1", "prompt": "What is AI?"}

    Single image (relative path):
    {"id": "2", "prompt": "Describe this image", "image": "images/photo.jpg"}

    Multiple images:
    {"id": "3", "prompt": "Compare these images", "images": ["img1.jpg", "img2.png"]}

    Mixed (image field takes precedence over images):
    {"id": "4", "prompt": "What's in this image?", "image": "photo.jpg", "images": ["other.jpg"]}

    Configuration:
        input_dir: Root directory to search for conv.jsonl files
        file_pattern: Glob pattern for files to load (default: "conv.jsonl")
        prompt_field: Field name containing the text prompt (default: "prompt")
        id_field: Field name containing the ID (default: "id")
        image_field: Field name for single image (default: "image")
        images_field: Field name for multiple images (default: "images")
        image_base_dir: Base directory for relative image paths (default: "")
                      If not specified, images are resolved relative to each conv.jsonl file's directory
        encode_images: Whether to encode images to base64 (default: True)
        recursive: Whether to search subdirectories (default: true)
        num_chunks: Total number of chunks (default: 1)
        chunk_index: Which chunk to process, 0-indexed (default: 0)

    Image resolution:
    - If image_base_dir is specified: relative paths are resolved from image_base_dir
    - If not specified: relative paths are resolved from the directory containing each conv.jsonl file
    - Absolute paths are used as-is

    Image field precedence:
    - If 'image' field exists: use it (single image)
    - Else if 'images' field exists: use it (multiple images)
    - Else: text-only mode

    Additional data included for each item:
        _source_file: Relative path from input_dir to the source conv.jsonl file
        _source_dir: Relative directory path from input_dir
    """

    def _initialize(self):
        """Initialize multimodal directory JSONL loader."""
        # Initialize multimodal base
        MultimodalDataLoader._initialize(self)

        # Directory-specific config
        self.input_dir = Path(self.config['input_dir'])
        self.file_pattern = self.config.get('file_pattern', 'conv.jsonl')
        self.prompt_field = self.config.get('prompt_field', 'prompt')
        self.id_field = self.config.get('id_field', 'id')
        self.image_field = self.config.get('image_field', 'image')
        self.images_field = self.config.get('images_field', 'images')
        self.recursive = self.config.get('recursive', True)

        # Initialize streaming configuration from StreamingLoaderMixin
        self._initialize_streaming()

        # Initialize chunking from ChunkedLoaderMixin
        self._initialize_chunking()

        # If image_base_dir is not specified, we'll use the directory of each conv.jsonl file
        self.use_source_dir_as_base = not self.config.get('image_base_dir', '')

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")

        if not self.input_dir.is_dir():
            raise ValueError(f"Input path is not a directory: {self.input_dir}")

        # Find all conv.jsonl files (sorted for deterministic order)
        if self.recursive:
            self.files = sorted(self.input_dir.rglob(self.file_pattern))
        else:
            self.files = sorted(self.input_dir.glob(self.file_pattern))

        if not self.files:
            raise ValueError(
                f"No files matching '{self.file_pattern}' found in {self.input_dir}"
            )

        logger.info(f"Found {len(self.files)} {self.file_pattern} files in {self.input_dir}")

        # For backwards compatibility, support non-streaming mode
        if not self.streaming:
            logger.info("Non-streaming mode: loading all data into memory")
            self.data = []
            for file_path in self.files:
                rel_path = file_path.relative_to(self.input_dir)
                rel_dir = rel_path.parent
                source_dir = file_path.parent

                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            # Use the mixin's parse_line method for extensibility
                            obj = self.parse_line(line, line_num, str(rel_path))
                            if obj is not None:
                                # Add source information
                                obj['_source_file'] = str(rel_path)
                                obj['_source_dir'] = str(rel_dir) if rel_dir != Path('.') else ''
                                obj['_source_dir_path'] = str(source_dir)
                                self.data.append(obj)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON in {rel_path}:{line_num}: {e}")
                            continue

            if not self.data:
                raise ValueError(f"No valid JSON objects found in {self.input_dir}")

            logger.info(f"Loaded {len(self.data)} items from {len(self.files)} files into memory")

            if self.num_chunks > 1:
                estimated = self._estimate_chunk_size(len(self.data))
                logger.info(
                    f"Dataset has {len(self.data)} total items, "
                    f"this chunk will process ~{estimated} items"
                )

    def _discover_sources(self) -> List[Tuple[Path, Path, Path, Path]]:
        """
        Discover all JSONL files along with their metadata.

        Returns:
            List of tuples (file_path, rel_path, rel_dir, source_dir) for each file
        """
        sources = []
        for file_path in self.files:
            rel_path = file_path.relative_to(self.input_dir)
            rel_dir = rel_path.parent
            source_dir = file_path.parent
            sources.append((file_path, rel_path, rel_dir, source_dir))
        return sources

    def _process_source(self, source: Tuple[Path, Path, Path, Path]) -> Iterator[MultimodalLoadResult]:
        """
        Process a single JSONL file for multimodal data.

        Args:
            source: Tuple of (file_path, rel_path, rel_dir, source_dir)

        Yields:
            MultimodalLoadResult objects from the file
        """
        file_path, rel_path, rel_dir, source_dir = source

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    # Use the mixin's parse_line method for extensibility
                    obj = self.parse_line(line, line_num, str(rel_path))
                    if obj is None:
                        # Line was skipped by the mixin
                        continue

                    # Add source information
                    obj['_source_file'] = str(rel_path)
                    obj['_source_dir'] = str(rel_dir) if rel_dir != Path('.') else ''
                    obj['_source_dir_path'] = str(source_dir)

                    prompt = self.extract_prompt(obj)
                    if prompt is None:
                        logger.debug(f"Skipping item {rel_path}:{line_num}: no '{self.prompt_field}' field")
                        continue

                    request_id = self.extract_request_id(obj, f"{rel_path}:{line_num}")
                    images = self.extract_images(obj)

                    # Extract additional data with proper field exclusion
                    additional_data = self._extract_additional_data_multimodal(obj)

                    yield self._create_multimodal_result(
                        text=prompt,
                        images=images,
                        request_id=str(request_id),
                        additional_data=additional_data or None
                    )
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in {rel_path}:{line_num}: {e}")
                    continue

    def _on_source_start(self, source: Tuple[Path, Path, Path, Path]):
        """Log when starting to process a file."""
        file_path, rel_path, _, _ = source
        logger.debug(f"Processing file: {rel_path}")

    def _on_source_complete(self, source: Tuple[Path, Path, Path, Path], item_count: int):
        """Log when file processing is complete."""
        file_path, rel_path, _, _ = source
        logger.debug(f"Completed {rel_path}: {item_count} items")

    def _on_source_error(self, source: Tuple[Path, Path, Path, Path], error: Exception):
        """Handle file processing errors."""
        file_path, rel_path, _, _ = source
        logger.error(f"Error reading file {rel_path}: {error}")

    def _resolve_image_path(self, image_path: str, source_dir_path: str) -> str:
        """
        Resolve image path relative to the source conv.jsonl file's directory.

        Args:
            image_path: Image path from JSON data
            source_dir_path: Directory containing the conv.jsonl file

        Returns:
            Resolved image path (absolute or relative to image_base_dir if set)
        """
        path = Path(image_path)

        # If already absolute, return as-is
        if path.is_absolute():
            return image_path

        # If use_source_dir_as_base, resolve relative to source file's directory
        if self.use_source_dir_as_base:
            source_dir = Path(source_dir_path)
            resolved = source_dir / path
            return str(resolved)

        # Otherwise use image_base_dir (handled by parent class's _encode_image_to_base64)
        return image_path

    def extract_images(self, item: Dict[str, Any]) -> Optional[List[str]]:
        """
        Extract and resolve image paths from a JSONL item.

        This method can be overridden to customize image extraction.
        Default implementation checks image_field and images_field.

        Args:
            item: Dictionary representing one JSONL line

        Returns:
            List of resolved image paths, or None if no images present
        """
        source_dir_path = item.get('_source_dir_path', '')

        # Check for single image field first (higher precedence)
        if self.image_field in item:
            image_value = item[self.image_field]
            if image_value:
                # Convert single value to list and resolve paths
                if isinstance(image_value, str):
                    resolved = self._resolve_image_path(image_value, source_dir_path)
                    return [resolved]
                elif isinstance(image_value, list):
                    return [
                        self._resolve_image_path(img, source_dir_path)
                        for img in image_value
                    ]
                else:
                    logger.warning(
                        f"Invalid {self.image_field} type in item {item.get(self.id_field)}: "
                        f"{type(image_value)}"
                    )

        # Check for multiple images field
        elif self.images_field in item:
            images_value = item[self.images_field]
            if images_value:
                if isinstance(images_value, list):
                    return [
                        self._resolve_image_path(img, source_dir_path)
                        for img in images_value
                    ]
                elif isinstance(images_value, str):
                    resolved = self._resolve_image_path(images_value, source_dir_path)
                    return [resolved]
                else:
                    logger.warning(
                        f"Invalid {self.images_field} type in item {item.get(self.id_field)}: "
                        f"{type(images_value)}"
                    )

        return None

    def load(self) -> Iterator[MultimodalLoadResult]:
        """Yield MultimodalLoadResult objects from directory JSONL data."""
        if self.streaming:
            # Use chunked streaming template method
            yield from self._chunked_stream_load_multimodal()
        else:
            # Batch mode: iterate over pre-loaded data
            for idx, item in enumerate(self.data):
                try:
                    # Check if this item belongs to the current chunk
                    if not self._should_process_item(idx):
                        continue

                    prompt = self.extract_prompt(item)
                    if prompt is None:
                        logger.debug(
                            f"Skipping item: no '{self.prompt_field}' field"
                        )
                        continue

                    request_id = self.extract_request_id(item, f"req_{id(item)}")
                    images = self.extract_images(item)

                    # Extract additional data with proper field exclusion
                    additional_data = self._extract_additional_data_multimodal(item)

                    yield self._create_multimodal_result(
                        text=prompt,
                        images=images,
                        request_id=str(request_id),
                        additional_data=additional_data or None
                    )
                except Exception as e:
                    logger.error(
                        f"Unexpected error processing item at index {idx}: {e}"
                    )
                    continue

    def _chunked_stream_load_multimodal(self) -> Iterator[MultimodalLoadResult]:
        """
        Streaming template method with chunking support for multimodal data.

        This overrides the default StreamingLoaderMixin._stream_load to
        add global item indexing for chunked processing across multiple files.
        """
        sources = self._discover_sources()
        logger.info(f"Streaming mode: discovered {len(sources)} sources")

        if self.num_chunks > 1:
            logger.info(
                f"Chunked streaming: processing chunk {self.chunk_index + 1}/"
                f"{self.num_chunks} (approximately {100.0 / self.num_chunks:.1f}% of data)"
            )

        global_idx = 0
        for source in sources:
            # Check if we should skip this source
            if self._should_skip_source(source):
                logger.debug(f"Skipping source: {source}")
                continue

            # Call pre-processing hook
            self._on_source_start(source)

            item_count = 0
            try:
                # Process this source
                for result in self._process_source_multimodal_with_index(source, global_idx):
                    item_count += 1
                    global_idx += 1
                    yield result

                # Call post-processing hook
                self._on_source_complete(source, item_count)

            except Exception as e:
                # Call error handling hook
                self._on_source_error(source, e)

    def _process_source_multimodal_with_index(
        self, source: Tuple[Path, Path, Path, Path], start_global_idx: int
    ) -> Iterator[MultimodalLoadResult]:
        """
        Process a single JSONL file with global indexing for chunking (multimodal).

        Args:
            source: Tuple of (file_path, rel_path, rel_dir, source_dir)
            start_global_idx: Starting global index for this file

        Yields:
            MultimodalLoadResult objects from the file
        """
        file_path, rel_path, rel_dir, source_dir = source

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                # Calculate global index for this line
                global_idx = start_global_idx + line_idx

                # Check if this line belongs to the current chunk
                if not self._should_process_item(global_idx):
                    continue

                line_num = line_idx + 1  # 1-indexed for error messages
                try:
                    # Use the mixin's parse_line method for extensibility
                    obj = self.parse_line(line, line_num, str(rel_path))
                    if obj is None:
                        # Line was skipped by the mixin
                        continue

                    # Add source information
                    obj['_source_file'] = str(rel_path)
                    obj['_source_dir'] = str(rel_dir) if rel_dir != Path('.') else ''
                    obj['_source_dir_path'] = str(source_dir)

                    prompt = self.extract_prompt(obj)
                    if prompt is None:
                        logger.debug(f"Skipping item {rel_path}:{line_num}: no '{self.prompt_field}' field")
                        continue

                    request_id = self.extract_request_id(obj, f"{rel_path}:{line_num}")
                    images = self.extract_images(obj)

                    # Extract additional data with proper field exclusion
                    additional_data = self._extract_additional_data_multimodal(obj)

                    yield self._create_multimodal_result(
                        text=prompt,
                        images=images,
                        request_id=str(request_id),
                        additional_data=additional_data or None
                    )
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in {rel_path}:{line_num}: {e}")
                    continue

    def _extract_additional_data_multimodal(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract additional data from parsed item (multimodal version).

        Excludes prompt, id, image fields, and source metadata.

        Args:
            item: Parsed dictionary from parse_line()

        Returns:
            Dictionary of additional data
        """
        excluded_fields = {
            self.prompt_field, self.id_field,
            self.image_field, self.images_field
        }
        return {
            k: v for k, v in item.items()
            if k not in excluded_fields and v is not None
        }

    def __len__(self):
        if not self.streaming:
            return len(self.data)
        # In streaming mode, we don't know the total count upfront
        raise NotImplementedError("Cannot get length in streaming mode")
