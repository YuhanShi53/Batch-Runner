"""
JSONL file data loader implementation.

Loads inference requests from a JSONL file.
Each line is a separate JSON object representing one data sample.

Supports both text-only and multimodal (text + images) data.
Supports both streaming and batch modes.
Supports chunked/distributed processing via ChunkedLoaderMixin.
"""
import json
from typing import Iterator, Dict, Any, Optional, List
from pathlib import Path
import logging

from .base import DataLoader, LoadResult
from .jsonl_mixin import JSONLLoaderMixin
from .streaming_mixin import StreamingLoaderMixin, MessagesBuilderMixin
from .multimodal_base import MultimodalDataLoader, MultimodalLoadResult
from .chunked_mixin import ChunkedLoaderMixin


logger = logging.getLogger(__name__)


class JSONLDataLoader(ChunkedLoaderMixin, StreamingLoaderMixin, MessagesBuilderMixin, JSONLLoaderMixin, DataLoader):
    """
    Load inference requests from a JSONL file (text-only mode).

    Expected JSONL format (one JSON object per line):
    {"id": "1", "prompt": "What is AI?", "category": "tech"}
    {"id": "2", "prompt": "Explain quantum computing"}
    {"id": "3", "prompt": "Tell me a joke", "tags": ["humor"]}

    Configuration:
        file_path: Path to JSONL file
        prompt_field: Field name containing the prompt (default: "prompt")
        id_field: Field name containing the ID (default: "id")
        streaming: Enable streaming mode (default: True for efficiency)
        num_chunks: Total number of chunks (default: 1)
        chunk_index: Which chunk to process, 0-indexed (default: 0)

    Customization:
        Override methods from JSONLLoaderMixin to customize parsing:
        - parse_line(): Parse custom JSONL formats (e.g., list-based lines)
        - should_skip_item(): Filter items
        - extract_request_id(): Custom ID extraction
        - extract_prompt(): Custom prompt extraction
        - extract_additional_data(): Custom additional data extraction
        - build_messages(): Custom message construction (from MessagesBuilderMixin)

    Example:
        class MyLoader(JSONLDataLoader):
            def parse_line(self, line, line_num, source):
                # Handle list-format: [{"text": "hello"}]
                obj = json.loads(line)
                if isinstance(obj, list):
                    return {"items": obj, "id": str(line_num)}
                return obj
    """

    def _initialize(self):
        """Initialize JSONL file loader."""
        self.file_path = Path(self.config['file_path'])
        self.prompt_field = self.config.get('prompt_field', 'prompt')
        self.id_field = self.config.get('id_field', 'id')

        # Initialize streaming configuration from StreamingLoaderMixin
        self._initialize_streaming()

        # Initialize chunking from ChunkedLoaderMixin
        self._initialize_chunking()

        if not self.file_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {self.file_path}")

        if not self.streaming:
            # Batch mode: Load all lines into memory
            logger.info(f"Batch mode: loading all data from {self.file_path} into memory")
            self.data = []
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue

                    try:
                        # Use the mixin's parse_line method for extensibility
                        obj = self.parse_line(line, line_num, str(self.file_path))
                        if obj is not None:
                            self.data.append(obj)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON on line {line_num}: {e}")

            if not self.data:
                raise ValueError(f"No valid JSON objects found in {self.file_path}")

            logger.info(f"Loaded {len(self.data)} items into memory")

            if self.num_chunks > 1:
                estimated = self._estimate_chunk_size(len(self.data))
                logger.info(
                    f"Dataset has {len(self.data)} total items, "
                    f"this chunk will process ~{estimated} items"
                )

    def _discover_sources(self) -> List[Any]:
        """Return the single JSONL file as the data source."""
        return [self.file_path]

    def _process_source(self, source: Path) -> Iterator[LoadResult]:
        """
        Process the JSONL file line by line.

        Args:
            source: Path to the JSONL file

        Yields:
            LoadResult objects from each line
        """
        with open(source, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                # Check if this line belongs to the current chunk
                if not self._should_process_item(line_idx):
                    continue

                line_num = line_idx + 1  # 1-indexed for error messages
                for result in self.process_line_to_load_result(
                    line=line,
                    line_num=line_num,
                    source=str(source),
                    default_id=f"req_{line_num}"
                ):
                    if result is not None:
                        yield result

    def _on_source_start(self, source: Path):
        """Log when starting to process the file."""
        logger.info(f"Processing JSONL file: {source}")

    def _on_source_complete(self, source: Path, item_count: int):
        """Log when file processing is complete."""
        logger.info(f"Completed {source}: {item_count} items loaded")

    def load(self) -> Iterator[LoadResult]:
        """Yield LoadResult objects from JSONL data."""
        if self.streaming:
            # Use StreamingLoaderMixin template method
            yield from self._stream_load()
        else:
            # Batch mode: iterate over pre-loaded data
            for idx, item in enumerate(self.data):
                try:
                    # Check if this item belongs to the current chunk
                    if not self._should_process_item(idx):
                        continue

                    # Use the mixin's template method for processing
                    default_id = f"req_{idx + 1}"
                    prompt = self.extract_prompt(item)

                    if prompt is None:
                        continue

                    request_id = self.extract_request_id(item, default_id)
                    additional_data = self.extract_additional_data(item)

                    # Build messages (supports MessagesBuilderMixin)
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

    def __len__(self):
        if self.streaming:
            raise NotImplementedError("Cannot get length in streaming mode")
        return len(self.data)


class MultimodalJSONLDataLoader(ChunkedLoaderMixin, StreamingLoaderMixin, JSONLLoaderMixin, MultimodalDataLoader):
    """
    Load multimodal inference requests from a JSONL file.

    Supports text + image data in OpenAI vision API format.

    Expected JSONL format (one JSON object per line):

    Text-only:
    {"id": "1", "prompt": "What is AI?"}

    Single image:
    {"id": "2", "prompt": "Describe this image", "image": "path/to/image.jpg"}

    Multiple images:
    {"id": "3", "prompt": "Compare these images", "images": ["img1.jpg", "img2.png"]}

    Mixed (image field takes precedence over images):
    {"id": "4", "prompt": "What's in this image?", "image": "photo.jpg", "images": ["other.jpg"]}

    Configuration:
        file_path: Path to JSONL file
        prompt_field: Field name containing the text prompt (default: "prompt")
        id_field: Field name containing the ID (default: "id")
        image_field: Field name for single image (default: "image")
        images_field: Field name for multiple images (default: "images")
        image_base_dir: Base directory for relative image paths (default: "")
        encode_images: Whether to encode images to base64 (default: True)
        streaming: Enable streaming mode (default: False for backwards compatibility)
        num_chunks: Total number of chunks (default: 1)
        chunk_index: Which chunk to process, 0-indexed (default: 0)

    Image field precedence:
    - If 'image' field exists: use it (single image)
    - Else if 'images' field exists: use it (multiple images)
    - Else: text-only mode

    Customization:
        Override methods from JSONLLoaderMixin to customize parsing:
        - parse_line(): Parse custom JSONL formats
        - extract_images(): Custom image extraction
        - extract_additional_data(): Custom additional data extraction

    Example:
        class MyMultimodalLoader(MultimodalJSONLDataLoader):
            def extract_images(self, item):
                # Support nested image structure
                if 'media' in item:
                    return [m['path'] for m in item['media'] if m['type'] == 'image']
                return super().extract_images(item)
    """

    def _initialize(self):
        """Initialize multimodal JSONL file loader."""
        # Initialize multimodal base
        MultimodalDataLoader._initialize(self)

        # JSONL-specific config
        self.file_path = Path(self.config['file_path'])
        self.prompt_field = self.config.get('prompt_field', 'prompt')
        self.id_field = self.config.get('id_field', 'id')
        self.image_field = self.config.get('image_field', 'image')
        self.images_field = self.config.get('images_field', 'images')

        # Initialize streaming configuration from StreamingLoaderMixin
        # Default to False for backwards compatibility with existing behavior
        self._initialize_streaming()

        # Initialize chunking from ChunkedLoaderMixin
        self._initialize_chunking()

        if not self.file_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {self.file_path}")

        if not self.streaming:
            # Batch mode: Load all lines into memory
            logger.info(f"Batch mode: loading all data from {self.file_path} into memory")
            self.data = []
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        # Use the mixin's parse_line method for extensibility
                        obj = self.parse_line(line, line_num, str(self.file_path))
                        if obj is not None:
                            self.data.append(obj)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON on line {line_num}: {e}")

            if not self.data:
                raise ValueError(f"No valid JSON objects found in {self.file_path}")

            logger.info(f"Loaded {len(self.data)} items into memory")

            if self.num_chunks > 1:
                estimated = self._estimate_chunk_size(len(self.data))
                logger.info(
                    f"Dataset has {len(self.data)} total items, "
                    f"this chunk will process ~{estimated} items"
                )

    def _discover_sources(self) -> List[Any]:
        """Return the single JSONL file as the data source."""
        return [self.file_path]

    def _process_source(self, source: Path) -> Iterator[MultimodalLoadResult]:
        """
        Process the JSONL file line by line for multimodal data.

        Args:
            source: Path to the JSONL file

        Yields:
            MultimodalLoadResult objects from each line
        """
        with open(source, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                # Check if this line belongs to the current chunk
                if not self._should_process_item(line_idx):
                    continue

                line_num = line_idx + 1  # 1-indexed for error messages

                # Parse the line
                item_or_items = self.parse_line(line, line_num, str(source))
                if item_or_items is None:
                    continue

                # Normalize to list for uniform processing
                items = [item_or_items] if isinstance(item_or_items, dict) else item_or_items
                if not items:
                    continue

                # Process each item
                for item_idx, item in enumerate(items):
                    # Check if we should skip this item
                    if self.should_skip_item(item):
                        logger.debug(f"Skipping item in {source}:{line_num}[{item_idx}]")
                        continue

                    # Extract prompt
                    prompt = self.extract_prompt(item)
                    if prompt is None:
                        logger.debug(f"Skipping item in {source}:{line_num}[{item_idx}]: no prompt found")
                        continue

                    # For multi-item case, add suffix to default_id
                    if len(items) > 1:
                        item_default_id = f"req_{line_num}_{item_idx}"
                    else:
                        item_default_id = f"req_{line_num}"

                    # Extract request_id
                    request_id = self.extract_request_id(item, item_default_id)

                    # Extract images using custom method
                    images = self.extract_images(item)

                    # Extract additional data using custom method
                    additional_data = self.extract_additional_data(item)

                    # Create multimodal result
                    yield self._create_multimodal_result(
                        text=prompt,
                        images=images,
                        request_id=str(request_id),
                        additional_data=additional_data or None
                    )

    def _on_source_start(self, source: Path):
        """Log when starting to process the file."""
        logger.info(f"Processing multimodal JSONL file: {source}")

    def _on_source_complete(self, source: Path, item_count: int):
        """Log when file processing is complete."""
        logger.info(f"Completed {source}: {item_count} items loaded")

    def extract_images(self, item: Dict[str, Any]) -> Optional[List[str]]:
        """
        Extract image paths from a JSONL item.

        This method can be overridden to customize image extraction.
        Default implementation checks image_field and images_field.

        Args:
            item: Dictionary representing one JSONL line

        Returns:
            List of image paths/strings, or None if no images present
        """
        # Check for single image field first (higher precedence)
        if self.image_field in item:
            image_value = item[self.image_field]
            if image_value:
                # Convert single value to list
                if isinstance(image_value, str):
                    return [image_value]
                elif isinstance(image_value, list):
                    return image_value
                else:
                    logger.warning(f"Invalid {self.image_field} type in item {item.get(self.id_field)}: {type(image_value)}")

        # Check for multiple images field
        elif self.images_field in item:
            images_value = item[self.images_field]
            if images_value:
                if isinstance(images_value, list):
                    return images_value
                elif isinstance(images_value, str):
                    return [images_value]
                else:
                    logger.warning(f"Invalid {self.images_field} type in item {item.get(self.id_field)}: {type(images_value)}")

        return None

    def extract_additional_data(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract additional data from parsed item (multimodal version).

        Excludes prompt, id, and image fields.

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

    def load(self) -> Iterator[MultimodalLoadResult]:
        """Yield MultimodalLoadResult objects from JSONL data."""
        if self.streaming:
            # Use StreamingLoaderMixin template method
            yield from self._stream_load()
        else:
            # Batch mode: iterate over pre-loaded data
            for idx, item in enumerate(self.data):
                try:
                    # Check if this item belongs to the current chunk
                    if not self._should_process_item(idx):
                        continue

                    # Use the mixin's extract_prompt method
                    prompt = self.extract_prompt(item)
                    if prompt is None:
                        logger.debug(
                            f"Skipping item {idx}: no '{self.prompt_field}' field"
                        )
                        continue

                    # Use the mixin's extract_request_id method
                    request_id = self.extract_request_id(item, f"req_{idx + 1}")

                    # Extract images using custom method
                    images = self.extract_images(item)

                    # Extract additional data using custom method
                    additional_data = self.extract_additional_data(item)

                    # Create multimodal result
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

    def __len__(self):
        if self.streaming:
            raise NotImplementedError("Cannot get length in streaming mode")
        return len(self.data)
