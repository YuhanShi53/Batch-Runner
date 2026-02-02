"""
Directory JSONL data loader implementation.

Recursively loads conv.jsonl files from a directory tree.
Preserves directory structure information for each loaded item.

Supports both text-only and multimodal (text + images) data.
"""
import json
from typing import Iterator, Dict, Any, Optional, List
from pathlib import Path
import logging

from .base import DataLoader, LoadResult
from .multimodal_base import MultimodalDataLoader, MultimodalLoadResult


logger = logging.getLogger(__name__)


class DirectoryJSONLDataLoader(DataLoader):
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
        multimodal: Set to True to enable multimodal support (default: False)

    For multimodal mode, set multimodal: True in config.
    See MultimodalDirectoryJSONLDataLoader for details.

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
        self.multimodal = self.config.get('multimodal', False)

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")

        if not self.input_dir.is_dir():
            raise ValueError(f"Input path is not a directory: {self.input_dir}")

        # Find all conv.jsonl files
        if self.recursive:
            self.files = list(self.input_dir.rglob(self.file_pattern))
        else:
            self.files = list(self.input_dir.glob(self.file_pattern))

        if not self.files:
            raise ValueError(
                f"No files matching '{self.file_pattern}' found in {self.input_dir}"
            )

        logger.info(f"Found {len(self.files)} {self.file_pattern} files in {self.input_dir}")

        # Load all data into memory with source information
        self.data = []
        for file_path in self.files:
            # Calculate relative path for output reconstruction
            rel_path = file_path.relative_to(self.input_dir)
            rel_dir = rel_path.parent

            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue

                    try:
                        obj = json.loads(line)
                        # Add source information
                        obj['_source_file'] = str(rel_path)
                        obj['_source_dir'] = str(rel_dir) if rel_dir != Path('.') else ''
                        self.data.append(obj)
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Invalid JSON in {rel_path}:{line_num}: {e}"
                        )
                        continue

        if not self.data:
            raise ValueError(f"No valid JSON objects found in {self.input_dir}")

        logger.info(f"Loaded {len(self.data)} items from {len(self.files)} files")

    def load(self) -> Iterator[LoadResult]:
        """Yield LoadResult objects from directory JSONL data."""
        for item in self.data:
            prompt = item.get(self.prompt_field)
            request_id = item.get(self.id_field, f"req_{id(item)}")

            if prompt is None:
                logger.debug(f"Skipping item {request_id}: no '{self.prompt_field}' field")
                continue

            # Extract additional data (everything except prompt and id)
            # Keep _source_file and _source_dir for output reconstruction
            additional_data = {
                k: v for k, v in item.items()
                if k not in [self.prompt_field, self.id_field]
            }

            yield LoadResult(
                messages=[{"role": "user", "content": prompt}],
                request_id=str(request_id),
                additional_data=additional_data or None
            )

    def __len__(self):
        return len(self.data)


class MultimodalDirectoryJSONLDataLoader(MultimodalDataLoader):
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
        super()._initialize()

        # Directory-specific config
        self.input_dir = Path(self.config['input_dir'])
        self.file_pattern = self.config.get('file_pattern', 'conv.jsonl')
        self.prompt_field = self.config.get('prompt_field', 'prompt')
        self.id_field = self.config.get('id_field', 'id')
        self.image_field = self.config.get('image_field', 'image')
        self.images_field = self.config.get('images_field', 'images')
        self.recursive = self.config.get('recursive', True)

        # If image_base_dir is not specified, we'll use the directory of each conv.jsonl file
        self.use_source_dir_as_base = not self.config.get('image_base_dir', '')

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")

        if not self.input_dir.is_dir():
            raise ValueError(f"Input path is not a directory: {self.input_dir}")

        # Find all conv.jsonl files
        if self.recursive:
            self.files = list(self.input_dir.rglob(self.file_pattern))
        else:
            self.files = list(self.input_dir.glob(self.file_pattern))

        if not self.files:
            raise ValueError(
                f"No files matching '{self.file_pattern}' found in {self.input_dir}"
            )

        logger.info(f"Found {len(self.files)} {self.file_pattern} files in {self.input_dir}")

        # Load all data into memory with source information
        self.data = []
        for file_path in self.files:
            # Calculate relative path for output reconstruction
            rel_path = file_path.relative_to(self.input_dir)
            rel_dir = rel_path.parent
            source_dir = file_path.parent

            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        obj = json.loads(line)
                        # Add source information
                        obj['_source_file'] = str(rel_path)
                        obj['_source_dir'] = str(rel_dir) if rel_dir != Path('.') else ''
                        obj['_source_dir_path'] = str(source_dir)  # Store for image resolution
                        self.data.append(obj)
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Invalid JSON in {rel_path}:{line_num}: {e}"
                        )
                        continue

        if not self.data:
            raise ValueError(f"No valid JSON objects found in {self.input_dir}")

        logger.info(f"Loaded {len(self.data)} items from {len(self.files)} files")

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

    def _extract_images(self, item: Dict[str, Any]) -> Optional[List[str]]:
        """
        Extract image paths from a JSONL item and resolve them.

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
        for item in self.data:
            prompt = item.get(self.prompt_field)
            request_id = item.get(self.id_field, f"req_{id(item)}")

            if prompt is None:
                logger.debug(f"Skipping item {request_id}: no '{self.prompt_field}' field")
                continue

            # Extract and resolve images
            images = self._extract_images(item)

            # Extract additional data (exclude prompt, id, image fields, and internal fields)
            excluded_fields = {
                self.prompt_field, self.id_field,
                self.image_field, self.images_field,
                '_source_file', '_source_dir', '_source_dir_path'
            }
            additional_data = {
                k: v for k, v in item.items()
                if k not in excluded_fields
            }

            # Create multimodal result
            yield self._create_multimodal_result(
                text=prompt,
                images=images,
                request_id=str(request_id),
                additional_data=additional_data or None
            )

    def __len__(self):
        return len(self.data)
