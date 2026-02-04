"""
JSON file data loader implementation.

Loads inference requests from a JSON file.

Supports both text-only and multimodal (text + images) data.
"""
import json
from typing import Iterator, Dict, Any, Optional, List
from pathlib import Path
import logging

from .base import DataLoader, LoadResult
from .multimodal_base import MultimodalDataLoader, MultimodalLoadResult


logger = logging.getLogger(__name__)


class JSONDataLoader(DataLoader):
    """
    Load inference requests from a JSON file (text-only mode).

    Expected JSON format:
    [
        {"id": "1", "prompt": "What is AI?", "category": "tech"},
        {"id": "2", "prompt": "Explain quantum computing"}
    ]

    Configuration:
        file_path: Path to JSON file
        batch_size: Number of items to load at once (default: 1)
        prompt_field: Field name containing the prompt (default: "prompt")
        id_field: Field name containing the ID (default: "id")
    """

    def _initialize(self):
        """Initialize JSON file loader."""
        self.file_path = Path(self.config['file_path'])
        self.prompt_field = self.config.get('prompt_field', 'prompt')
        self.id_field = self.config.get('id_field', 'id')

        if not self.file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {self.file_path}")

        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        if not isinstance(self.data, list):
            raise ValueError("JSON root must be a list of objects")

    def load(self) -> Iterator[LoadResult]:
        """Yield LoadResult objects from JSON data."""
        for idx, item in enumerate(self.data, 1):
            try:
                prompt = item.get(self.prompt_field)
                request_id = item.get(
                    self.id_field, f"req_{id(item)}"
                )

                if prompt is None:
                    continue

                # Extract additional data (everything except prompt and id)
                additional_data = {
                    k: v for k, v in item.items()
                    if k not in [self.prompt_field, self.id_field]
                }

                yield LoadResult(
                    messages=[{"role": "user", "content": prompt}],
                    request_id=str(request_id),
                    additional_data=additional_data or None
                )
            except Exception as e:
                logger.error(
                    f"Unexpected error processing item at index {idx}: {e}"
                )
                continue

    def __len__(self):
        return len(self.data)


class MultimodalJSONDataLoader(MultimodalDataLoader):
    """
    Load multimodal inference requests from a JSON file.

    Supports text + image data in OpenAI vision API format.

    Expected JSON format:
    [
        {"id": "1", "prompt": "What is AI?"},
        {"id": "2", "prompt": "Describe this image", "image": "path/to/image.jpg"},
        {"id": "3", "prompt": "Compare these", "images": ["img1.jpg", "img2.png"]}
    ]

    Configuration:
        file_path: Path to JSON file
        prompt_field: Field name containing the text prompt (default: "prompt")
        id_field: Field name containing the ID (default: "id")
        image_field: Field name for single image (default: "image")
        images_field: Field name for multiple images (default: "images")
        image_base_dir: Base directory for relative image paths (default: "")
        encode_images: Whether to encode images to base64 (default: True)

    Image field precedence:
    - If 'image' field exists: use it (single image)
    - Else if 'images' field exists: use it (multiple images)
    - Else: text-only mode
    """

    def _initialize(self):
        """Initialize multimodal JSON file loader."""
        super()._initialize()

        self.file_path = Path(self.config['file_path'])
        self.prompt_field = self.config.get('prompt_field', 'prompt')
        self.id_field = self.config.get('id_field', 'id')
        self.image_field = self.config.get('image_field', 'image')
        self.images_field = self.config.get('images_field', 'images')

        if not self.file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {self.file_path}")

        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        if not isinstance(self.data, list):
            raise ValueError("JSON root must be a list of objects")

    def _extract_images(self, item: Dict[str, Any]) -> Optional[List[str]]:
        """
        Extract image paths from a JSON item.

        Args:
            item: Dictionary representing one JSON object

        Returns:
            List of image paths/strings, or None if no images present
        """
        # Check for single image field first (higher precedence)
        if self.image_field in item:
            image_value = item[self.image_field]
            if image_value:
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

    def load(self) -> Iterator[MultimodalLoadResult]:
        """Yield MultimodalLoadResult objects from JSON data."""
        for idx, item in enumerate(self.data, 1):
            try:
                prompt = item.get(self.prompt_field)
                request_id = item.get(self.id_field, f"req_{id(item)}")

                if prompt is None:
                    logger.debug(
                        f"Skipping item {request_id}: no '{self.prompt_field}' field"
                    )
                    continue

                # Extract images
                images = self._extract_images(item)

                # Extract additional data (exclude prompt, id, and image fields)
                excluded_fields = {
                    self.prompt_field, self.id_field,
                    self.image_field, self.images_field
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
            except Exception as e:
                logger.error(
                    f"Unexpected error processing item at index {idx}: {e}"
                )
                continue

    def __len__(self):
        return len(self.data)
