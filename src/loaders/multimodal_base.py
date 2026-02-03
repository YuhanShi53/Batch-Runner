"""
Multimodal data loader base module.

Provides abstract base class for loaders that handle both text and image data.
Supports OpenAI-compatible vision API format.
"""
from abc import abstractmethod
from typing import Iterator, Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass
import base64
import logging

from .base import DataLoader, LoadResult


logger = logging.getLogger(__name__)


@dataclass
class MultimodalLoadResult(LoadResult):
    """
    Extended LoadResult for multimodal data.

    Attributes:
        messages: OpenAI API format messages (can contain image content)
        request_id: Unique identifier for the request
        additional_data: Any extra data to be saved with results
        images: List of image paths or base64-encoded strings
    """
    images: Optional[List[str]] = None


class MultimodalDataLoader(DataLoader):
    """
    Abstract base class for multimodal data loading.

    Handles loading of text and image data, encoding images to base64,
    and constructing OpenAI-compatible vision API messages.

    Image handling:
    - Supports both file paths and base64-encoded strings
    - Automatically encodes image files to base64
    - Constructs multimodal content arrays with text and images

    Example:
        >>> class MyMultimodalLoader(MultimodalDataLoader):
        ...     def _initialize(self):
        ...         self.data = [
        ...             {"text": "What's in this image?", "image": "path/to/image.jpg"}
        ...         ]
        ...
        ...     def load(self):
        ...         for item in self.data:
        ...             yield self._create_multimodal_result(
        ...                 text=item["text"],
        ...                 images=[item["image"]],
        ...                 request_id=item.get("id", "default")
        ...             )
    """

    def _initialize(self):
        """
        Initialize loader-specific resources.

        Subclasses should call super()._initialize() if they override this method.
        """
        self.image_base_dir = Path(self.config.get('image_base_dir', ''))
        self.encode_images = self.config.get('encode_images', True)

        # Validate image_base_dir if specified
        if self.image_base_dir and not self.image_base_dir.exists():
            logger.warning(f"image_base_dir does not exist: {self.image_base_dir}")

    def _encode_image_to_base64(self, image_path: str) -> str:
        """
        Encode an image file to base64 string.

        Args:
            image_path: Path to image file (relative or absolute)

        Returns:
            Base64-encoded image string with data URI scheme

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image encoding fails
        """
        # Resolve path (handle relative paths from image_base_dir)
        path = Path(image_path)
        if not path.is_absolute() and self.image_base_dir:
            path = self.image_base_dir / path

        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        try:
            # Determine MIME type
            mime_type = self._get_mime_type(path.suffix)

            # Read and encode file
            with open(path, 'rb') as f:
                image_data = f.read()
                base64_data = base64.b64encode(image_data).decode('utf-8')
                return f"data:{mime_type};base64,{base64_data}"
        except Exception as e:
            raise ValueError(f"Failed to encode image {path}: {e}")

    def _get_mime_type(self, file_extension: str) -> str:
        """
        Get MIME type for image file.

        Args:
            file_extension: File extension (e.g., '.jpg', '.png')

        Returns:
            MIME type string

        Raises:
            ValueError: If file type is not supported
        """
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.bmp': 'image/bmp',
        }

        ext = file_extension.lower()
        if ext not in mime_types:
            raise ValueError(f"Unsupported image format: {ext}")

        return mime_types[ext]

    def _process_images(self, images: List[str]) -> List[str]:
        """
        Process a list of images.

        If encode_images is True, converts file paths to base64.
        If already base64-encoded (starts with "data:"), returns as-is.

        Args:
            images: List of image paths or base64 strings

        Returns:
            List of base64-encoded image strings (if encode_images=True)
            or original image paths/strings
        """
        processed = []

        for img in images:
            if not img:
                continue

            # Check if already base64-encoded
            if img.startswith('data:'):
                processed.append(img)
            elif self.encode_images:
                try:
                    encoded = self._encode_image_to_base64(img)
                    processed.append(encoded)
                except (FileNotFoundError, ValueError) as e:
                    logger.warning(f"Failed to encode image {img}: {e}. Using original path.")
                    processed.append(img)
            else:
                # Use as-is (path or URL)
                processed.append(img)

        return processed

    def _create_multimodal_content(
        self,
        text: str,
        images: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Create OpenAI-compatible multimodal content array.

        Args:
            text: Text content
            images: List of image paths or base64 strings

        Returns:
            Content array suitable for OpenAI vision API

        Example:
            >>> content = self._create_multimodal_content(
            ...     text="Describe this image",
            ...     images=["path/to/image.jpg"]
            ... )
            >>> # Returns:
            >>> # [
            >>> #     {"type": "text", "text": "Describe this image"},
            >>> #     {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
            >>> # ]
        """
        content = []

        # Add image content
        if images:
            processed_images = self._process_images(images)
            for img in processed_images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": img
                    }
                })

        # Add text content
        if text:
            content.append({
                "type": "text",
                "text": text
            })

        return content

    def _create_multimodal_result(
        self,
        text: str,
        images: Optional[List[str]] = None,
        request_id: str = "default",
        additional_data: Optional[Dict[str, Any]] = None
    ) -> MultimodalLoadResult:
        """
        Create a MultimodalLoadResult with properly formatted messages.

        This is a convenience method for subclasses to create results.

        Args:
            text: Text prompt
            images: List of image paths or base64 strings
            request_id: Unique request identifier
            additional_data: Additional metadata to pass through

        Returns:
            MultimodalLoadResult with OpenAI-compatible messages
        """
        content = self._create_multimodal_content(text, images)

        messages = [{
            "role": "user",
            "content": content
        }]

        return MultimodalLoadResult(
            messages=messages,
            request_id=request_id,
            additional_data=additional_data,
            images=images
        )

    @abstractmethod
    def load(self) -> Iterator[MultimodalLoadResult]:
        """
        Load multimodal data and yield MultimodalLoadResult objects.

        Returns:
            Iterator of MultimodalLoadResult

        Example:
            >>> def load(self):
            ...     for item in self.data:
            ...         yield self._create_multimodal_result(
            ...             text=item["prompt"],
            ...             images=item.get("images", []),
            ...             request_id=item["id"]
            ...         )
        """
        pass
