"""
Streaming data loader mixin for efficient on-demand data loading.

This mixin provides a template method pattern for implementing streaming
data loaders that process data on-demand rather than loading everything
into memory up front.
"""
from typing import Iterator, Dict, Any, Optional, List, Tuple
from pathlib import Path
import logging

from .base import LoadResult


logger = logging.getLogger(__name__)


class StreamingLoaderMixin:
    """
    Mixin class that provides streaming data loading capabilities.

    This mixin implements the Template Method pattern for streaming data
    processing, allowing subclasses to customize how data is discovered
    and processed without rewriting the entire streaming logic.

    Key benefits:
    - Constant memory usage regardless of dataset size
    - Immediate processing starts as soon as first data is available
    - Better resource utilization (I/O and computation happen concurrently)

    Usage:
        class MyStreamingLoader(StreamingLoaderMixin, DataLoader):
            def _discover_sources(self):
                # Return list of data sources (files, URLs, etc.)
                return ["/path/to/file1.jsonl", "/path/to/file2.jsonl"]

            def _process_source(self, source):
                # Yield LoadResult objects from a single source
                with open(source) as f:
                    for line in f:
                        yield self._parse_line(line)

    Configuration:
        streaming: Enable streaming mode (default: True)
        stream_queue_size: Internal buffer size (not used in mixin, for runner)
    """

    def _initialize_streaming(self):
        """
        Initialize streaming-specific configuration.

        Subclasses should call this in their _initialize() method.
        """
        self.streaming = self.config.get('streaming', True)
        if not self.streaming:
            logger.info(f"{self.__class__.__name__}: Non-streaming mode (batch loading)")

    def _discover_sources(self) -> List[Any]:
        """
        Discover available data sources (files, streams, etc.).

        This method should be overridden by subclasses to return a list
        of data sources to be processed sequentially.

        Returns:
            List of source identifiers (file paths, URLs, etc.)

        Example:
            def _discover_sources(self):
                input_dir = Path(self.config['input_dir'])
                return sorted(input_dir.rglob('*.jsonl'))
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _discover_sources()"
        )

    def _process_source(self, source: Any) -> Iterator[LoadResult]:
        """
        Process a single data source and yield LoadResult objects.

        This method should be overridden by subclasses to process one
        data source at a time.

        Args:
            source: A single source from _discover_sources()

        Yields:
            LoadResult objects from this source

        Example:
            def _process_source(self, source):
                with open(source) as f:
                    for line in f:
                        yield self._create_load_result(line)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _process_source()"
        )

    def _should_skip_source(self, source: Any) -> bool:
        """
        Determine if a source should be skipped.

        Override this method to implement custom source filtering.
        Default implementation doesn't skip any sources.

        Args:
            source: A source from _discover_sources()

        Returns:
            True to skip this source, False otherwise

        Example:
            def _should_skip_source(self, source):
                # Skip temporary files
                return str(source).endswith('.tmp')
        """
        return False

    def _on_source_start(self, source: Any):
        """
        Hook called before processing a source.

        Override for custom logging, initialization, etc.

        Args:
            source: A source from _discover_sources()

        Example:
            def _on_source_start(self, source):
                logger.info(f"Processing {source}")
        """
        pass

    def _on_source_complete(self, source: Any, item_count: int):
        """
        Hook called after successfully processing a source.

        Override for custom logging, cleanup, etc.

        Args:
            source: The source that was processed
            item_count: Number of items yielded from this source

        Example:
            def _on_source_complete(self, source, item_count):
                logger.info(f"Completed {source}: {item_count} items")
        """
        pass

    def _on_source_error(self, source: Any, error: Exception):
        """
        Hook called when an error occurs while processing a source.

        Override for custom error handling. By default, logs a warning
        and continues to the next source.

        Args:
            source: The source that failed
            error: The exception that occurred

        Example:
            def _on_source_error(self, source, error):
                logger.error(f"Failed to process {source}: {error}")
                raise  # Re-raise to stop entire batch
        """
        logger.warning(f"Error processing {source}: {error}. Skipping.")

    def _stream_load(self) -> Iterator[LoadResult]:
        """
        Main streaming template method.

        Orchestrates the streaming process by:
        1. Discovering sources
        2. Processing each source sequentially
        3. Calling hooks at appropriate times
        4. Handling errors gracefully

        Yields:
            LoadResult objects from all sources

        This is the main entry point for streaming. Subclasses typically
        call this from their load() method.
        """
        sources = self._discover_sources()
        logger.info(f"Streaming mode: discovered {len(sources)} sources")

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
                for result in self._process_source(source):
                    item_count += 1
                    yield result

                # Call post-processing hook
                self._on_source_complete(source, item_count)

            except Exception as e:
                # Call error handling hook
                self._on_source_error(source, e)


class MessagesBuilderMixin:
    """
    Mixin class for flexible message construction.

    Provides extensible hooks for customizing how prompts are transformed
    into OpenAI-format message arrays. Supports both text-only and
    multimodal message construction.

    Usage:
        class MyLoader(MessagesBuilderMixin, DataLoader):
            def build_messages(self, prompt, additional_data=None):
                # Custom message construction logic
                messages = super().build_messages(prompt, additional_data)
                # Add system prompt, transform messages, etc.
                return messages
    """

    def build_messages(
        self,
        prompt: str,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Build OpenAI-format messages from a prompt.

        This method can be overridden to customize message construction.
        Default implementation creates a simple user message.

        Args:
            prompt: The text prompt
            additional_data: Optional additional data from the source

        Returns:
            List of message dictionaries in OpenAI format

        Example override:
            def build_messages(self, prompt, additional_data=None):
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."}
                ]
                # Add conversation history from additional_data
                if additional_data and 'history' in additional_data:
                    messages.extend(additional_data['history'])
                # Add current prompt
                messages.append({"role": "user", "content": prompt})
                return messages
        """
        return [{"role": "user", "content": prompt}]

    def build_multimodal_messages(
        self,
        text: str,
        images: Optional[List[str]] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Build OpenAI-format multimodal messages with text and images.

        This method can be overridden to customize multimodal message construction.
        Default implementation creates a user message with text and image_url content.

        Args:
            text: The text prompt
            images: List of image paths or base64 strings
            additional_data: Optional additional data from the source

        Returns:
            List of message dictionaries in OpenAI vision API format

        Example override:
            def build_multimodal_messages(self, text, images=None, additional_data=None):
                content = [{"type": "text", "text": text}]
                if images:
                    for img in images:
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": img}
                        })
                # Add additional context from additional_data
                if additional_data and 'context' in additional_data:
                    content[0]["text"] += f"\\n\\nContext: {additional_data['context']}"
                return [{"role": "user", "content": content}]
        """
        content = []

        if text:
            content.append({"type": "text", "text": text})

        if images:
            for img in images:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img}
                })

        return [{"role": "user", "content": content}]


class PromptExtractorMixin:
    """
    Mixin class for flexible prompt extraction from source data.

    Provides extensible hooks for customizing how prompts are extracted
    and transformed from raw data items.

    Usage:
        class MyLoader(PromptExtractorMixin, DataLoader):
            def extract_prompt(self, item):
                # Try multiple fields in priority order
                for field in ['prompt', 'question', 'text', 'input']:
                    if field in item:
                        return str(item[field])
                return None
    """

    def extract_prompt(self, item: Dict[str, Any]) -> Optional[str]:
        """
        Extract prompt text from a data item.

        This method should be overridden by subclasses to customize
        prompt extraction logic.

        Args:
            item: Dictionary representing a data item

        Returns:
            Prompt string, or None if no prompt found

        Example override:
            def extract_prompt(self, item):
                # Support nested prompt structure
                if 'messages' in item:
                    return item['messages'][-1]['content']
                return item.get('prompt')
        """
        prompt_field = getattr(self, 'prompt_field', 'prompt')
        return item.get(prompt_field)

    def transform_prompt(self, prompt: str, additional_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Transform a prompt after extraction.

        This method can be overridden to apply transformations to prompts,
        such as adding prefixes, formatting, etc.

        Args:
            prompt: The extracted prompt
            additional_data: Optional additional data from the source

        Returns:
            Transformed prompt string

        Example override:
            def transform_prompt(self, prompt, additional_data=None):
                # Add instruction prefix
                if additional_data and additional_data.get('task_type') == 'qa':
                    return f"Answer this question: {prompt}"
                return prompt
        """
        return prompt


class MultimodalInputMixin:
    """
    Mixin class for handling multimodal inputs (text + images).

    Provides extensible hooks for customizing image extraction and
    multimodal content construction.

    Usage:
        class MyLoader(MultimodalInputMixin, MultimodalDataLoader):
            def extract_images(self, item):
                # Support custom image field structure
                if 'media' in item:
                    return [m['url'] for m in item['media'] if m['type'] == 'image']
                return super().extract_images(item)
    """

    def extract_images(self, item: Dict[str, Any]) -> Optional[List[str]]:
        """
        Extract image paths/URLs from a data item.

        This method should be overridden by subclasses to customize
        image extraction logic.

        Args:
            item: Dictionary representing a data item

        Returns:
            List of image paths/URLs, or None if no images

        Example override:
            def extract_images(self, item):
                # Support both 'image' and 'img' fields
                if 'image' in item:
                    return [item['image']] if isinstance(item['image'], str) else item['image']
                if 'img' in item:
                    return [item['img']] if isinstance(item['img'], str) else item['img']
                return None
        """
        image_field = getattr(self, 'image_field', 'image')
        images_field = getattr(self, 'images_field', 'images')

        # Check for single image field first (higher precedence)
        if image_field in item:
            image_value = item[image_field]
            if image_value:
                if isinstance(image_value, str):
                    return [image_value]
                elif isinstance(image_value, list):
                    return image_value

        # Check for multiple images field
        elif images_field in item:
            images_value = item[images_field]
            if images_value:
                if isinstance(images_value, list):
                    return images_value
                elif isinstance(images_value, str):
                    return [images_value]

        return None

    def validate_image(self, image_path: str) -> bool:
        """
        Validate an image path/URL before processing.

        Override this method to implement custom image validation.

        Args:
            image_path: Image path or URL

        Returns:
            True if image is valid, False otherwise

        Example override:
            def validate_image(self, image_path):
                # Skip remote URLs in offline mode
                if image_path.startswith('http') and self.offline_mode:
                    return False
                return True
        """
        return True
