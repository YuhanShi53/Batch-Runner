"""
JSONL processing mixins for customizable line parsing (loaders only).

These mixins provide a template method pattern where users can override
specific methods to customize JSONL line processing without rewriting
the entire loader.

Note: For saver mixins, see src/savers/jsonl_mixin.py
"""
import json
from typing import Iterator, Dict, Any, Optional, List, Tuple
from pathlib import Path
import logging

from .base import LoadResult
from .streaming_mixin import MessagesBuilderMixin, PromptExtractorMixin


logger = logging.getLogger(__name__)


class JSONLLoaderMixin(PromptExtractorMixin):
    """
    Mixin class that provides customizable JSONL line parsing for loaders.

    This mixin implements the Template Method pattern, allowing subclasses
    to customize how individual JSONL lines are parsed without rewriting
    the entire loading logic.

    Integrates with PromptExtractorMixin for flexible prompt extraction.

    Usage:
        class MyCustomLoader(JSONLLoaderMixin, DataLoader):
            def parse_line(self, line: str, line_num: int, source: str) -> Optional[Dict[str, Any]]:
                # Custom parsing logic
                obj = json.loads(line)
                # Transform the data as needed
                return obj

            def extract_prompt(self, item):
                # Custom prompt extraction
                return item.get('custom_prompt_field')
    """

    def parse_line(self, line: str, line_num: int, source: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single JSONL line into a dictionary.

        This method can be overridden to customize line parsing behavior.
        Default implementation uses json.loads().

        Args:
            line: Raw line content from JSONL file
            line_num: Line number in the file (1-indexed)
            source: Source identifier (e.g., file path)

        Returns:
            Parsed dictionary, or None to skip this line

        Raises:
            json.JSONDecodeError: If JSON parsing fails (default behavior)

        Example override:
            def parse_line(self, line: str, line_num: int, source: str) -> Optional[Dict[str, Any]]:
                # Handle list-format JSONL: [{"key": "value"}]
                obj = json.loads(line)
                if isinstance(obj, list):
                    return {"items": obj}
                return obj
        """
        try:
            return json.loads(line)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in {source}:{line_num}: {e}")
            raise

    def should_skip_item(self, item: Dict[str, Any]) -> bool:
        """
        Determine if an item should be skipped.

        Override this method to implement custom filtering logic.
        Default implementation doesn't skip any items.

        Args:
            item: Parsed dictionary from parse_line()

        Returns:
            True to skip this item, False otherwise

        Example override:
            def should_skip_item(self, item: Dict[str, Any]) -> bool:
                # Skip items without required fields
                return 'prompt' not in item or 'id' not in item
        """
        return False

    def extract_request_id(self, item: Dict[str, Any], default_id: str) -> str:
        """
        Extract request_id from parsed item.

        Override this method to customize request ID extraction.
        Default implementation uses configured id_field.

        Args:
            item: Parsed dictionary from parse_line()
            default_id: Default ID to use if extraction fails

        Returns:
            Request ID string

        Example override:
            def extract_request_id(self, item: Dict[str, Any], default_id: str) -> str:
                # Use composite key
                return f"{item.get('doc_id')}_{item.get('line_num', default_id)}"
        """
        id_field = getattr(self, 'id_field', 'id')
        request_id = item.get(id_field, default_id)
        return str(request_id)

    def extract_additional_data(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract additional data from parsed item.

        Override this method to customize what data is preserved.
        Default implementation excludes prompt_field and id_field.

        Args:
            item: Parsed dictionary from parse_line()

        Returns:
            Dictionary of additional data

        Example override:
            def extract_additional_data(self, item: Dict[str, Any]) -> Dict[str, Any]:
                # Preserve all fields except prompt
                excluded = {'prompt', 'id'}
                return {k: v for k, v in item.items() if k not in excluded}
        """
        excluded_fields = {
            getattr(self, 'prompt_field', 'prompt'),
            getattr(self, 'id_field', 'id')
        }
        # Remove None values
        return {
            k: v for k, v in item.items()
            if k not in excluded_fields and v is not None
        }

    def process_line_to_load_result(
        self,
        line: str,
        line_num: int,
        source: str,
        default_id: str
    ) -> Optional[LoadResult]:
        """
        Process a single JSONL line into a LoadResult.

        This is the main template method that orchestrates the parsing process.
        Override individual methods above to customize behavior.

        Args:
            line: Raw line content from JSONL file
            line_num: Line number in the file (1-indexed)
            source: Source identifier (e.g., file path)
            default_id: Default ID to use if extraction fails

        Returns:
            LoadResult object, or None if line should be skipped

        Raises:
            json.JSONDecodeError: If JSON parsing fails
        """
        try:
            # Parse the line
            item = self.parse_line(line, line_num, source)
            if item is None:
                return None

            # Check if we should skip this item
            if self.should_skip_item(item):
                logger.debug(f"Skipping item in {source}:{line_num}")
                return None

            # Extract prompt (uses PromptExtractorMixin.extract_prompt if available,
            # otherwise uses the method defined in this class)
            prompt = self.extract_prompt(item)
            if prompt is None:
                logger.debug(f"Skipping item in {source}:{line_num}: no prompt found")
                return None

            # Transform prompt if transform_prompt is available
            if hasattr(self, 'transform_prompt'):
                prompt = self.transform_prompt(prompt, item)

            # Extract request_id
            request_id = self.extract_request_id(item, default_id)

            # Extract additional data
            additional_data = self.extract_additional_data(item)

            # Build messages (uses MessagesBuilderMixin.build_messages if available)
            if hasattr(self, 'build_messages'):
                messages = self.build_messages(prompt, additional_data)
            else:
                messages = [{"role": "user", "content": prompt}]

            return LoadResult(
                messages=messages,
                request_id=request_id,
                additional_data=additional_data or None
            )
        except Exception as e:
            # Catch unexpected errors and log them, but don't stop iteration
            logger.error(f"Unexpected error processing {source}:{line_num}: {e}")
            return None
