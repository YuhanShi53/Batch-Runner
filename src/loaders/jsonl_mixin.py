"""
JSONL processing mixins for customizable line parsing (loaders only).

These mixins provide a template method pattern where users can override
specific methods to customize JSONL line processing without rewriting
the entire loader.

Note: For saver mixins, see src/savers/jsonl_mixin.py
"""
import json
from typing import Iterator, Dict, Any, Optional, List, Tuple, Union
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

    def parse_line(
        self, line: str, line_num: int, source: str
    ) -> Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]:
        """
        Parse a single JSONL line into a dictionary or list of dictionaries.

        This method can be overridden to customize line parsing behavior.
        Default implementation uses json.loads().

        Args:
            line: Raw line content from JSONL file
            line_num: Line number in the file (1-indexed)
            source: Source identifier (e.g., file path)

        Returns:
            - Parsed dictionary (single item)
            - List of dictionaries (multiple items from one line)
            - None to skip this line

        Raises:
            json.JSONDecodeError: If JSON parsing fails (default behavior)

        Example override for multiple items (样本裂变):
            def parse_line(self, line, line_num, source):
                obj = json.loads(line)
                # 从单个样本生成多个请求
                if 'variations' in obj:
                    base_id = obj.get('id', f'line_{line_num}')
                    return [
                        {
                            **obj,
                            'prompt': variation,
                            'id': f'{base_id}_var_{i}',
                            '_variation_index': i
                        }
                        for i, variation in enumerate(obj['variations'])
                    ]
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
        Extract request_id from parsed item, with hash-based fallback.

        Priority:
        1. Use id_field if present in data (maintains backward compatibility)
        2. Generate hash-based ID from content (when id is missing)
        3. Fall back to default_id (line number/index, rarely used)

        This ensures:
        - Existing data with 'id' field keeps original IDs
        - New data without 'id' gets deterministic hash-based IDs
        - Same content always generates same hash (for resume feature)

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

        # Try to get ID from field (priority #1 - backward compatibility)
        if id_field in item and item[id_field] is not None:
            return str(item[id_field])

        # No ID field - generate hash-based ID (priority #2 - new behavior)
        return self.generate_request_id_hash(item)

    def generate_request_id_hash(self, item: Dict[str, Any]) -> str:
        """
        Generate a stable hash-based request_id from item content.

        Uses SHA-256 for excellent collision resistance and deterministic output.

        Creates a deterministic hash using:
        - The prompt content
        - Key fields from additional_data
        - Excludes volatile fields like timestamps

        Returns:
            Hexadecimal hash string (64 chars, e.g., full SHA-256 output)
        """
        import hashlib
        import json

        # Get the prompt field
        prompt_field = getattr(self, 'prompt_field', 'prompt')
        prompt = item.get(prompt_field, '')

        # Create a deterministic string representation
        # Sort keys for consistent hashing
        hashable_content = json.dumps({
            'prompt': prompt,
            'data': {k: v for k, v in sorted(item.items())
                    if k != prompt_field and k != 'id'}
        }, sort_keys=True)

        # Use SHA-256 for full 64-character hash
        return hashlib.sha256(hashable_content.encode()).hexdigest()

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
    ) -> Iterator[Optional[LoadResult]]:
        """
        Process a single JSONL line into one or more LoadResults.

        Now always returns an iterator for consistent handling of single/multiple items.

        This is the main template method that orchestrates the parsing process.
        Override individual methods above to customize behavior.

        Args:
            line: Raw line content from JSONL file
            line_num: Line number in the file (1-indexed)
            source: Source identifier (e.g., file path)
            default_id: Default ID to use if extraction fails

        Returns:
            Iterator that yields LoadResult objects (or nothing if skipped)

        Yields:
            LoadResult objects (1+), or nothing if line should be skipped

        Raises:
            json.JSONDecodeError: If JSON parsing fails
        """
        try:
            # Parse the line
            item_or_items = self.parse_line(line, line_num, source)
            if item_or_items is None:
                return

            # Normalize to list for uniform processing
            items = [item_or_items] if isinstance(item_or_items, dict) else item_or_items

            # Handle empty list (treat same as None)
            if not items:
                return

            # Process each item
            for idx, item in enumerate(items):
                # Check if we should skip this item
                if self.should_skip_item(item):
                    logger.debug(f"Skipping item in {source}:{line_num}[{idx}]")
                    continue

                # Extract prompt (uses PromptExtractorMixin.extract_prompt if available,
                # otherwise uses the method defined in this class)
                prompt = self.extract_prompt(item)
                if prompt is None:
                    logger.debug(f"Skipping item in {source}:{line_num}[{idx}]: no prompt found")
                    continue

                # Transform prompt if transform_prompt is available
                if hasattr(self, 'transform_prompt'):
                    prompt = self.transform_prompt(prompt, item)

                # For multi-item case, add suffix to default_id
                if len(items) > 1:
                    item_default_id = f"{default_id}_{idx}"
                else:
                    item_default_id = default_id

                # Extract request_id
                request_id = self.extract_request_id(item, item_default_id)

                # Extract additional data
                additional_data = self.extract_additional_data(item)

                # Build messages (uses MessagesBuilderMixin.build_messages if available)
                if hasattr(self, 'build_messages'):
                    messages = self.build_messages(prompt, additional_data)
                else:
                    messages = [{"role": "user", "content": prompt}]

                yield LoadResult(
                    messages=messages,
                    request_id=request_id,
                    additional_data=additional_data or None
                )
        except Exception as e:
            # Catch unexpected errors and log them, but don't stop iteration
            logger.error(f"Unexpected error processing {source}:{line_num}: {e}")
            return
