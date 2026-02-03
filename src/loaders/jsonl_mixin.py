"""
JSONL processing mixins for customizable line parsing.

These mixins provide a template method pattern where users can override
specific methods to customize JSONL line processing without rewriting
the entire loader/saver.
"""
import json
from typing import Iterator, Dict, Any, Optional, List, Tuple
from pathlib import Path
import logging

from .base import LoadResult, SaveResult


logger = logging.getLogger(__name__)


class JSONLLoaderMixin:
    """
    Mixin class that provides customizable JSONL line parsing for loaders.

    This mixin implements the Template Method pattern, allowing subclasses
    to customize how individual JSONL lines are parsed without rewriting
    the entire loading logic.

    Usage:
        class MyCustomLoader(JSONLLoaderMixin, DataLoader):
            def parse_line(self, line: str, line_num: int, source: str) -> Optional[Dict[str, Any]]:
                # Custom parsing logic
                obj = json.loads(line)
                # Transform the data as needed
                return obj
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

    def extract_prompt(self, item: Dict[str, Any]) -> Optional[str]:
        """
        Extract prompt text from parsed item.

        Override this method to customize prompt extraction.
        Default implementation uses configured prompt_field.

        Args:
            item: Parsed dictionary from parse_line()

        Returns:
            Prompt string, or None if not found

        Example override:
            def extract_prompt(self, item: Dict[str, Any]) -> Optional[str]:
                # Try multiple fields in order
                for field in ['prompt', 'question', 'text', 'input']:
                    if field in item:
                        return str(item[field])
                return None
        """
        prompt_field = getattr(self, 'prompt_field', 'prompt')
        return item.get(prompt_field)

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
        # Parse the line
        item = self.parse_line(line, line_num, source)
        if item is None:
            return None

        # Check if we should skip this item
        if self.should_skip_item(item):
            logger.debug(f"Skipping item in {source}:{line_num}")
            return None

        # Extract prompt
        prompt = self.extract_prompt(item)
        if prompt is None:
            logger.debug(f"Skipping item in {source}:{line_num}: no prompt found")
            return None

        # Extract request_id
        request_id = self.extract_request_id(item, default_id)

        # Extract additional data
        additional_data = self.extract_additional_data(item)

        return LoadResult(
            messages=[{"role": "user", "content": prompt}],
            request_id=request_id,
            additional_data=additional_data or None
        )


class JSONLSaverMixin:
    """
    Mixin class that provides customizable JSONL line formatting for savers.

    This mixin implements the Template Method pattern, allowing subclasses
    to customize how results are formatted for JSONL output.

    Usage:
        class MyCustomSaver(JSONLSaverMixin, ResultSaver):
            def format_result(self, result: SaveResult) -> Dict[str, Any]:
                # Custom formatting logic
                return {
                    "id": result.request_id,
                    "output": result.model_output['choices'][0]['message']['content']
                }
    """

    def format_result(self, result: SaveResult) -> Dict[str, Any]:
        """
        Format a SaveResult into a dictionary for JSONL output.

        This method can be overridden to customize output format.
        Default implementation creates a standard structure.

        Args:
            result: SaveResult object containing model output and metadata

        Returns:
            Dictionary to be serialized as JSON

        Example override:
            def format_result(self, result: SaveResult) -> Dict[str, Any]:
                content = result.model_output['choices'][0]['message']['content']
                return {
                    "id": result.request_id,
                    "response": content,
                    "tokens": result.model_output.get('usage', {}).get('total_tokens', 0)
                }
        """
        from datetime import datetime

        output_data = {
            'request_id': result.request_id,
            'model_output': result.model_output,
            'additional_data': result.additional_data,
            'timestamp': datetime.now().isoformat()
        }

        if result.error:
            output_data['error'] = result.error

        return output_data

    def serialize_output(self, output_data: Dict[str, Any]) -> str:
        """
        Serialize formatted output dictionary to JSON string.

        Override this method to customize serialization behavior.
        Default implementation uses json.dumps().

        Args:
            output_data: Dictionary from format_result()

        Returns:
            JSON string

        Example override:
            def serialize_output(self, output_data: Dict[str, Any]) -> str:
                # Custom serialization with specific options
                return json.dumps(output_data, ensure_ascii=False, indent=None)
        """
        return json.dumps(output_data, ensure_ascii=False)

    def process_result_to_line(self, result: SaveResult) -> str:
        """
        Process a SaveResult into a JSONL line string.

        This is the main template method that orchestrates the formatting process.
        Override individual methods above to customize behavior.

        Args:
            result: SaveResult object containing model output and metadata

        Returns:
            JSON string ready to be written to file (without newline)

        Example:
            >>> line = self.process_result_to_line(result)
            >>> file.write(line + '\\n')
        """
        output_data = self.format_result(result)
        return self.serialize_output(output_data)
