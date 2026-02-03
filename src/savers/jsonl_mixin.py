"""
JSONL processing mixins for customizable line formatting.

These mixins provide a template method pattern where users can override
specific methods to customize JSONL output formatting without rewriting
the entire saver.
"""
import json
from typing import Dict, Any
from datetime import datetime


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

    def format_result(self, result: 'SaveResult') -> Dict[str, Any]:
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

    def process_result_to_line(self, result: 'SaveResult') -> str:
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
