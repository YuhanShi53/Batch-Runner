"""
JSONL processing mixins for customizable line formatting.

These mixins provide a template method pattern where users can override
specific methods to customize JSONL output formatting without rewriting
the entire saver.
"""
from datetime import datetime
from typing import Dict, Any

from .base import SaveResult
from ..utils.json_codec import json_codec


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
        projection = getattr(self, "output_projection", self.config.get("output_projection", "full"))
        include_timestamp = getattr(
            self,
            "include_timestamp",
            self.config.get("include_timestamp", projection == "full"),
        )

        if projection == "minimal":
            choices = result.model_output.get("choices", []) if result.model_output else []
            first_choice = choices[0] if choices else {}
            message = first_choice.get("message", {}) if isinstance(first_choice, dict) else {}
            output_data = {
                "request_id": result.request_id,
                "content": message.get("content"),
                "finish_reason": first_choice.get("finish_reason"),
                "usage": result.model_output.get("usage", {}) if result.model_output else {},
            }
            if result.additional_data is not None:
                output_data["additional_data"] = result.additional_data
        else:
            output_data = {
                "request_id": result.request_id,
                "model_output": result.model_output,
                "additional_data": result.additional_data,
            }
            if include_timestamp:
                output_data["timestamp"] = datetime.now().isoformat()

        if result.error:
            output_data["error"] = result.error

        output_fields = getattr(self, "output_fields", self.config.get("output_fields"))
        if output_fields:
            output_data = {
                key: value for key, value in output_data.items()
                if key in output_fields
            }

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
        return json_codec.dumps_text(output_data)

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
