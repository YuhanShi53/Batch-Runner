"""
Console result saver implementation.

Prints inference results to console for debugging and testing.
"""
import threading
from typing import Dict, Any

from .base import ResultSaver, SaveResult


class ConsoleResultSaver(ResultSaver):
    """
    Print inference results to console.

    Useful for debugging and testing.

    Configuration:
        show_details: Whether to print full model output (default: false)
        separator: Separator between outputs (default: "-" * 60)
    """

    def _initialize(self):
        """Initialize console saver."""
        self.show_details = self.config.get('show_details', False)
        self.separator = self.config.get('separator', '-' * 60)
        self._lock = threading.Lock()

    def save(self, result: SaveResult):
        """
        Print result to console.

        Thread-safe for concurrent writes.
        """
        with self._lock:
            print(self.separator)
            print(f"Request ID: {result.request_id}")

            if result.error:
                print(f"Error: {result.error}")
            else:
                # Extract content
                contents = self.extract_contents(result)
                content = contents[0] or '' if contents else ''

                print(f"Response: {content[:200]}{'...' if len(content) > 200 else ''}")
                if len(contents) > 1:
                    print(f"Choices: {len(contents)}")

                # Show usage info
                usage = result.model_output.get('usage', {}) if result.model_output else {}
                if usage:
                    print(f"Tokens: {usage.get('total_tokens', 'N/A')}")

            # Show additional data
            if result.additional_data:
                print(f"Additional: {result.additional_data}")

            if self.show_details and result.model_output:
                print(f"Full Output: {result.model_output}")

            print(self.separator)
            print()

    def cleanup(self):
        """No cleanup needed for console saver."""
        pass
