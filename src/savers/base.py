"""
ResultSaver base module.

Provides abstract base class for all result savers.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class SaveResult:
    """
    Standardized structure for save results.

    Attributes:
        request_id: Unique identifier for the request
        model_output: Complete vLLM response
        additional_data: Extra data from the loader
        error: Error message if the request failed
    """
    request_id: str
    model_output: Dict[str, Any]
    additional_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ResultSaver(ABC):
    """
    Abstract base class for result saving.

    Users inherit from this class to implement custom result saving logic.
    The saver must be thread-safe for concurrent writing.

    This base class provides integration points with various mixins:
    - StreamingSaverMixin: For immediate write-back
    - OutputFormatterMixin: For flexible output formatting
    - MultimodalOutputMixin: For multimodal output handling
    - BatchWriterMixin: For batched writing optimization

    Example (basic):
        >>> class MySaver(ResultSaver):
        ...     def _initialize(self):
        ...         self.file = open(self.config['output_path'], 'w')
        ...
        ...     def save(self, result: SaveResult):
        ...         output = self.format_output(result)
        ...         self.file.write(json.dumps(output) + '\\n')

    Example (with mixins):
        >>> class MyStreamingSaver(StreamingSaverMixin,
        ...                         OutputFormatterMixin,
        ...                         ResultSaver):
        ...     def _initialize(self):
        ...         self._initialize_streaming()
        ...         self.output_path = Path(self.config['output_path'])
        ...
        ...     def _get_output_path(self, result):
        ...         return self.output_path
        ...
        ...     def _write_result(self, path, formatted_data):
        ...         with open(path, 'a') as f:
        ...             f.write(json.dumps(formatted_data) + '\\n')
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the saver with configuration.

        Args:
            config: Configuration dictionary from YAML
        """
        self.config = config
        self._initialize()

    @abstractmethod
    def _initialize(self):
        """
        Initialize saver-specific resources (files, databases, etc.).
        Called during __init__.
        """
        pass

    @abstractmethod
    def save(self, result: SaveResult):
        """
        Save a single result.

        Args:
            result: SaveResult object containing model output and metadata

        Example:
            >>> def save(self, result: SaveResult):
            ...     self.file.write(f"{result.request_id}: {result.model_output}\\n")
        """
        pass

    def save_batch(self, results: List[SaveResult]):
        """
        Save multiple results at once (optional optimization).

        Default implementation calls save() for each result.
        Override this for batch-specific optimizations (e.g., bulk database inserts).

        Args:
            results: List of SaveResult objects
        """
        for result in results:
            self.save(result)

    # ===== Integration hooks for mixins =====

    def format_output(self, result: SaveResult) -> Dict[str, Any]:
        """
        Format a SaveResult into a dictionary for output.

        This method provides a default implementation. Subclasses can
        override this directly or use OutputFormatterMixin for more options.

        Args:
            result: SaveResult object containing model output and metadata

        Returns:
            Dictionary to be serialized/written to output
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

    def extract_content(self, result: SaveResult) -> Optional[str]:
        """
        Extract the main content from a SaveResult.

        This is a convenience method for subclasses.

        Args:
            result: SaveResult object

        Returns:
            Generated text content, or None if not found
        """
        try:
            return result.model_output['choices'][0]['message']['content']
        except (KeyError, IndexError, TypeError):
            return None

    # ===== Standard dunder methods =====

    def is_completed(self, request_id: str) -> bool:
        """
        Check if a request_id has already been saved to the output.

        This method is used for resume functionality - it reads the output file
        and checks if the given request_id has already been processed.

        Default implementation loads all completed IDs on first call and caches them.
        Override this method for custom behavior.

        Args:
            request_id: The request ID to check (with or without _rollout_N suffix)

        Returns:
            True if the request_id (or its base ID for rollouts) is in output
        """
        # Lazy load completed IDs on first call
        if not hasattr(self, '_completed_ids'):
            self._completed_ids = self._load_completed_ids()

        # Handle rollout IDs: strip _rollout_N suffix
        base_id = request_id.split('_rollout_')[0]
        return base_id in self._completed_ids

    def _load_completed_ids(self) -> set:
        """
        Load all completed request_ids from the output file.

        Should be overridden by subclasses to handle their specific format.
        Default implementation returns empty set (no resume support).

        Returns:
            Set of completed request_id strings (base IDs, without rollout suffix)
        """
        return set()

    def cleanup(self):
        """
        Clean up resources (close files, connections, etc.).
        Called after batch processing completes.
        """
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
        return False
