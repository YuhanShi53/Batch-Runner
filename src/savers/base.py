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

    Example:
        >>> class MySaver(ResultSaver):
        ...     def _initialize(self):
        ...         self.file = open(self.config['output_path'], 'w')
        ...
        ...     def save(self, result: SaveResult):
        ...         content = result.model_output['choices'][0]['message']['content']
        ...         self.file.write(f"{result.request_id}\\t{content}\\n")
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
