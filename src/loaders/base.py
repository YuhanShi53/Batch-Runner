"""
DataLoader base module.

Provides abstract base class for all data loaders.
"""
from abc import ABC, abstractmethod
from typing import Iterator, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class LoadResult:
    """
    Standardized data structure for loader output.

    Attributes:
        messages: OpenAI API format messages
        request_id: Unique identifier for the request
        additional_data: Any extra data to be saved with results
    """
    messages: list
    request_id: str
    additional_data: Optional[Dict[str, Any]] = None


class DataLoader(ABC):
    """
    Abstract base class for data loading.

    Users inherit from this class to implement custom data loading logic.
    The loader must be thread-safe for concurrent data reading.

    Example:
        >>> class MyLoader(DataLoader):
        ...     def _initialize(self):
        ...         self.data = [{"text": "Hello", "id": "1"}]
        ...
        ...     def load(self):
        ...         for item in self.data:
        ...             yield LoadResult(
        ...                 messages=[{"role": "user", "content": item["text"]}],
        ...                 request_id=item["id"]
        ...             )
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the loader with configuration.

        Args:
            config: Configuration dictionary from YAML
        """
        self.config = config
        self._initialize()

    @abstractmethod
    def _initialize(self):
        """
        Initialize loader-specific resources (files, connections, etc.).
        Called during __init__.
        """
        pass

    @abstractmethod
    def load(self) -> Iterator[LoadResult]:
        """
        Load data and yield LoadResult objects.

        Returns:
            Iterator of LoadResult containing messages, request_id, and additional_data

        Example:
            >>> for item in self.load():
            ...     yield LoadResult(
            ...         messages=[{"role": "user", "content": "Hello"}],
            ...         request_id="req_001",
            ...         additional_data={"source": "file1.txt"}
            ...     )
        """
        pass

    def __iter__(self):
        """Make the loader iterable."""
        return self.load()

    def __len__(self) -> int:
        """
        Return the total number of items (optional).

        Returns:
            Total items if known, otherwise raises NotImplementedError
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support __len__")

    def cleanup(self):
        """
        Clean up resources (close files, connections, etc.).
        Called after batch processing completes.
        """
        pass
