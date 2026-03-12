"""
DataLoader base module.

Provides abstract base class for all data loaders.
"""
from abc import ABC, abstractmethod
from typing import Iterator, Dict, Any, Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class LoadResult:
    """
    Standardized data structure for loader output.

    Attributes:
        messages: OpenAI API format messages
        request_id: Unique identifier for the request
        additional_data: Any extra data to be saved with results
        resume_key: Stable tuple for bitmap-style resume backends
        dispatch_cost: Estimated cost used for admission and load balancing
    """
    messages: list
    request_id: str
    additional_data: Optional[Dict[str, Any]] = None
    resume_key: Optional[Tuple[str, int, int]] = None
    dispatch_cost: Optional[float] = None


class DataLoader(ABC):
    """
    Abstract base class for data loading.

    Users inherit from this class to implement custom data loading logic.
    The loader must be thread-safe for concurrent data reading.

    This base class provides integration points with various mixins:
    - StreamingLoaderMixin: For on-demand data loading
    - MessagesBuilderMixin: For flexible message construction
    - PromptExtractorMixin: For customizable prompt extraction
    - MultimodalInputMixin: For multimodal input handling

    Example (basic):
        >>> class MyLoader(DataLoader):
        ...     def _initialize(self):
        ...         self.data = [{"text": "Hello", "id": "1"}]
        ...
        ...     def load(self):
        ...         for item in self.data:
        ...             yield LoadResult(
        ...                 messages=self.build_messages(item["text"]),
        ...                 request_id=item["id"]
        ...             )

    Example (with mixins):
        >>> class MyStreamingLoader(StreamingLoaderMixin,
        ...                         MessagesBuilderMixin,
        ...                         DataLoader):
        ...     def _initialize(self):
        ...         self._initialize_streaming()
        ...         self.input_path = Path(self.config['input_path'])
        ...
        ...     def _discover_sources(self):
        ...         return list(self.input_path.glob('*.jsonl'))
        ...
        ...     def _process_source(self, source):
        ...         with open(source) as f:
        ...             for line in f:
        ...                 item = json.loads(line)
        ...                 prompt = self.extract_prompt(item)
        ...                 messages = self.build_messages(prompt, item)
        ...                 yield LoadResult(messages, item['id'], item)
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

    # ===== Integration hooks for mixins =====

    def build_messages(
        self,
        prompt: str,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Build OpenAI-format messages from a prompt.

        This method provides a default implementation. Subclasses can
        override this directly or use MessagesBuilderMixin for more options.

        Args:
            prompt: The text prompt
            additional_data: Optional additional data from the source

        Returns:
            List of message dictionaries in OpenAI format
        """
        return [{"role": "user", "content": prompt}]

    def extract_prompt(self, item: Dict[str, Any]) -> Optional[str]:
        """
        Extract prompt text from a data item.

        This method provides a default implementation. Subclasses can
        override this directly or use PromptExtractorMixin for more options.

        Args:
            item: Dictionary representing a data item

        Returns:
            Prompt string, or None if not found
        """
        prompt_field = getattr(self, 'prompt_field', 'prompt')
        return item.get(prompt_field)

    def estimate_dispatch_cost(
        self,
        prompt: Optional[str],
        additional_data: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Estimate dispatch cost for routing heuristics.

        This lightweight heuristic prefers using upstream metadata when present
        and otherwise falls back to prompt size.
        """
        additional_data = additional_data or {}

        for key in ("dispatch_cost", "estimated_tokens", "input_tokens", "prompt_tokens"):
            value = additional_data.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return float(value)

        prompt_text = prompt or ""
        return float(max(1, len(prompt_text) // 4))

    # ===== Standard dunder methods =====

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
