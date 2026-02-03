"""
Model adapter base module.

Provides abstract base class for model API adapters.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import requests


class ModelAdapter(ABC):
    """
    Abstract base class for model API adapters.

    Adapters handle:
    - Building request payloads for specific API formats
    - Parsing responses to extract model output
    - URL construction for different endpoint structures

    Example:
        >>> class MyCustomAdapter(ModelAdapter):
        ...     def build_request(self, config, messages):
        ...         return {"prompt": messages[0]["content"]}
        ...
        ...     def parse_response(self, response):
        ...         return {"text": response.json()["output"]}
    """

    @abstractmethod
    def build_request(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Build request payload for the API.

        Args:
            model_name: Model identifier
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters (top_p, frequency_penalty, etc.)

        Returns:
            Request payload dictionary
        """
        pass

    @abstractmethod
    def parse_response(self, response: requests.Response) -> Dict[str, Any]:
        """
        Parse API response to extract model output.

        Args:
            response: HTTP response object

        Returns:
            Standardized model output dict with keys:
            - choices: List of choice dicts with 'message' and 'finish_reason'
            - usage: Dict with 'total_tokens' (optional)
        """
        pass

    def get_chat_url(self, base_url: str) -> str:
        """
        Get the chat completions endpoint URL.

        Args:
            base_url: Base URL of the server (e.g., "http://localhost:8000")

        Returns:
            Full URL to the chat completions endpoint
        """
        return f"{base_url}/v1/chat/completions"

    def get_health_url(self, base_url: str) -> str:
        """
        Get the health check endpoint URL.

        Args:
            base_url: Base URL of the server

        Returns:
            Full URL to the health endpoint
        """
        return f"{base_url}/health"
