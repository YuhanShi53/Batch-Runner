"""
OpenAI-compatible API adapter.

Handles standard OpenAI API format used by vLLM and many other providers.
"""
from typing import Dict, Any, List
import requests

from .base import ModelAdapter


class OpenAIAdapter(ModelAdapter):
    """
    Adapter for OpenAI-compatible APIs.

    This is the standard format used by:
    - OpenAI API
    - vLLM
    - Azure OpenAI
    - Many OpenAI-compatible providers
    """

    def build_request(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Build OpenAI-compatible request payload.

        Args:
            model_name: Model identifier
            messages: List of message dicts
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
            **kwargs: Additional parameters

        Returns:
            OpenAI-format request payload
        """
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }

        # Add any additional parameters
        payload.update(kwargs)

        return payload

    def parse_response(self, response: requests.Response) -> Dict[str, Any]:
        """
        Parse OpenAI-compatible response.

        Args:
            response: HTTP response object

        Returns:
            Standardized output dict matching OpenAI format
        """
        return response.json()
