"""
Simple completion API adapter.

For APIs that use simple prompt-to-completion format without message structure.
"""
from typing import Dict, Any, List

from .base import ModelAdapter


class SimpleAdapter(ModelAdapter):
    """
    Adapter for simple prompt-based APIs.

    Some APIs use a simpler format:
    Request: {"prompt": "your text here", "max_tokens": 100}
    Response: {"text": "generated response", "tokens_used": 50}

    This adapter converts OpenAI-style messages to that format.
    """

    def build_request(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Build simple prompt-based request.

        Concatenates system and user messages into a single prompt.
        """
        # Build prompt from messages
        prompt_parts = []
        for msg in messages:
            if msg["role"] == "system":
                prompt_parts.append(f"System: {msg['content']}")
            elif msg["role"] == "user":
                prompt_parts.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                prompt_parts.append(f"Assistant: {msg['content']}")

        prompt = "\n".join(prompt_parts)

        return {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

    def parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse simple response to OpenAI-compatible format.

        Converts:
        {"text": "response", "tokens_used": 50}
        To:
        {"choices": [{"message": {"content": "response"}}], "usage": {"total_tokens": 50}}

        Args:
            response: Response dict (already parsed from JSON)
        """
        data = response

        # Extract text from common field names
        text = data.get("text") or data.get("completion") or data.get("output") or ""
        tokens = data.get("tokens_used") or data.get("tokens") or 0

        # Convert to OpenAI format
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop"
                }
            ],
            "usage": {"total_tokens": tokens}
        }

    def get_chat_url(self, base_url: str) -> str:
        """Override to use /generate endpoint."""
        return f"{base_url}/generate"

    def get_health_url(self, base_url: str) -> str:
        """Override to use /status endpoint."""
        return f"{base_url}/status"
