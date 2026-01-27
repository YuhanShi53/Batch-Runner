"""
Simple prompt list data loader implementation.

Loads inference requests from a list of prompts in configuration.
"""
from typing import Iterator, Dict, Any, List

from .base import DataLoader, LoadResult


class PromptListLoader(DataLoader):
    """
    Load inference requests from a list of prompts in configuration.

    Configuration:
        prompts: List of prompt strings or list of dicts with 'prompt' and 'id' fields

    Example configuration:
        prompts:
            - "What is the capital of France?"
            - "Explain quantum computing"
            - prompt: "Write a poem about AI"
              id: "poem_001"

    Or as list of strings:
        prompts:
            - "Hello, how are you?"
            - "What is AI?"
    """

    def _initialize(self):
        """Initialize prompt list loader."""
        prompts = self.config.get('prompts')

        if not prompts:
            raise ValueError("Configuration must contain 'prompts' field")

        if not isinstance(prompts, list):
            raise ValueError("'prompts' must be a list")

        # Normalize to list of dicts
        self.data = []
        for idx, item in enumerate(prompts):
            if isinstance(item, str):
                self.data.append({'prompt': item, 'id': f'prompt_{idx}'})
            elif isinstance(item, dict):
                prompt = item.get('prompt')
                if not prompt:
                    continue
                self.data.append({
                    'prompt': prompt,
                    'id': item.get('id', f'prompt_{idx}')
                })

    def load(self) -> Iterator[LoadResult]:
        """Yield LoadResult objects from prompt list."""
        for item in self.data:
            additional_data = {k: v for k, v in item.items() if k not in ['prompt', 'id']}

            yield LoadResult(
                messages=[{"role": "user", "content": item['prompt']}],
                request_id=item['id'],
                additional_data=additional_data or None
            )

    def __len__(self):
        return len(self.data)
