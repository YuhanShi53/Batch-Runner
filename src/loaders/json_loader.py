"""
JSON file data loader implementation.

Loads inference requests from a JSON file.
"""
import json
from typing import Iterator, Dict, Any
from pathlib import Path

from .base import DataLoader, LoadResult


class JSONDataLoader(DataLoader):
    """
    Load inference requests from a JSON file.

    Expected JSON format:
    [
        {"id": "1", "prompt": "What is AI?", "category": "tech"},
        {"id": "2", "prompt": "Explain quantum computing"}
    ]

    Configuration:
        file_path: Path to JSON file
        batch_size: Number of items to load at once (default: 1)
        prompt_field: Field name containing the prompt (default: "prompt")
        id_field: Field name containing the ID (default: "id")
    """

    def _initialize(self):
        """Initialize JSON file loader."""
        self.file_path = Path(self.config['file_path'])
        self.prompt_field = self.config.get('prompt_field', 'prompt')
        self.id_field = self.config.get('id_field', 'id')

        if not self.file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {self.file_path}")

        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        if not isinstance(self.data, list):
            raise ValueError("JSON root must be a list of objects")

    def load(self) -> Iterator[LoadResult]:
        """Yield LoadResult objects from JSON data."""
        for item in self.data:
            prompt = item.get(self.prompt_field)
            request_id = item.get(self.id_field, f"req_{id(item)}")

            if prompt is None:
                continue

            # Extract additional data (everything except prompt and id)
            additional_data = {
                k: v for k, v in item.items()
                if k not in [self.prompt_field, self.id_field]
            }

            yield LoadResult(
                messages=[{"role": "user", "content": prompt}],
                request_id=str(request_id),
                additional_data=additional_data or None
            )

    def __len__(self):
        return len(self.data)
