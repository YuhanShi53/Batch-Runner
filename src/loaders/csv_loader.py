"""
CSV file data loader implementation.

Loads inference requests from a CSV file.
"""
import csv
import logging
from typing import Iterator
from pathlib import Path

from .base import DataLoader, LoadResult


logger = logging.getLogger(__name__)


class CSVDataLoader(DataLoader):
    """
    Load inference requests from a CSV file.

    Expected CSV format:
    id,prompt,category
    1,What is AI?,tech
    2,Explain quantum computing,science

    Configuration:
        file_path: Path to CSV file
        prompt_field: Column name containing the prompt (default: "prompt")
        id_field: Column name containing the ID (default: "id")
        encoding: File encoding (default: "utf-8")
    """

    def _initialize(self):
        """Initialize CSV file loader."""
        self.file_path = Path(self.config['file_path'])
        self.prompt_field = self.config.get('prompt_field', 'prompt')
        self.id_field = self.config.get('id_field', 'id')
        self.encoding = self.config.get('encoding', 'utf-8')

        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")

        # Read all rows
        with open(self.file_path, 'r', encoding=self.encoding, newline='') as f:
            reader = csv.DictReader(f)
            if self.prompt_field not in reader.fieldnames:
                raise ValueError(f"CSV must have '{self.prompt_field}' column")
            self.data = list(reader)

        self._len = len(self.data)

    def load(self) -> Iterator[LoadResult]:
        """Yield LoadResult objects from CSV data."""
        for idx, item in enumerate(self.data, 1):
            try:
                prompt = item.get(self.prompt_field, '')
                request_id = item.get(self.id_field, f"req_{id(item)}")

                if not prompt:
                    continue

                # Extract additional data
                additional_data = {
                    k: v for k, v in item.items()
                    if k not in [self.prompt_field, self.id_field] and v
                }

                yield LoadResult(
                    messages=[{"role": "user", "content": prompt}],
                    request_id=str(request_id),
                    additional_data=additional_data or None
                )
            except Exception as e:
                logger.error(
                    f"Unexpected error processing CSV row at index {idx}: {e}"
                )
                continue

    def __len__(self):
        return self._len
