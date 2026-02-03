"""
Examples of custom loaders and savers using the refactored framework.

This file demonstrates how to:
1. Create custom loaders/savers outside the src/loaders and src/savers directories
2. Use the registration mechanism to register custom components
3. Inherit from streaming mixins for efficient processing
4. Override customization hooks for flexible data transformation
5. Support both text-only and multimodal inputs/outputs

Usage:
    # Import this file before running the CLI
    python -m src.cli --config configs/config.yaml

    # Or specify custom_modules in your config.yaml:
    # custom_modules:
    #   - examples/custom_components.py
"""

from typing import Iterator, Dict, Any, Optional, List
from pathlib import Path
import json
import logging

# Import base classes and mixins
from src.loaders.base import DataLoader, LoadResult
from src.savers.base import ResultSaver, SaveResult
from src.loaders.streaming_mixin import (
    StreamingLoaderMixin,
    MessagesBuilderMixin,
    PromptExtractorMixin,
    MultimodalInputMixin,
)
from src.savers.streaming_mixin import (
    StreamingSaverMixin,
    OutputFormatterMixin,
    MultimodalOutputMixin,
)
from src.loaders.jsonl_mixin import JSONLLoaderMixin
from src.savers.jsonl_mixin import JSONLSaverMixin
from src.utils.registry import register_loader, register_saver


logger = logging.getLogger(__name__)


# =================================================================
# Example 1: Simple custom loader with registration
# =================================================================

@register_loader
class MyCustomLoader(DataLoader):
    """
    A simple custom loader that loads prompts from a list.

    This demonstrates basic customization and the registration decorator.
    """

    def _initialize(self):
        """Initialize with prompts from config or a default list."""
        self.prompts = self.config.get(
            'prompts',
            [
                "What is AI?",
                "Explain machine learning",
                "Tell me a joke"
            ]
        )

    def load(self) -> Iterator[LoadResult]:
        """Yield LoadResult objects for each prompt."""
        for idx, prompt in enumerate(self.prompts, 1):
            yield LoadResult(
                messages=[{"role": "user", "content": prompt}],
                request_id=f"custom_{idx}"
            )


# =================================================================
# Example 2: Custom JSONL loader with streaming and prompt extraction
# =================================================================

@register_loader
class CustomJSONLLoader(JSONLLoaderMixin, DataLoader):
    """
    Custom JSONL loader with multi-field prompt extraction.

    Demonstrates overriding the extract_prompt method to try multiple fields.
    """

    def _initialize(self):
        """Initialize JSONL loader."""
        self.file_path = Path(self.config['file_path'])
        self.prompt_fields = self.config.get(
            'prompt_fields',
            ['prompt', 'question', 'text', 'input', 'query']
        )
        self.id_field = self.config.get('id_field', 'id')
        self.streaming = self.config.get('streaming', True)

        if not self.file_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {self.file_path}")

    def extract_prompt(self, item: Dict[str, Any]) -> Optional[str]:
        """
        Try multiple fields to extract the prompt.

        This overrides the default extract_prompt from JSONLLoaderMixin.
        """
        for field in self.prompt_fields:
            if field in item and item[field]:
                return str(item[field])
        return None

    def load(self) -> Iterator[LoadResult]:
        """Load data with streaming support."""
        if self.streaming:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    result = self.process_line_to_load_result(
                        line=line,
                        line_num=line_num,
                        source=str(self.file_path),
                        default_id=f"req_{line_num}"
                    )
                    if result is not None:
                        yield result
        else:
            # Batch mode (for comparison)
            data = []
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))

            for idx, item in enumerate(data, 1):
                prompt = self.extract_prompt(item)
                if prompt is None:
                    continue

                request_id = str(item.get(self.id_field, f"req_{idx}"))
                additional_data = self.extract_additional_data(item)

                yield LoadResult(
                    messages=[{"role": "user", "content": prompt}],
                    request_id=request_id,
                    additional_data=additional_data or None
                )


# =================================================================
# Example 3: Custom loader with message building
# =================================================================

@register_loader
class ChatMessagesLoader(MessagesBuilderMixin, DataLoader):
    """
    Loader that constructs chat messages from conversation history.

    Demonstrates using MessagesBuilderMixin to build complex messages.
    """

    def _initialize(self):
        """Initialize with conversation data."""
        self.file_path = Path(self.config['file_path'])

        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.conversations = json.load(f)

    def build_messages(
        self,
        prompt: str,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Build chat messages from conversation history.

        This overrides the default build_messages to include history.
        """
        messages = []

        # Add system prompt if configured
        system_prompt = self.config.get('system_prompt')
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add conversation history if available
        if additional_data and 'history' in additional_data:
            messages.extend(additional_data['history'])

        # Add current prompt
        messages.append({"role": "user", "content": prompt})

        return messages

    def load(self) -> Iterator[LoadResult]:
        """Yield LoadResult objects for each conversation."""
        for idx, conv in enumerate(self.conversations, 1):
            prompt = conv.get('prompt', conv.get('input', ''))
            additional_data = {
                'history': conv.get('history', []),
                'metadata': {k: v for k, v in conv.items()
                            if k not in ['prompt', 'input', 'history']}
            }

            messages = self.build_messages(prompt, additional_data)

            yield LoadResult(
                messages=messages,
                request_id=conv.get('id', f"conv_{idx}"),
                additional_data=additional_data
            )


# =================================================================
# Example 4: Custom saver with output formatting
# =================================================================

@register_saver
class FormattedJSONLSaver(JSONLSaverMixin, ResultSaver):
    """
    Custom JSONL saver with simplified output format.

    Demonstrates overriding format_result for custom output.
    """

    def _initialize(self):
        """Initialize JSONL saver."""
        self.output_path = Path(self.config['output_path'])
        self.append = self.config.get('append', True)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        mode = 'a' if self.append else 'w'
        self.file = open(self.output_path, mode, encoding='utf-8')
        self._lock = __import__('threading').Lock()

    def format_result(self, result: SaveResult) -> Dict[str, Any]:
        """
        Format result with a simplified structure.

        This overrides the default format_result from JSONLSaverMixin.
        """
        # Extract the generated content
        try:
            content = result.model_output['choices'][0]['message']['content']
        except (KeyError, IndexError, TypeError):
            content = None

        # Extract usage info
        try:
            usage = result.model_output.get('usage', {})
            tokens = usage.get('total_tokens', 0)
        except (AttributeError, TypeError):
            tokens = 0

        return {
            'id': result.request_id,
            'response': content,
            'tokens': tokens,
            'success': result.error is None
        }

    def save(self, result: SaveResult):
        """Save result using the mixin's template method."""
        line = self.process_result_to_line(result)

        with self._lock:
            self.file.write(line + '\n')
            self.file.flush()

    def cleanup(self):
        """Close the file."""
        with self._lock:
            if hasattr(self, 'file') and not self.file.closed:
                self.file.close()


# =================================================================
# Example 5: Custom streaming directory loader
# =================================================================

@register_loader
class StreamingDirectoryLoader(StreamingLoaderMixin, DataLoader):
    """
    Streaming loader for processing multiple files in a directory.

    Demonstrates using StreamingLoaderMixin for efficient processing.
    """

    def _initialize(self):
        """Initialize directory loader."""
        self._initialize_streaming()
        self.input_dir = Path(self.config['input_dir'])
        self.file_pattern = self.config.get('file_pattern', '*.txt')

    def _discover_sources(self) -> List[Path]:
        """Discover all files matching the pattern."""
        files = sorted(self.input_dir.rglob(self.file_pattern))
        logger.info(f"Discovered {len(files)} files")
        return files

    def _process_source(self, source: Path) -> Iterator[LoadResult]:
        """Process a single file."""
        with open(source, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                yield LoadResult(
                    messages=[{"role": "user", "content": line}],
                    request_id=f"{source.name}:{line_num}",
                    additional_data={'source_file': str(source)}
                )


# =================================================================
# Example 6: Multimodal custom loader
# =================================================================

@register_loader
class MultimodalCustomLoader(MultimodalInputMixin, DataLoader):
    """
    Custom loader for multimodal data (text + images).

    Demonstrates using MultimodalInputMixin for image handling.
    """

    def _initialize(self):
        """Initialize multimodal loader."""
        self.file_path = Path(self.config['file_path'])
        self.image_base_dir = Path(self.config.get('image_base_dir', ''))

        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def extract_images(self, item: Dict[str, Any]) -> Optional[List[str]]:
        """
        Extract images with support for nested structures.

        This overrides the default extract_images from MultimodalInputMixin.
        """
        # Check for direct image field
        if 'image' in item:
            return [item['image']] if isinstance(item['image'], str) else item['image']

        # Check for images field
        if 'images' in item:
            return item['images'] if isinstance(item['images'], list) else [item['images']]

        # Check for nested media structure
        if 'media' in item:
            return [m['path'] for m in item['media'] if m.get('type') == 'image']

        return None

    def load(self) -> Iterator[LoadResult]:
        """Yield multimodal LoadResult objects."""
        for idx, item in enumerate(self.data, 1):
            prompt = item.get('prompt', item.get('text', ''))
            images = self.extract_images(item)

            if images:
                # Multimodal content
                content = [{"type": "text", "text": prompt}]
                for img in images:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": img}
                    })
                messages = [{"role": "user", "content": content}]
            else:
                # Text-only
                messages = [{"role": "user", "content": prompt}]

            yield LoadResult(
                messages=messages,
                request_id=item.get('id', f"multimodal_{idx}"),
                additional_data={'images': images} if images else None
            )


# =================================================================
# Example configuration YAML
# =================================================================

"""
Example config.yaml using custom components:

```yaml
# Option 1: Use custom_modules to auto-register
custom_modules:
  - examples/custom_components.py

loader:
  class: CustomJSONLLoader
  params:
    file_path: data/input.jsonl
    prompt_fields:
      - prompt
      - question
      - text
    streaming: true

saver:
  class: FormattedJSONLSaver
  params:
    output_path: results/output.jsonl
    append: true

runner:
  max_concurrency: 10
  model_name: "meta-llama/Llama-3-8b"
  servers_dir: ./servers
```

# Option 2: Import and register manually in a script
```python
from examples import custom_components
from src.utils.config import load_config, get_loader_class, get_saver_class

config = load_config('configs/config.yaml')
loader_class = get_loader_class('CustomJSONLLoader')
saver_class = get_saver_class('FormattedJSONLSaver')
```
"""
