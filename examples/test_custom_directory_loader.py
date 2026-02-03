"""
Test: Custom Directory JSONL Loader with Stream Processing

This test demonstrates that custom parse_line and extract_prompt
work in both streaming and non-streaming modes.
"""
import json
import tempfile
import shutil
from pathlib import Path

# Create test data
def create_test_data(temp_dir):
    """Create test directory with JSONL files"""
    data_dir = temp_dir / "data"
    data_dir.mkdir()

    # Create test file with mixed format data
    test_file = data_dir / "test.jsonl"
    with open(test_file, 'w') as f:
        # List format lines (will be converted by custom parse_line)
        f.write('[{"text": "What is AI?", "id": "1"}]\n')
        f.write('[{"text": "Explain quantum computing", "id": "2"}]\n')
        # Dict format lines
        f.write('{"prompt": "Tell me a joke", "id": "3"}\n')
        f.write('{"question": "What is the capital of France?", "id": "4"}\n')

    return str(data_dir)


def test_list_format_loader():
    """Test list-format loader in both streaming and non-streaming modes"""
    from src.loaders.directory_jsonl_loader import DirectoryJSONLDataLoader
    import json
    from typing import Dict, Any, Optional

    class ListFormatLoader(DirectoryJSONLDataLoader):
        """Handle list-format JSONL"""

        def parse_line(self, line: str, line_num: int, source: str) -> Optional[Dict[str, Any]]:
            obj = json.loads(line)
            if isinstance(obj, list):
                if len(obj) == 1 and isinstance(obj[0], dict):
                    return obj[0]
                return {
                    "items": obj,
                    "id": f"{source}:{line_num}",
                    "prompt": " ".join(item.get("text", item.get("prompt", "")) for item in obj)
                }
            return obj

    # Create temp directory
    temp_dir = Path(tempfile.mkdtemp())
    try:
        data_dir = create_test_data(temp_dir)

        # Test streaming mode
        print("Testing ListFormatLoader in STREAMING mode...")
        config_streaming = {
            'input_dir': data_dir,
            'file_pattern': '*.jsonl',
            'streaming': True
        }
        loader_streaming = ListFormatLoader(config_streaming)

        items_streaming = list(loader_streaming.load())
        print(f"  Loaded {len(items_streaming)} items in streaming mode")
        for item in items_streaming:
            print(f"    - {item.request_id}: {item.messages[0]['content'][:30]}...")

        # Test non-streaming mode
        print("\nTesting ListFormatLoader in NON-STREAMING mode...")
        config_batch = {
            'input_dir': data_dir,
            'file_pattern': '*.jsonl',
            'streaming': False
        }
        loader_batch = ListFormatLoader(config_batch)

        items_batch = list(loader_batch.load())
        print(f"  Loaded {len(items_batch)} items in non-streaming mode")
        for item in items_batch:
            print(f"    - {item.request_id}: {item.messages[0]['content'][:30]}...")

        assert len(items_streaming) == len(items_batch), "Mode mismatch!"
        print("\n✓ ListFormatLoader works in both modes!")

    finally:
        shutil.rmtree(temp_dir)


def test_multi_field_loader():
    """Test multi-field prompt loader in both streaming and non-streaming modes"""
    from src.loaders.directory_jsonl_loader import DirectoryJSONLDataLoader
    from typing import Dict, Any, Optional

    class MultiFieldLoader(DirectoryJSONLDataLoader):
        """Try multiple fields for prompt extraction"""

        def parse_line(self, line: str, line_num: int, source: str) -> Optional[Dict[str, Any]]:
            obj = json.loads(line)
            # Handle list format
            if isinstance(obj, list):
                if len(obj) == 1 and isinstance(obj[0], dict):
                    return obj[0]
                return {
                    "items": obj,
                    "id": f"{source}:{line_num}",
                    "prompt": " ".join(item.get("text", item.get("prompt", "")) for item in obj)
                }
            return obj

        def extract_prompt(self, item: Dict[str, Any]) -> Optional[str]:
            for field in ['prompt', 'question', 'text', 'input']:
                if field in item:
                    return str(item[field])
            return None

    # Create temp directory
    temp_dir = Path(tempfile.mkdtemp())
    try:
        data_dir = create_test_data(temp_dir)

        # Test streaming mode
        print("\nTesting MultiFieldLoader in STREAMING mode...")
        config_streaming = {
            'input_dir': data_dir,
            'file_pattern': '*.jsonl',
            'streaming': True
        }
        loader_streaming = MultiFieldLoader(config_streaming)

        items_streaming = list(loader_streaming.load())
        print(f"  Loaded {len(items_streaming)} items in streaming mode")
        for item in items_streaming:
            print(f"    - {item.request_id}: {item.messages[0]['content'][:30]}...")

        # Test non-streaming mode
        print("\nTesting MultiFieldLoader in NON-STREAMING mode...")
        config_batch = {
            'input_dir': data_dir,
            'file_pattern': '*.jsonl',
            'streaming': False
        }
        loader_batch = MultiFieldLoader(config_batch)

        items_batch = list(loader_batch.load())
        print(f"  Loaded {len(items_batch)} items in non-streaming mode")
        for item in items_batch:
            print(f"    - {item.request_id}: {item.messages[0]['content'][:30]}...")

        assert len(items_streaming) == len(items_batch), "Mode mismatch!"
        print("\n✓ MultiFieldLoader works in both modes!")

    finally:
        shutil.rmtree(temp_dir)


def test_combined_loader():
    """Test combined customizations in both streaming and non-streaming modes"""
    from src.loaders.directory_jsonl_loader import DirectoryJSONLDataLoader
    import json
    from typing import Dict, Any, Optional

    class CombinedLoader(DirectoryJSONLDataLoader):
        """Combined customizations"""

        def parse_line(self, line: str, line_num: int, source: str) -> Optional[Dict[str, Any]]:
            obj = json.loads(line)
            if isinstance(obj, list):
                if len(obj) == 1 and isinstance(obj[0], dict):
                    return obj[0]
                return {
                    "items": obj,
                    "id": f"{source}:{line_num}",
                    "prompt": " ".join(item.get("text", item.get("prompt", "")) for item in obj)
                }
            return obj

        def extract_prompt(self, item: Dict[str, Any]) -> Optional[str]:
            for field in ['prompt', 'question', 'text']:
                if field in item:
                    return str(item[field])
            return None

        def extract_request_id(self, item: Dict[str, Any], default_id: str) -> str:
            source_file = item.get('_source_file', 'unknown')
            filename = source_file.split('/')[-1] if '/' in source_file else source_file
            item_id = item.get('id', default_id)
            return f"{filename}:{item_id}"

    # Create temp directory
    temp_dir = Path(tempfile.mkdtemp())
    try:
        data_dir = create_test_data(temp_dir)

        # Test streaming mode
        print("\nTesting CombinedLoader in STREAMING mode...")
        config_streaming = {
            'input_dir': data_dir,
            'file_pattern': '*.jsonl',
            'streaming': True
        }
        loader_streaming = CombinedLoader(config_streaming)

        items_streaming = list(loader_streaming.load())
        print(f"  Loaded {len(items_streaming)} items in streaming mode")
        for item in items_streaming:
            source = item.additional_data.get('_source_file', 'unknown')
            print(f"    - {item.request_id} (from {source}): {item.messages[0]['content'][:30]}...")

        # Test non-streaming mode
        print("\nTesting CombinedLoader in NON-STREAMING mode...")
        config_batch = {
            'input_dir': data_dir,
            'file_pattern': '*.jsonl',
            'streaming': False
        }
        loader_batch = CombinedLoader(config_batch)

        items_batch = list(loader_batch.load())
        print(f"  Loaded {len(items_batch)} items in non-streaming mode")
        for item in items_batch:
            source = item.additional_data.get('_source_file', 'unknown')
            print(f"    - {item.request_id} (from {source}): {item.messages[0]['content'][:30]}...")

        assert len(items_streaming) == len(items_batch), "Mode mismatch!"
        print("\n✓ CombinedLoader works in both modes!")

    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Custom Directory JSONL Loaders")
    print("=" * 60)

    test_list_format_loader()
    test_multi_field_loader()
    test_combined_loader()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    print("\nCustom parse_line and extract_prompt work in both:")
    print("  - STREAMING mode (streaming: true)")
    print("  - NON-STREAMING mode (streaming: false)")
