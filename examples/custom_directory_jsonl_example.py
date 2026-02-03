"""
Example: Custom Directory JSONL Loader with Stream Processing

This example demonstrates how to create custom directory JSONL loaders
that support both streaming and batch modes with customizable line processing.
"""
import json
from typing import Dict, Any, Optional

from src.loaders.directory_jsonl_loader import DirectoryJSONLDataLoader
from src.loaders.multimodal_base import MultimodalDataLoader


# Example 1: Handle list-format JSONL in directory loader
class ListFormatDirectoryLoader(DirectoryJSONLDataLoader):
    """
    Load from directory where each JSONL line is a list:
    [{"prompt": "What is AI?", "id": "1"}]

    Works in both streaming and non-streaming modes.
    """

    def parse_line(self, line: str, line_num: int, source: str) -> Optional[Dict[str, Any]]:
        obj = json.loads(line)

        # Convert list to dict
        if isinstance(obj, list):
            if len(obj) == 1 and isinstance(obj[0], dict):
                # Single item list - unwrap it
                return obj[0]
            # Multiple items - combine prompts
            return {
                "items": obj,
                "id": f"{source}:{line_num}",
                "prompt": " ".join(
                    item.get("text", item.get("prompt", "")) for item in obj
                )
            }

        return obj


# Example 2: Multi-field prompt directory loader
class MultiFieldDirectoryLoader(DirectoryJSONLDataLoader):
    """
    Try multiple fields for prompt extraction in directory files.
    Useful for datasets with inconsistent naming.

    Works in both streaming and non-streaming modes.
    """

    def extract_prompt(self, item: Dict[str, Any]) -> Optional[str]:
        # Try fields in order of preference
        for field in ['prompt', 'question', 'text', 'input', 'instruction']:
            if field in item:
                prompt = item[field]
                # Handle cases where prompt might be a list
                if isinstance(prompt, list):
                    return " ".join(str(p) for p in prompt)
                return str(prompt)
        return None


# Example 3: Filter by quality and language (directory version)
class FilteredDirectoryLoader(DirectoryJSONLDataLoader):
    """
    Only load high-quality English items from directory.

    Works in both streaming and non-streaming modes.
    """

    def should_skip_item(self, item: Dict[str, Any]) -> bool:
        # Skip non-English items
        if item.get("language") != "en":
            return True

        # Skip low-quality items
        quality = item.get("quality_score", 1.0)
        if quality < 0.8:
            return True

        return False


# Example 4: Composite ID generator with source tracking
class SourceAwareDirectoryLoader(DirectoryJSONLDataLoader):
    """
    Track source file information in request_id and additional_data.

    Works in both streaming and non-streaming modes.
    """

    def extract_request_id(self, item: Dict[str, Any], default_id: str) -> str:
        # Include source file in ID
        source_file = item.get('_source_file', 'unknown')
        item_id = item.get('id', default_id)
        # Extract just the filename from the path
        filename = source_file.split('/')[-1] if '/' in source_file else source_file
        return f"{filename}:{item_id}"

    def extract_additional_data(self, item: Dict[str, Any]) -> Dict[str, Any]:
        # Include source information
        data = super().extract_additional_data(item)

        # Explicitly preserve source info
        if '_source_file' in item:
            data['_source_file'] = item['_source_file']
        if '_source_dir' in item:
            data['_source_dir'] = item['_source_dir']

        return data


# Example 5: Handle nested structures in directory files
class NestedDirectoryLoader(DirectoryJSONLDataLoader):
    """
    Handle nested structures like {"data": {"prompt": "hello", "id": "1"}}

    Works in both streaming and non-streaming modes.
    """

    def parse_line(self, line: str, line_num: int, source: str) -> Optional[Dict[str, Any]]:
        obj = json.loads(line)

        # Flatten nested structure
        if "data" in obj and isinstance(obj["data"], dict):
            return obj["data"]

        # Handle other nested patterns
        if "content" in obj and isinstance(obj["content"], dict):
            return obj["content"]

        return obj


# Example 6: Combined customizations for directory loader
class CompleteCustomDirectoryLoader(DirectoryJSONLDataLoader):
    """
    A fully customized directory loader combining multiple features:
    1. Handles list-format lines
    2. Filters by quality
    3. Uses composite IDs with source tracking
    4. Tries multiple prompt fields
    5. Preserves selective metadata

    Works in both streaming and non-streaming modes.
    """

    def parse_line(self, line: str, line_num: int, source: str) -> Optional[Dict[str, Any]]:
        obj = json.loads(line)

        # Handle list format
        if isinstance(obj, list):
            if len(obj) == 1:
                return obj[0]
            return {"items": obj, "id": str(line_num)}

        # Handle nested format
        if "data" in obj and isinstance(obj["data"], dict):
            return obj["data"]

        return obj

    def should_skip_item(self, item: Dict[str, Any]) -> bool:
        # Skip low-quality items
        quality = item.get("quality_score", 1.0)
        return quality < 0.5

    def extract_request_id(self, item: Dict[str, Any], default_id: str) -> str:
        # Use category-based ID with source
        source_file = item.get('_source_file', 'unknown')
        filename = source_file.split('/')[-1] if '/' in source_file else source_file
        category = item.get("category", "misc")
        item_id = item.get("id", default_id)
        return f"{filename}_{category}_{item_id}"

    def extract_prompt(self, item: Dict[str, Any]) -> Optional[str]:
        # Try multiple fields
        for field in ['prompt', 'question', 'text', 'input']:
            if field in item:
                return str(item[field])
        return None

    def extract_additional_data(self, item: Dict[str, Any]) -> Dict[str, Any]:
        # Only preserve metadata
        metadata_fields = [
            'category', 'tags', 'language',
            '_source_file', '_source_dir'
        ]
        return {
            k: v for k, v in item.items()
            if k in metadata_fields and v is not None
        }


# Example 7: Multimodal directory loader with custom media structure
class CustomMediaDirectoryLoader:
    """
    Custom multimodal directory loader supporting nested media structure.

    Example format:
    {
        "id": "1",
        "prompt": "Describe this image",
        "media": [
            {"type": "image", "path": "img1.jpg"},
            {"type": "image", "path": "img2.png"}
        ]
    }

    Works in both streaming and non-streaming modes.
    """

    # This would inherit from MultimodalDirectoryJSONLDataLoader
    # and override extract_images method

    # def extract_images(self, item: Dict[str, Any]) -> Optional[List[str]]:
    #     # Try custom media structure first
    #     if 'media' in item and isinstance(item['media'], list):
    #         images = [
    #             m['path'] for m in item['media']
    #             if m.get('type') == 'image' and 'path' in m
    #         ]
    #         return images if images else None
    #
    #     # Fall back to default behavior
    #     return super().extract_images(item)


# Example 8: Handle conversation format in directory files
class ConversationDirectoryLoader(DirectoryJSONLDataLoader):
    """
    Handle conversation format where each line has a messages array.

    Example format:
    {
        "id": "conv1",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
    }

    Extracts the last user message as the prompt.
    """

    def parse_line(self, line: str, line_num: int, source: str) -> Optional[Dict[str, Any]]:
        obj = json.loads(line)

        # Handle conversation format
        if "messages" in obj and isinstance(obj["messages"], list):
            # Find the last user message
            last_user_msg = None
            for msg in reversed(obj["messages"]):
                if msg.get("role") == "user":
                    last_user_msg = msg.get("content")
                    break

            if last_user_msg:
                return {
                    "id": obj.get("id", f"{source}:{line_num}"),
                    "prompt": last_user_msg,
                    "conversation": obj["messages"]  # Preserve full conversation
                }

        return obj


# Example 9: Dynamic field mapping directory loader
class DynamicFieldMappingLoader(DirectoryJSONLDataLoader):
    """
    Map fields dynamically based on item content.

    This example shows how to handle datasets where field names
    vary based on item type or category.
    """

    def extract_prompt(self, item: Dict[str, Any]) -> Optional[str]:
        # Check item type and use appropriate field
        item_type = item.get("type", "default")

        if item_type == "qa":
            # Q&A format
            return item.get("question")
        elif item_type == "instruction":
            # Instruction format
            return item.get("instruction")
        else:
            # Default format
            for field in ['prompt', 'text', 'input']:
                if field in item:
                    return str(item[field])

        return None

    def extract_request_id(self, item: Dict[str, Any], default_id: str) -> str:
        # Use different ID fields based on item type
        item_type = item.get("type", "default")

        if item_type == "qa":
            return item.get("qa_id", default_id)
        elif item_type == "instruction":
            return item.get("task_id", default_id)
        else:
            return item.get("id", default_id)


# Usage example
if __name__ == "__main__":
    # Example configuration for using custom directory loader
    streaming_config = {
        "loader": {
            "class": "custom_directory_jsonl_example.MultiFieldDirectoryLoader",
            "params": {
                "input_dir": "data/conversations",
                "file_pattern": "*.jsonl",
                "prompt_field": "prompt",  # Will try multiple fields anyway
                "id_field": "id",
                "streaming": True  # Enable streaming mode
            }
        }
    }

    batch_config = {
        "loader": {
            "class": "custom_directory_jsonl_example.CompleteCustomDirectoryLoader",
            "params": {
                "input_dir": "data/dataset",
                "file_pattern": "conv.jsonl",
                "prompt_field": "prompt",
                "id_field": "id",
                "streaming": False  # Non-streaming mode (load all into memory)
            }
        }
    }

    print("Custom Directory JSONL Loader Examples")
    print("=" * 60)
    print("\nAvailable custom directory loaders:")
    print("  1. ListFormatDirectoryLoader - Handle list-format JSONL")
    print("  2. MultiFieldDirectoryLoader - Try multiple prompt fields")
    print("  3. FilteredDirectoryLoader - Filter by quality and language")
    print("  4. SourceAwareDirectoryLoader - Track source file info")
    print("  5. NestedDirectoryLoader - Handle nested structures")
    print("  6. CompleteCustomDirectoryLoader - Combined customizations")
    print("  7. CustomMediaDirectoryLoader - Custom media structure")
    print("  8. ConversationDirectoryLoader - Handle conversation format")
    print("  9. DynamicFieldMappingLoader - Dynamic field mapping")
    print("\nAll loaders support both streaming and non-streaming modes!")
    print("\nStreaming mode (streaming: true):")
    print("  - Minimal memory usage")
    print("  - Processes files on-demand")
    print("  - Best for large datasets")
    print("\nNon-streaming mode (streaming: false):")
    print("  - Loads all data into memory")
    print("  - Easier debugging")
    print("  - Requires sufficient memory")
    print("\nSee JSONL_CUSTOMIZATION.md for detailed documentation.")
