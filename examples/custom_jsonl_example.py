"""
Example: Custom JSONL Loader and Saver

This example demonstrates how to create custom JSONL loaders and savers
by overriding methods from the mixin classes.
"""
import json
from typing import Dict, Any, Optional

from src.loaders.jsonl_loader import JSONLDataLoader
from src.savers.jsonl_saver import JSONLResultSaver
from src.loaders.directory_jsonl_loader import DirectoryJSONLDataLoader


# Example 1: Handle list-format JSONL
class ListFormatLoader(JSONLDataLoader):
    """
    Load from JSONL where each line is a list:
    [{"prompt": "What is AI?", "id": "1"}]
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
                "prompt": " ".join(item.get("text", item.get("prompt", "")) for item in obj)
            }

        return obj


# Example 2: Multi-field prompt loader
class MultiFieldPromptLoader(JSONLDataLoader):
    """
    Try multiple fields for prompt extraction.
    Useful for datasets with inconsistent naming.
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


# Example 3: Filter by quality and language
class FilteredLoader(JSONLDataLoader):
    """
    Only load high-quality English items.
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


# Example 4: Composite ID generator
class CompositeIDLoader(JSONLDataLoader):
    """
    Generate composite IDs from multiple fields.
    """

    def extract_request_id(self, item: Dict[str, Any], default_id: str) -> str:
        category = item.get("category", "unknown")
        doc_id = item.get("id", default_id)
        return f"{category}_{doc_id}"


# Example 5: Minimal output saver
class MinimalSaver(JSONLResultSaver):
    """
    Save only the essential information.
    """

    def format_result(self, result) -> Dict[str, Any]:
        # Extract generated text
        content = ""
        if 'choices' in result.model_output and len(result.model_output['choices']) > 0:
            content = result.model_output['choices'][0]['message']['content']

        return {
            "id": result.request_id,
            "response": content
        }


# Example 6: Token-aware saver
class TokenAwareSaver(JSONLResultSaver):
    """
    Include token usage in output.
    """

    def format_result(self, result) -> Dict[str, Any]:
        content = ""
        if 'choices' in result.model_output and len(result.model_output['choices']) > 0:
            content = result.model_output['choices'][0]['message']['content']

        usage = result.model_output.get('usage', {})

        return {
            "id": result.request_id,
            "response": content,
            "prompt_tokens": usage.get('prompt_tokens', 0),
            "completion_tokens": usage.get('completion_tokens', 0),
            "total_tokens": usage.get('total_tokens', 0)
        }


# Example 7: Metadata-preserving saver
class MetadataSaver(JSONLResultSaver):
    """
    Flatten output and preserve metadata.
    """

    def format_result(self, result) -> Dict[str, Any]:
        output = {}

        # Extract content
        if 'choices' in result.model_output and len(result.model_output['choices']) > 0:
            output['response'] = result.model_output['choices'][0]['message']['content']

        # Flatten additional_data
        if result.additional_data:
            # Selectively include metadata
            for key in ['category', 'tags', 'language', '_source_file']:
                if key in result.additional_data:
                    output[key] = result.additional_data[key]

        # Add error if present
        if result.error:
            output['error'] = result.error

        return output


# Example 8: Directory loader with source tracking
class SourceAwareDirectoryLoader(DirectoryJSONLDataLoader):
    """
    Track source file information in request_id and additional_data.
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


# Example 9: Combined customization
class CompleteCustomLoader(JSONLDataLoader):
    """
    A fully customized loader combining multiple features:
    1. Handles list-format lines
    2. Filters by quality
    3. Uses composite IDs
    4. Tries multiple prompt fields
    5. Preserves selective metadata
    """

    def parse_line(self, line: str, line_num: int, source: str) -> Optional[Dict[str, Any]]:
        obj = json.loads(line)

        # Handle list format
        if isinstance(obj, list):
            if len(obj) == 1:
                return obj[0]
            return {"items": obj, "id": str(line_num)}

        return obj

    def should_skip_item(self, item: Dict[str, Any]) -> bool:
        # Skip low-quality items
        quality = item.get("quality_score", 1.0)
        return quality < 0.5

    def extract_request_id(self, item: Dict[str, Any], default_id: str) -> str:
        # Use category-based ID
        category = item.get("category", "misc")
        item_id = item.get("id", default_id)
        return f"{category}_{item_id}"

    def extract_prompt(self, item: Dict[str, Any]) -> Optional[str]:
        # Try multiple fields
        for field in ['prompt', 'question', 'text', 'input']:
            if field in item:
                return str(item[field])
        return None

    def extract_additional_data(self, item: Dict[str, Any]) -> Dict[str, Any]:
        # Only preserve metadata
        metadata_fields = ['category', 'tags', 'language', '_source_file']
        return {
            k: v for k, v in item.items()
            if k in metadata_fields and v is not None
        }


# Usage example
if __name__ == "__main__":
    # Example configuration for using custom loader
    example_config = {
        "loader": {
            "class": "custom_jsonl_example.MultiFieldPromptLoader",
            "params": {
                "file_path": "data/input.jsonl",
                "prompt_field": "prompt",  # Will try multiple fields anyway
                "id_field": "id"
            }
        },
        "saver": {
            "class": "custom_jsonl_example.TokenAwareSaver",
            "params": {
                "output_path": "output/results.jsonl"
            }
        }
    }

    print("Custom JSONL Loader and Saver Examples")
    print("=" * 50)
    print("\nAvailable custom loaders:")
    print("  1. ListFormatLoader - Handle list-format JSONL")
    print("  2. MultiFieldPromptLoader - Try multiple prompt fields")
    print("  3. FilteredLoader - Filter by quality and language")
    print("  4. CompositeIDLoader - Generate composite IDs")
    print("  5. SourceAwareDirectoryLoader - Track source file info")
    print("  6. CompleteCustomLoader - Combined customizations")
    print("\nAvailable custom savers:")
    print("  1. MinimalSaver - Save only ID and response")
    print("  2. TokenAwareSaver - Include token usage")
    print("  3. MetadataSaver - Preserve metadata")
    print("\nSee JSONL_CUSTOMIZATION.md for detailed documentation.")
