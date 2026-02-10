# JSONL Customization Refactoring Summary

## Overview

This refactoring introduces a **Template Method pattern** with mixin classes to make JSONL loaders and savers customizable without requiring users to rewrite entire classes.

## Changes Made

### New Files Created

1. **[src/loaders/jsonl_mixin.py](src/loaders/jsonl_mixin.py)** - Mixin class for JSONL loaders
   - `JSONLLoaderMixin` with customizable parsing methods:
     - `parse_line()` - Parse raw JSONL lines
     - `should_skip_item()` - Filter items
     - `extract_request_id()` - Extract request IDs
     - `extract_prompt()` - Extract prompt text
     - `extract_additional_data()` - Extract metadata
     - `process_line_to_load_result()` - Template method orchestrating the above

2. **[src/savers/jsonl_mixin.py](src/savers/jsonl_mixin.py)** - Mixin class for JSONL savers
   - `JSONLSaverMixin` with customizable formatting methods:
     - `format_result()` - Format output dictionary
     - `serialize_output()` - Serialize to JSON string
     - `process_result_to_line()` - Template method orchestrating the above

3. **[docs/JSONL_CUSTOMIZATION.md](docs/JSONL_CUSTOMIZATION.md)** - Comprehensive customization guide
   - Detailed explanation of each overridable method
   - Multiple examples for common use cases
   - Best practices and configuration examples
   - Directory loader examples with streaming/non-streaming modes

4. **[examples/custom_jsonl_example.py](examples/custom_jsonl_example.py)** - Example implementations
   - 9 complete working examples
   - Covers loaders and savers
   - Single-file and directory-based variants

5. **[examples/custom_directory_jsonl_example.py](examples/custom_directory_jsonl_example.py)** - Directory-specific examples
   - 9 directory loader examples
   - Shows streaming vs non-streaming mode usage
   - Conversation format, nested structures, dynamic field mapping

6. **[examples/test_custom_directory_loader.py](examples/test_custom_directory_loader.py)** - Test suite
   - Demonstrates parse_line and extract_prompt work in both modes
   - All tests pass ✓

### Modified Files

#### Loaders

1. **[src/loaders/jsonl_loader.py](src/loaders/jsonl_loader.py)**
   - `JSONLDataLoader` now inherits from `JSONLLoaderMixin`
   - `MultimodalJSONLDataLoader` now inherits from `JSONLLoaderMixin`
   - Uses mixin methods for parsing and data extraction
   - Backward compatible - existing configs still work

2. **[src/loaders/directory_jsonl_loader.py](src/loaders/directory_jsonl_loader.py)**
   - `DirectoryJSONLDataLoader` now inherits from `JSONLLoaderMixin`
   - `MultimodalDirectoryJSONLDataLoader` now inherits from `JSONLLoaderMixin`
   - **Uses `parse_line()` in BOTH streaming and non-streaming modes**
   - Custom `extract_images()` method for multimodal support
   - Full customization support for directory-based processing

#### Savers

1. **[src/savers/jsonl_saver.py](src/savers/jsonl_saver.py)**
   - `JSONLResultSaver` now inherits from `JSONLSaverMixin`
   - Uses `process_result_to_line()` for formatting
   - Removed unused `json` and `datetime` imports

2. **[src/savers/directory_jsonl_saver.py](src/savers/directory_jsonl_saver.py)**
   - `DirectoryJSONLResultSaver` now inherits from `JSONLSaverMixin`
   - Uses `process_result_to_line()` for formatting
   - Removed unused `json` and `datetime` imports

## Key Features

### 1. Template Method Pattern

The core idea is that common logic is implemented once in the mixin classes, and users override specific methods to customize behavior. This reduces code duplication and makes customizations safer.

### 2. Backward Compatibility

All existing configurations continue to work without modification. The default implementations in the mixins match the original behavior.

### 3. Flexible Customization

Users can override any combination of methods:
- Override just one method (e.g., `extract_prompt`) for simple changes
- Override multiple methods for complex customizations
- Use `super()` to extend rather than replace behavior

### 4. Type Safety

All methods include proper type hints, making it easier to write correct customizations.

## Use Cases

This refactoring enables users to handle:

1. **Different JSONL formats**
   - Dict format: `{"prompt": "hello", "id": "1"}`
   - List format: `[{"text": "hello"}]`
   - Nested structures: `{"data": {"prompt": "hello"}}`

2. **Different field names**
   - Try multiple fields: `prompt`, `question`, `text`, `input`
   - Composite IDs: `category_docId`

3. **Filtering**
   - By language, quality score, or any custom criteria
   - Skip items without required fields

4. **Data transformation**
   - Flatten nested structures
   - Convert data types (e.g., tags list to comma-separated string)
   - Combine multiple fields

5. **Custom output formats**
   - Minimal output: just ID and response
   - Include token usage
   - Flatten and preserve metadata

## Migration Guide

### For Existing Users

No changes needed! Existing configurations continue to work:

```yaml
loader:
  class: src.loaders.jsonl_loader.JSONLDataLoader
  params:
    file_path: data/input.jsonl
```

### For New Customization

Instead of copying and modifying the entire loader class, create a new class that inherits from the existing one and override only what you need:

**Before (old way - not recommended):**
```python
# Had to copy entire class
class MyLoader(JSONLDataLoader):
    def _initialize(self):
        # ... copy all initialization code ...

    def load(self):
        # ... copy all load logic ...
        # Only change: different prompt extraction
```

**After (new way - recommended):**
```python
# Just override what you need
class MyLoader(JSONLDataLoader):
    def extract_prompt(self, item):
        # Custom prompt extraction
        return item.get('my_custom_field')
```

## Testing

All refactored classes have been verified to import correctly:
- ✅ `JSONLDataLoader`
- ✅ `MultimodalJSONLDataLoader`
- ✅ `DirectoryJSONLDataLoader`
- ✅ `MultimodalDirectoryJSONLDataLoader`
- ✅ `JSONLResultSaver`
- ✅ `DirectoryJSONLResultSaver`

## Documentation

- **[docs/JSONL_CUSTOMIZATION.md](docs/JSONL_CUSTOMIZATION.md)** - Full customization guide with examples
- **[examples/custom_jsonl_example.py](examples/custom_jsonl_example.py)** - 9 working examples
- **Existing CLAUDE.md** - Updated to reference the new mixin system

## Benefits

1. **Reduced Code Duplication** - Common logic implemented once
2. **Safer Customization** - Override small methods instead of rewriting classes
3. **Better Maintainability** - Changes to core logic automatically benefit custom classes
4. **Easier Testing** - Can test individual methods in isolation
5. **Clearer Intent** - Method names clearly indicate what they customize
6. **Extensibility** - New customization hooks can be added to mixins without breaking existing code
