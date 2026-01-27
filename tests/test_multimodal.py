"""
Tests for multimodal loader functionality.

Run with: python -m pytest tests/test_multimodal.py -v
"""
import json
import tempfile
from pathlib import Path
import pytest

from src.loaders.jsonl_loader import MultimodalJSONLDataLoader
from src.loaders.json_loader import MultimodalJSONDataLoader


class TestMultimodalJSONLDataLoader:
    """Test MultimodalJSONLDataLoader functionality."""

    def test_load_text_only(self):
        """Test loading text-only samples."""
        # Create temporary JSONL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"id": "1", "prompt": "What is AI?"}\n')
            f.write('{"id": "2", "prompt": "Explain ML"}\n')
            temp_path = f.name

        try:
            config = {
                'file_path': temp_path,
                'prompt_field': 'prompt',
                'id_field': 'id',
                'image_base_dir': '',
                'encode_images': False
            }

            loader = MultimodalJSONLDataLoader(config)
            results = list(loader)

            assert len(results) == 2
            assert results[0].request_id == "1"
            assert results[0].images is None
            assert len(results[0].messages) == 1
            assert results[0].messages[0]["role"] == "user"
            assert results[0].messages[0]["content"][0]["type"] == "text"
            assert results[0].messages[0]["content"][0]["text"] == "What is AI?"
        finally:
            Path(temp_path).unlink()

    def test_load_with_single_image(self):
        """Test loading samples with single image."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"id": "1", "prompt": "Describe this", "image": "photo.jpg"}\n')
            temp_path = f.name

        try:
            config = {
                'file_path': temp_path,
                'prompt_field': 'prompt',
                'id_field': 'id',
                'image_base_dir': '',
                'encode_images': False  # Don't try to encode for test
            }

            loader = MultimodalJSONLDataLoader(config)
            results = list(loader)

            assert len(results) == 1
            assert results[0].images == ["photo.jpg"]

            # Check message structure
            content = results[0].messages[0]["content"]
            assert len(content) == 2
            assert content[0]["type"] == "text"
            assert content[0]["text"] == "Describe this"
            assert content[1]["type"] == "image_url"
            assert content[1]["image_url"]["url"] == "photo.jpg"
        finally:
            Path(temp_path).unlink()

    def test_load_with_multiple_images(self):
        """Test loading samples with multiple images."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"id": "1", "prompt": "Compare these", "images": ["a.jpg", "b.png"]}\n')
            temp_path = f.name

        try:
            config = {
                'file_path': temp_path,
                'prompt_field': 'prompt',
                'id_field': 'id',
                'image_base_dir': '',
                'encode_images': False
            }

            loader = MultimodalJSONLDataLoader(config)
            results = list(loader)

            assert len(results) == 1
            assert results[0].images == ["a.jpg", "b.png"]

            # Check message structure
            content = results[0].messages[0]["content"]
            assert len(content) == 3
            assert content[0]["type"] == "text"
            assert content[1]["type"] == "image_url"
            assert content[2]["type"] == "image_url"
        finally:
            Path(temp_path).unlink()

    def test_image_field_precedence(self):
        """Test that 'image' field takes precedence over 'images'."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"id": "1", "prompt": "Test", "image": "single.jpg", "images": ["multi.jpg"]}\n')
            temp_path = f.name

        try:
            config = {
                'file_path': temp_path,
                'prompt_field': 'prompt',
                'id_field': 'id',
                'image_base_dir': '',
                'encode_images': False
            }

            loader = MultimodalJSONLDataLoader(config)
            results = list(loader)

            assert results[0].images == ["single.jpg"]
        finally:
            Path(temp_path).unlink()

    def test_additional_data_preserved(self):
        """Test that additional data fields are preserved."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"id": "1", "prompt": "Test", "category": "tech", "priority": 5}\n')
            temp_path = f.name

        try:
            config = {
                'file_path': temp_path,
                'prompt_field': 'prompt',
                'id_field': 'id',
                'image_base_dir': '',
                'encode_images': False
            }

            loader = MultimodalJSONLDataLoader(config)
            results = list(loader)

            assert results[0].additional_data is not None
            assert results[0].additional_data["category"] == "tech"
            assert results[0].additional_data["priority"] == 5
        finally:
            Path(temp_path).unlink()


class TestMultimodalJSONDataLoader:
    """Test MultimodalJSONDataLoader functionality."""

    def test_load_from_json_array(self):
        """Test loading from JSON array file."""
        # Create temporary JSON file
        data = [
            {"id": "1", "prompt": "What is this?", "image": "img1.jpg"},
            {"id": "2", "prompt": "Describe", "images": ["a.jpg", "b.jpg"]}
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name

        try:
            config = {
                'file_path': temp_path,
                'prompt_field': 'prompt',
                'id_field': 'id',
                'image_base_dir': '',
                'encode_images': False
            }

            loader = MultimodalJSONDataLoader(config)
            results = list(loader)

            assert len(results) == 2
            assert results[0].request_id == "1"
            assert results[0].images == ["img1.jpg"]
            assert results[1].request_id == "2"
            assert results[1].images == ["a.jpg", "b.jpg"]
        finally:
            Path(temp_path).unlink()


class TestMultimodalBase:
    """Test MultimodalDataLoader base functionality."""

    def test_create_multimodal_content_text_only(self):
        """Test creating content with text only."""
        from src.loaders.multimodal_base import MultimodalDataLoader

        class TestLoader(MultimodalDataLoader):
            def load(self):
                pass

        config = {'encode_images': False, 'image_base_dir': ''}
        loader = TestLoader(config)

        content = loader._create_multimodal_content("Hello world", None)

        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Hello world"

    def test_create_multimodal_content_with_images(self):
        """Test creating content with text and images."""
        from src.loaders.multimodal_base import MultimodalDataLoader

        class TestLoader(MultimodalDataLoader):
            def load(self):
                pass

        config = {'encode_images': False, 'image_base_dir': ''}
        loader = TestLoader(config)

        content = loader._create_multimodal_content("Describe this", ["img1.jpg", "img2.png"])

        assert len(content) == 3
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Describe this"
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"] == "img1.jpg"
        assert content[2]["type"] == "image_url"
        assert content[2]["image_url"]["url"] == "img2.png"

    def test_base64_image_already_encoded(self):
        """Test that already base64-encoded images are not re-encoded."""
        from src.loaders.multimodal_base import MultimodalDataLoader

        class TestLoader(MultimodalDataLoader):
            def load(self):
                pass

        config = {'encode_images': True, 'image_base_dir': ''}
        loader = TestLoader(config)

        already_encoded = "data:image/jpeg;base64,ABC123"
        processed = loader._process_images([already_encoded])

        assert len(processed) == 1
        assert processed[0] == already_encoded  # Should not change


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
