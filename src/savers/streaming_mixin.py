"""
Streaming result saver mixin for efficient on-demand result saving.

This mixin provides a template method pattern for implementing streaming
result savers that write results immediately rather than buffering
everything in memory.
"""
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
import threading

from .base import SaveResult


logger = logging.getLogger(__name__)


class StreamingSaverMixin:
    """
    Mixin class that provides streaming result saving capabilities.

    This mixin implements the Template Method pattern for streaming result
    writing, allowing subclasses to customize how results are written
    without rewriting the entire streaming logic.

    Key benefits:
    - Immediate write-back (results persisted as they arrive)
    - Constant memory usage regardless of result count
    - Better fault tolerance (data saved even if process crashes)

    Usage:
        class MyStreamingSaver(StreamingSaverMixin, ResultSaver):
            def _get_output_path(self, result):
                # Determine output location based on result
                return Path("output") / f"{result.request_id}.json"

            def _format_result(self, result):
                # Format result for output
                return {"id": result.request_id, "data": result.model_output}

            def _write_result(self, path, formatted_data):
                # Write formatted data to path
                with open(path, 'w') as f:
                    json.dump(formatted_data, f)

    Configuration:
        streaming: Enable streaming mode (default: True)
        immediate_flush: Flush to disk after each write (default: True)
    """

    def _initialize_streaming(self):
        """
        Initialize streaming-specific configuration.

        Subclasses should call this in their _initialize() method.
        """
        self.streaming = self.config.get('streaming', True)
        self.immediate_flush = self.config.get('immediate_flush', True)
        self._lock = threading.Lock()

        if not self.streaming:
            logger.info(f"{self.__class__.__name__}: Non-streaming mode (batch saving)")

    def _get_output_path(self, result: SaveResult) -> Path:
        """
        Determine the output path for a result.

        This method should be overridden by subclasses to determine
        where each result should be written.

        Args:
            result: The SaveResult to be saved

        Returns:
            Path where the result should be written

        Example:
            def _get_output_path(self, result):
                output_dir = Path(self.config['output_dir'])
                source_file = result.additional_data.get('_source_file', 'default')
                return output_dir / source_file
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _get_output_path()"
        )

    def _format_result(self, result: SaveResult) -> Any:
        """
        Format a result for output.

        This method should be overridden by subclasses to convert
        SaveResult into the format needed for output (dict, string, etc.).

        Args:
            result: The SaveResult to format

        Returns:
            Formatted data ready for writing

        Example:
            def _format_result(self, result):
                content = result.model_output['choices'][0]['message']['content']
                return {
                    "id": result.request_id,
                    "response": content,
                    "tokens": result.model_output.get('usage', {}).get('total_tokens', 0)
                }
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _format_result()"
        )

    def _write_result(self, output_path: Path, formatted_data: Any) -> None:
        """
        Write formatted data to the output path.

        This method should be overridden by subclasses to implement
        the actual writing logic.

        Args:
            output_path: Path where data should be written
            formatted_data: Data from _format_result()

        Example:
            def _write_result(self, output_path, formatted_data):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(formatted_data) + '\\n')
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _write_result()"
        )

    def _should_save_result(self, result: SaveResult) -> bool:
        """
        Determine if a result should be saved.

        Override this method to implement custom filtering.
        Default implementation saves all results.

        Args:
            result: The SaveResult to check

        Returns:
            True to save, False to skip

        Example:
            def _should_save_result(self, result):
                # Skip failed requests
                return result.error is None
        """
        return True

    def _on_save_start(self, result: SaveResult):
        """
        Hook called before saving a result.

        Override for custom logging, preprocessing, etc.

        Args:
            result: The SaveResult about to be saved

        Example:
            def _on_save_start(self, result):
                logger.debug(f"Saving result for {result.request_id}")
        """
        pass

    def _on_save_complete(self, result: SaveResult, output_path: Path):
        """
        Hook called after successfully saving a result.

        Override for custom logging, metrics, etc.

        Args:
            result: The SaveResult that was saved
            output_path: Path where result was written

        Example:
            def _on_save_complete(self, result, output_path):
                logger.info(f"Saved {result.request_id} to {output_path}")
        """
        pass

    def _on_save_error(self, result: SaveResult, error: Exception):
        """
        Hook called when an error occurs while saving.

        Override for custom error handling. By default, logs a warning
        and continues.

        Args:
            result: The SaveResult that failed to save
            error: The exception that occurred

        Example:
            def _on_save_error(self, result, error):
                logger.error(f"Failed to save {result.request_id}: {error}")
                raise  # Re-raise to stop entire batch
        """
        logger.warning(f"Error saving result {result.request_id}: {error}. Skipping.")

    def _stream_save(self, result: SaveResult) -> None:
        """
        Main streaming template method.

        Orchestrates the streaming save process by:
        1. Checking if result should be saved
        2. Getting output path
        3. Formatting the result
        4. Writing to output
        5. Calling hooks at appropriate times
        6. Handling errors gracefully

        Args:
            result: The SaveResult to save

        This is the main entry point for streaming. Subclasses typically
        call this from their save() method.
        """
        # Check if we should save this result
        if not self._should_save_result(result):
            logger.debug(f"Skipping result {result.request_id}")
            return

        # Call pre-save hook
        self._on_save_start(result)

        try:
            # Get output path
            output_path = self._get_output_path(result)

            # Format result
            formatted_data = self._format_result(result)

            # Write result
            with self._lock:
                self._write_result(output_path, formatted_data)

                # Flush if configured
                if self.immediate_flush:
                    self._flush(output_path)

            # Call post-save hook
            self._on_save_complete(result, output_path)

        except Exception as e:
            # Call error handling hook
            self._on_save_error(result, e)

    def _flush(self, output_path: Path):
        """
        Flush output to disk.

        Override if using file handles that need explicit flushing.
        Default implementation does nothing.

        Args:
            output_path: Path that may need flushing
        """
        pass


class OutputFormatterMixin:
    """
    Mixin class for flexible output formatting.

    Provides extensible hooks for customizing how results are
    transformed into output format.

    Usage:
        class MySaver(OutputFormatterMixin, ResultSaver):
            def format_output(self, result):
                content = result.model_output['choices'][0]['message']['content']
                return {
                    "id": result.request_id,
                    "response": content,
                    "timestamp": datetime.now().isoformat()
                }
    """

    def format_output(self, result: SaveResult) -> Dict[str, Any]:
        """
        Format a SaveResult into a dictionary for output.

        This method can be overridden to customize output format.
        Default implementation creates a standard structure.

        Args:
            result: SaveResult object containing model output and metadata

        Returns:
            Dictionary to be serialized/ written to output

        Example override:
            def format_output(self, result):
                content = result.model_output['choices'][0]['message']['content']
                return {
                    "id": result.request_id,
                    "response": content,
                    "tokens": result.model_output.get('usage', {}).get('total_tokens', 0),
                    "model": result.model_output.get('model', 'unknown')
                }
        """
        from datetime import datetime

        output_data = {
            'request_id': result.request_id,
            'model_output': result.model_output,
            'additional_data': result.additional_data,
            'timestamp': datetime.now().isoformat()
        }

        if result.error:
            output_data['error'] = result.error

        return output_data

    def extract_content(self, result: SaveResult) -> Optional[str]:
        """
        Extract the main content from a SaveResult.

        This is a convenience method for subclasses that want to
        extract just the generated text from the model output.

        Args:
            result: SaveResult object

        Returns:
            Generated text content, or None if not found

        Example:
            def format_output(self, result):
                content = self.extract_content(result)
                return {"response": content}
        """
        try:
            return result.model_output['choices'][0]['message']['content']
        except (KeyError, IndexError, TypeError):
            return None

    def extract_usage(self, result: SaveResult) -> Dict[str, int]:
        """
        Extract token usage information from a SaveResult.

        This is a convenience method for subclasses that want to
        include token counts in their output.

        Args:
            result: SaveResult object

        Returns:
            Dictionary with usage stats (empty if not found)

        Example:
            def format_output(self, result):
                usage = self.extract_usage(result)
                return {"tokens": usage.get('total_tokens', 0)}
        """
        try:
            return result.model_output.get('usage', {})
        except (AttributeError, TypeError):
            return {}


class MultimodalOutputMixin:
    """
    Mixin class for handling multimodal outputs.

    Provides extensible hooks for customizing how multimodal
    results (text + potential image references) are formatted.

    Usage:
        class MySaver(MultimodalOutputMixin, ResultSaver):
            def format_multimodal_output(self, result):
                data = self.format_output(result)
                # Add image references from additional_data
                if result.additional_data and 'images' in result.additional_data:
                    data['input_images'] = result.additional_data['images']
                return data
    """

    def format_multimodal_output(self, result: SaveResult) -> Dict[str, Any]:
        """
        Format a multimodal SaveResult for output.

        This method can be overridden to customize multimodal output.
        Default implementation calls format_output() and adds image info.

        Args:
            result: SaveResult object (may have images in additional_data)

        Returns:
            Dictionary with multimodal output information

        Example override:
            def format_multimodal_output(self, result):
                data = self.format_output(result)
                content = self.extract_content(result)
                # Parse image references from content
                data['has_images'] = '<image>' in content
                return data
        """
        output_data = self.format_output(result)

        # Add image information if available
        if result.additional_data:
            if 'images' in result.additional_data:
                output_data['input_images'] = result.additional_data['images']
            elif 'image' in result.additional_data:
                output_data['input_images'] = [result.additional_data['image']]

        return output_data


class BatchWriterMixin:
    """
    Mixin class for batched writing optimization.

    Provides buffering and batch writing capabilities for savers
    that can benefit from writing multiple results at once.

    Usage:
        class MyBatchSaver(BatchWriterMixin, ResultSaver):
            def _write_batch(self, batch_data):
                # Write all results in batch_data at once
                with open(self.output_path, 'a') as f:
                    for item in batch_data:
                        f.write(json.dumps(item) + '\\n')
    """

    def _initialize_batch(self, default_batch_size: int = 100):
        """
        Initialize batch-specific configuration.

        Subclasses should call this in their _initialize() method.

        Args:
            default_batch_size: Default batch size if not in config
        """
        self.batch_size = self.config.get('batch_size', default_batch_size)
        self._batch_buffer = []
        self._batch_lock = threading.Lock()

    def _add_to_batch(self, formatted_data: Any) -> None:
        """
        Add formatted data to batch buffer.

        If buffer size reaches batch_size, automatically flushes.

        Args:
            formatted_data: Formatted data to buffer
        """
        with self._batch_lock:
            self._batch_buffer.append(formatted_data)

            if len(self._batch_buffer) >= self.batch_size:
                self._flush_batch()

    def _flush_batch(self) -> None:
        """
        Flush buffered batch data to storage.

        This method should be overridden by subclasses to implement
        the actual batch writing logic.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _flush_batch()"
        )

    def _get_batch(self) -> List[Any]:
        """
        Get and clear the current batch buffer.

        Returns:
            List of buffered items, and clears the buffer

        Example:
            def _flush_batch(self):
                batch = self._get_batch()
                # Write batch to storage
        """
        with self._batch_lock:
            batch = self._batch_buffer
            self._batch_buffer = []
            return batch
