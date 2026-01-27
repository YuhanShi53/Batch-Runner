"""
Retry mechanism with exponential backoff.

Provides decorator for retrying functions with exponential backoff.
"""
import time
import functools
from typing import Callable, Type, Tuple, Optional
import logging


logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable] = None
):
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        exponential_base: Multiplier for exponential backoff
        exceptions: Tuple of exceptions to catch and retry
        on_retry: Optional callback function called on each retry

    Returns:
        Decorated function

    Example:
        >>> @retry_with_backoff(max_retries=3, base_delay=1.0)
        ... def fetch_data(url):
        ...     return requests.get(url)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(f"Failed after {max_retries} retries: {e}")
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(
                        base_delay * (exponential_base ** attempt),
                        max_delay
                    )

                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )

                    # Call on_retry callback if provided
                    if on_retry:
                        try:
                            on_retry(attempt, e)
                        except Exception:
                            pass

                    time.sleep(delay)

            # Should never reach here, but just in case
            raise last_exception

        return wrapper
    return decorator


class RetryContext:
    """
    Context manager for retrying operations.

    Example:
        >>> with RetryContext(max_retries=3) as retry:
        ...     while retry.should_continue():
        ...         try:
        ...             result = risky_operation()
        ...             break
        ...         except Exception as e:
        ...             retry.handle_error(e)
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.attempt = 0
        self.last_error = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def should_continue(self) -> bool:
        """Check if we should continue retrying."""
        return self.attempt <= self.max_retries

    def handle_error(self, error: Exception):
        """Handle an error and sleep before next retry."""
        self.last_error = error

        if self.attempt == self.max_retries:
            logger.error(f"Failed after {self.max_retries} retries: {error}")
            raise error

        # Calculate delay
        delay = min(
            self.base_delay * (self.exponential_base ** self.attempt),
            self.max_delay
        )

        logger.warning(
            f"Attempt {self.attempt + 1}/{self.max_retries} failed: {error}. "
            f"Retrying in {delay:.2f}s..."
        )

        time.sleep(delay)
        self.attempt += 1
