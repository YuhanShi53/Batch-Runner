"""
Configuration loading and management module.

Loads YAML configuration files and dynamically imports loader/saver classes.
Integrates with the registration system for custom loaders/savers.
"""
import yaml
import importlib
from pathlib import Path
from typing import Dict, Any, Type

from ..loaders.base import DataLoader
from ..savers.base import ResultSaver
from ..adapters.base import ModelAdapter
from .registry import (
    get_registered_loader,
    get_registered_saver,
    load_custom_modules_from_config
)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Load custom modules before validation (in case they register loaders/savers)
    load_custom_modules_from_config(config)

    # Validate required fields
    _validate_config(config)

    return config


def _validate_config(config: Dict[str, Any]):
    """
    Validate configuration has required fields.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If required fields are missing
    """
    required_fields = ['loader', 'saver', 'runner']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")

    # Validate loader config
    if 'class' not in config['loader']:
        raise ValueError("Loader config must specify 'class'")

    if 'params' not in config['loader']:
        config['loader']['params'] = {}

    # Validate saver config
    if 'class' not in config['saver']:
        raise ValueError("Saver config must specify 'class'")

    if 'params' not in config['saver']:
        config['saver']['params'] = {}

    # Set defaults for runner config
    runner_defaults = {
        'max_concurrency': 10,
        'max_retries': 3,
        'retry_delay': 1.0,
        'request_timeout': 120,
        'http_max_connections': 4096,  # HTTP client max connections (shared across all requests)
        'http_max_keepalive_connections': 1000,  # HTTP client keepalive connections (shared)
        'http2': True,  # Enable HTTP/2
        'model_name': 'default',
        'temperature': 0.7,
        'max_tokens': 1000,
        'rollout_n': 1,
        'system_prompt': '',
        'load_balancing_strategy': 'round_robin',
        'progress_report_interval': 10,
        'adapter_class': 'OpenAIAdapter',
        'adapter_params': {},
        'health_check_interval': 30,
        'max_failures': 5,
        'allow_unhealthy_fallback': False,
        'success_rate_threshold': 0.5,
        'success_rate_window': 10,
        'max_active_requests': 50,
        'resume': True,  # Enable resuming from existing output (default: True)
        'resume_backend': 'legacy_output_scan',
        'producer_prefetch': 100,
        'writer_queue_size': 1000,
        'writer_batch_size': 100,
        'writer_flush_interval_ms': 100,
        'writer_workers': 1,
        'selection_sample_size': 2,
        'max_inflight_cost': 0.0,
        'image_encode_workers': 4,
    }

    for key, value in runner_defaults.items():
        if key not in config['runner']:
            config['runner'][key] = value

    if 'producer_prefetch' not in config['runner'] and 'stream_queue_size' in config['runner']:
        config['runner']['producer_prefetch'] = config['runner']['stream_queue_size']


def get_loader_class(class_name: str, module_prefix: str = 'src.loaders') -> Type[DataLoader]:
    """
    Dynamically import loader class.

    Checks in this order:
    1. Custom registered loaders (via @register_loader decorator)
    2. Built-in loaders
    3. Custom modules in src/loaders directory

    Args:
        class_name: Name of the loader class (e.g., 'JSONDataLoader')
        module_prefix: Module prefix for import (default: 'src.loaders')

    Returns:
        DataLoader class

    Raises:
        ValueError: If class is not found

    Example:
        # Use a built-in loader
        loader_class = get_loader_class('JSONDataLoader')

        # Use a custom registered loader
        # (assuming MyCustomLoader was registered with @register_loader)
        loader_class = get_loader_class('MyCustomLoader')
    """
    # Check custom registered loaders first
    custom_loader = get_registered_loader(class_name)
    if custom_loader is not None:
        return custom_loader

    # Built-in loaders
    builtin_loaders = {
        'JSONDataLoader': 'json_loader',
        'MultimodalJSONDataLoader': 'json_loader',
        'JSONLDataLoader': 'jsonl_loader',
        'MultimodalJSONLDataLoader': 'jsonl_loader',
        'CSVDataLoader': 'csv_loader',
        'PromptListLoader': 'prompt_list_loader',
        'DirectoryJSONLDataLoader': 'directory_jsonl_loader',
        'MultimodalDirectoryJSONLDataLoader': 'directory_jsonl_loader',
    }

    if class_name in builtin_loaders:
        # Import built-in loader
        module_name = builtin_loaders[class_name]
        module = importlib.import_module(f'{module_prefix}.{module_name}')
        return getattr(module, class_name)

    # Try to import from custom module
    # Assume class_name is defined in a file named after the class (snake_case)
    module_name = _camel_to_snake(class_name)
    try:
        module = importlib.import_module(f'{module_prefix}.{module_name}')
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ValueError(
            f"Unknown loader class: {class_name}. "
            f"Error: {e}. "
            f"Make sure the class is registered with @register_loader decorator "
            f"or defined in {module_prefix}/{module_name}.py"
        )


def get_saver_class(class_name: str, module_prefix: str = 'src.savers') -> Type[ResultSaver]:
    """
    Dynamically import saver class.

    Checks in this order:
    1. Custom registered savers (via @register_saver decorator)
    2. Built-in savers
    3. Custom modules in src/savers directory

    Args:
        class_name: Name of the saver class (e.g., 'JSONResultSaver')
        module_prefix: Module prefix for import (default: 'src.savers')

    Returns:
        ResultSaver class

    Raises:
        ValueError: If class is not found

    Example:
        # Use a built-in saver
        saver_class = get_saver_class('JSONLResultSaver')

        # Use a custom registered saver
        # (assuming MyCustomSaver was registered with @register_saver)
        saver_class = get_saver_class('MyCustomSaver')
    """
    # Check custom registered savers first
    custom_saver = get_registered_saver(class_name)
    if custom_saver is not None:
        return custom_saver

    # Built-in savers
    builtin_savers = {
        'JSONResultSaver': 'json_saver',
        'JSONLResultSaver': 'jsonl_saver',
        'CSVResultSaver': 'csv_saver',
        'ConsoleResultSaver': 'console_saver',
        'DirectoryJSONLResultSaver': 'directory_jsonl_saver',
    }

    if class_name in builtin_savers:
        # Import built-in saver
        module_name = builtin_savers[class_name]
        module = importlib.import_module(f'{module_prefix}.{module_name}')
        return getattr(module, class_name)

    # Try to import from custom module
    module_name = _camel_to_snake(class_name)
    try:
        module = importlib.import_module(f'{module_prefix}.{module_name}')
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ValueError(
            f"Unknown saver class: {class_name}. "
            f"Error: {e}. "
            f"Make sure the class is registered with @register_saver decorator "
            f"or defined in {module_prefix}/{module_name}.py"
        )


def _camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case."""
    import re
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def get_adapter_class(class_name: str, module_prefix: str = 'src.adapters') -> Type[ModelAdapter]:
    """
    Dynamically import adapter class.

    Args:
        class_name: Name of the adapter class (e.g., 'OpenAIAdapter')
        module_prefix: Module prefix for import (default: 'src.adapters')

    Returns:
        ModelAdapter class

    Raises:
        ValueError: If class is not found
    """
    # Built-in adapters
    builtin_adapters = {
        'OpenAIAdapter': 'openai_adapter',
        'SimpleAdapter': 'simple_adapter',
    }

    if class_name not in builtin_adapters:
        # Try to import from custom module
        module_name = _camel_to_snake(class_name)
        try:
            module = importlib.import_module(f'{module_prefix}.{module_name}')
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Unknown adapter class: {class_name}. Error: {e}")

    # Import built-in adapter
    module_name = builtin_adapters[class_name]
    module = importlib.import_module(f'{module_prefix}.{module_name}')
    return getattr(module, class_name)
