"""
Configuration loading and management module.

Loads YAML configuration files and dynamically imports loader/saver classes.
"""
import yaml
import importlib
from pathlib import Path
from typing import Dict, Any, Type

from ..loaders.base import DataLoader
from ..savers.base import ResultSaver
from ..adapters.base import ModelAdapter


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
        'num_rollouts': 1,
        'model_name': 'default',
        'temperature': 0.7,
        'max_tokens': 1000,
        'system_prompt': '',
        'load_balancing_strategy': 'round_robin',
        'progress_report_interval': 10,
        'adapter_class': 'OpenAIAdapter',
        'adapter_params': {},
        'enable_checkpoint': False,
        'checkpoint_path': 'checkpoints/batch_checkpoint.json',
        'checkpoint_interval': 10,
    }

    for key, value in runner_defaults.items():
        if key not in config['runner']:
            config['runner'][key] = value


def get_loader_class(class_name: str, module_prefix: str = 'src.loaders') -> Type[DataLoader]:
    """
    Dynamically import loader class.

    Args:
        class_name: Name of the loader class (e.g., 'JSONDataLoader')
        module_prefix: Module prefix for import (default: 'src.loaders')

    Returns:
        DataLoader class

    Raises:
        ValueError: If class is not found
    """
    # Built-in loaders
    builtin_loaders = {
        'JSONDataLoader': 'json_loader',
        'MultimodalJSONDataLoader': 'json_loader',
        'JSONLDataLoader': 'jsonl_loader',
        'MultimodalJSONLDataLoader': 'jsonl_loader',
        'CSVDataLoader': 'csv_loader',
        'PromptListLoader': 'prompt_list_loader',
    }

    if class_name not in builtin_loaders:
        # Try to import from custom module
        # Assume class_name is defined in a file named after the class (snake_case)
        module_name = _camel_to_snake(class_name)
        try:
            module = importlib.import_module(f'{module_prefix}.{module_name}')
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Unknown loader class: {class_name}. Error: {e}")

    # Import built-in loader
    module_name = builtin_loaders[class_name]
    module = importlib.import_module(f'{module_prefix}.{module_name}')
    return getattr(module, class_name)


def get_saver_class(class_name: str, module_prefix: str = 'src.savers') -> Type[ResultSaver]:
    """
    Dynamically import saver class.

    Args:
        class_name: Name of the saver class (e.g., 'JSONResultSaver')
        module_prefix: Module prefix for import (default: 'src.savers')

    Returns:
        ResultSaver class

    Raises:
        ValueError: If class is not found
    """
    # Built-in savers
    builtin_savers = {
        'JSONResultSaver': 'json_saver',
        'JSONLResultSaver': 'jsonl_saver',
        'CSVResultSaver': 'csv_saver',
        'ConsoleResultSaver': 'console_saver',
    }

    if class_name not in builtin_savers:
        # Try to import from custom module
        module_name = _camel_to_snake(class_name)
        try:
            module = importlib.import_module(f'{module_prefix}.{module_name}')
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Unknown saver class: {class_name}. Error: {e}")

    # Import built-in saver
    module_name = builtin_savers[class_name]
    module = importlib.import_module(f'{module_prefix}.{module_name}')
    return getattr(module, class_name)


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
