"""
Registration system for custom loaders and savers.

This module provides a decorator-based registration mechanism that allows
users to register custom DataLoader and ResultSaver classes from outside
the src/loaders and src/savers directories without modifying the project
source code.

Usage:
    # In your custom.py file (outside the project):
    from src.loaders.base import DataLoader
    from src.savers.base import ResultSaver
    from src.utils.registry import register_loader, register_saver

    @register_loader
    class MyCustomLoader(DataLoader):
        def _initialize(self):
            # Your initialization logic
            pass

        def load(self):
            # Your loading logic
            yield LoadResult(...)

    @register_saver
    class MyCustomSaver(ResultSaver):
        def _initialize(self):
            # Your initialization logic
            pass

        def save(self, result):
            # Your saving logic
            pass

    # Then in your config.yaml:
    loader:
      class: MyCustomLoader
      params:
        # Your params
    saver:
      class: MyCustomSaver
      params:
        # Your params
"""
import sys
import importlib
from typing import Type, Dict, Optional, Any
from pathlib import Path

from ..loaders.base import DataLoader
from ..savers.base import ResultSaver


# Global registries
_custom_loaders: Dict[str, Type[DataLoader]] = {}
_custom_savers: Dict[str, Type[ResultSaver]] = {}


def register_loader(cls: Type[DataLoader]) -> Type[DataLoader]:
    """
    Decorator to register a custom DataLoader class.

    The class name will be used as the registration key.

    Usage:
        @register_loader
        class MyCustomLoader(DataLoader):
            def _initialize(self):
                self.data = []

            def load(self):
                for item in self.data:
                    yield LoadResult(...)

    Args:
        cls: The DataLoader class to register

    Returns:
        The same class (unchanged)

    Example:
        # In config.yaml:
        loader:
          class: MyCustomLoader
          params:
            data_source: "path/to/data"
    """
    if not issubclass(cls, DataLoader):
        raise TypeError(
            f"Class {cls.__name__} must inherit from DataLoader"
        )

    _custom_loaders[cls.__name__] = cls
    return cls


def register_saver(cls: Type[ResultSaver]) -> Type[ResultSaver]:
    """
    Decorator to register a custom ResultSaver class.

    The class name will be used as the registration key.

    Usage:
        @register_saver
        class MyCustomSaver(ResultSaver):
            def _initialize(self):
                self.output_path = Path(self.config['output'])

            def save(self, result):
                with open(self.output_path, 'a') as f:
                    f.write(result.model_output)

    Args:
        cls: The ResultSaver class to register

    Returns:
        The same class (unchanged)

    Example:
        # In config.yaml:
        saver:
          class: MyCustomSaver
          params:
            output: "results/output.jsonl"
    """
    if not issubclass(cls, ResultSaver):
        raise TypeError(
            f"Class {cls.__name__} must inherit from ResultSaver"
        )

    _custom_savers[cls.__name__] = cls
    return cls


def register_loader_class(name: str, cls: Type[DataLoader]) -> None:
    """
    Register a custom DataLoader class with a specific name.

    Use this function when you want to register a class with a different
    name than the class name itself.

    Args:
        name: The name to register the class under
        cls: The DataLoader class to register

    Example:
        class MyCustomLoader(DataLoader):
            ...

        # Register with a shorter name
        register_loader_class('MyLoader', MyCustomLoader)

        # Now you can use 'MyLoader' in config.yaml
    """
    if not issubclass(cls, DataLoader):
        raise TypeError(
            f"Class {cls.__name__} must inherit from DataLoader"
        )

    _custom_loaders[name] = cls


def register_saver_class(name: str, cls: Type[ResultSaver]) -> None:
    """
    Register a custom ResultSaver class with a specific name.

    Use this function when you want to register a class with a different
    name than the class name itself.

    Args:
        name: The name to register the class under
        cls: The ResultSaver class to register

    Example:
        class MyCustomSaver(ResultSaver):
            ...

        # Register with a shorter name
        register_saver_class('MySaver', MyCustomSaver)

        # Now you can use 'MySaver' in config.yaml
    """
    if not issubclass(cls, ResultSaver):
        raise TypeError(
            f"Class {cls.__name__} must inherit from ResultSaver"
        )

    _custom_savers[name] = cls


def get_registered_loader(name: str) -> Optional[Type[DataLoader]]:
    """
    Get a registered custom loader class by name.

    Args:
        name: The name of the loader class

    Returns:
        The loader class if found, None otherwise
    """
    return _custom_loaders.get(name)


def get_registered_saver(name: str) -> Optional[Type[ResultSaver]]:
    """
    Get a registered custom saver class by name.

    Args:
        name: The name of the saver class

    Returns:
        The saver class if found, None otherwise
    """
    return _custom_savers.get(name)


def list_registered_loaders() -> list[str]:
    """
    List all registered custom loader names.

    Returns:
        List of registered loader class names
    """
    return list(_custom_loaders.keys())


def list_registered_savers() -> list[str]:
    """
    List all registered custom saver names.

    Returns:
        List of registered saver class names
    """
    return list(_custom_savers.keys())


def unregister_loader(name: str) -> bool:
    """
    Unregister a custom loader class.

    Args:
        name: The name of the loader class to unregister

    Returns:
        True if the loader was unregistered, False if not found
    """
    if name in _custom_loaders:
        del _custom_loaders[name]
        return True
    return False


def unregister_saver(name: str) -> bool:
    """
    Unregister a custom saver class.

    Args:
        name: The name of the saver class to unregister

    Returns:
        True if the saver was unregistered, False if not found
    """
    if name in _custom_savers:
        del _custom_savers[name]
        return True
    return False


def load_custom_modules_from_config(config: Dict[str, Any]) -> None:
    """
    Load custom modules specified in configuration.

    This function looks for 'custom_modules' in the config and imports
    them, which will trigger any @register_loader or @register_saver
    decorators in those modules.

    Config format:
        custom_modules:
          - path/to/custom_loaders.py
          - path/to/custom_savers.py

    Args:
        config: The configuration dictionary
    """
    custom_modules = config.get('custom_modules', [])

    for module_path in custom_modules:
        try:
            # Convert file path to module path
            module_path_obj = Path(module_path)

            # Add parent directory to sys.path if needed
            parent_dir = str(module_path_obj.parent)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)

            # Import the module (without .py extension)
            module_name = module_path_obj.stem
            importlib.import_module(module_name)

        except Exception as e:
            # Log warning but don't fail - custom modules are optional
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to load custom module {module_path}: {e}")


def clear_registries() -> None:
    """
    Clear all registered loaders and savers.

    Useful for testing or when you need to reload modules.
    """
    _custom_loaders.clear()
    _custom_savers.clear()
