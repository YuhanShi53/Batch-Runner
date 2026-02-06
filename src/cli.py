"""
Command-line interface for vLLM batch runner.

Provides CLI for executing batch inference jobs.
"""
import argparse
import sys
from pathlib import Path

from .utils.config import load_config, get_loader_class, get_saver_class, get_adapter_class
from .utils.logger import setup_logging_from_config
from .batch_runner import BatchRunner, BatchConfig
from .loaders.base import DataLoader
from .savers.base import ResultSaver


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="vLLM Batch Inference Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  python -m src.cli --config configs/config.yaml

  # Run with custom concurrency
  python -m src.cli --config configs/config.yaml --concurrency 20

  # Run multiple rollouts
  python -m src.cli --config configs/config.yaml --rollouts 3

  # Override model parameters
  python -m src.cli --config configs/config.yaml --temperature 0.5 --max-tokens 2000
        """
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file (default: configs/config.yaml)'
    )

    parser.add_argument(
        '--concurrency',
        type=int,
        help='Override max concurrency'
    )

    parser.add_argument(
        '--rollouts',
        type=int,
        help='Override number of rollouts per sample'
    )

    parser.add_argument(
        '--model',
        type=str,
        help='Override model name'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        help='Override sampling temperature'
    )

    parser.add_argument(
        '--max-tokens',
        type=int,
        help='Override max tokens to generate'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging (DEBUG level)'
    )

    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Apply CLI overrides
    if args.concurrency:
        config['runner']['max_concurrency'] = args.concurrency
    if args.rollouts:
        config['runner']['num_rollouts'] = args.rollouts
    if args.model:
        config['runner']['model_name'] = args.model
    if args.temperature is not None:
        config['runner']['temperature'] = args.temperature
    if args.max_tokens is not None:
        config['runner']['max_tokens'] = args.max_tokens
    if args.verbose:
        config['logging']['level'] = 'DEBUG'

    # Setup logging
    setup_logging_from_config(config)

    import logging
    logger = logging.getLogger(__name__)

    # Initialize loader
    try:
        loader_class = get_loader_class(config['loader']['class'])
        loader = loader_class(config['loader']['params'])
        logger.info(f"Loaded data loader: {config['loader']['class']}")
    except Exception as e:
        logger.error(f"Error initializing loader: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Initialize saver
    try:
        saver_class = get_saver_class(config['saver']['class'])
        saver = saver_class(config['saver']['params'])
        logger.info(f"Loaded result saver: {config['saver']['class']}")
    except Exception as e:
        logger.error(f"Error initializing saver: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Initialize adapter
    try:
        adapter_class_name = config['runner'].get('adapter_class', 'OpenAIAdapter')
        adapter_class = get_adapter_class(adapter_class_name)
        adapter_params = config['runner'].get('adapter_params', {})
        adapter = adapter_class(**adapter_params)
        logger.info(f"Loaded model adapter: {adapter_class_name}")
    except Exception as e:
        logger.error(f"Error initializing adapter: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Create batch config with adapter
    try:
        # Remove adapter_class and adapter_params from runner config before creating BatchConfig
        runner_config = config['runner'].copy()
        runner_config.pop('adapter_class', None)
        runner_config.pop('adapter_params', None)
        batch_config = BatchConfig(**runner_config)
        batch_config.adapter = adapter
    except Exception as e:
        logger.error(f"Error creating batch config: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Run batch inference
    try:
        runner = BatchRunner(batch_config, loader, saver)
        runner.run()
    except KeyboardInterrupt:
        logger.info("\nBatch inference interrupted by user")
        saver.cleanup()
        logging.shutdown()
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during batch inference: {e}")
        import traceback
        traceback.print_exc()
        saver.cleanup()
        logging.shutdown()
        sys.exit(1)

    # Cleanup logging on successful completion
    logging.shutdown()


if __name__ == '__main__':
    main()
