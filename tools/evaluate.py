"""
Evaluate Romanian OCR model performance.

Usage:
    python tools/evaluate.py [--model MODEL] [--config CONFIG]
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate(config_path: Path, model_path: str = None):
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    logger.info("Evaluating model")
    logger.info(f"Config: {config_path}")

    # Build evaluation command
    cmd = [
        sys.executable,
        'PaddleOCR/tools/eval.py',
        '-c', str(config_path),
    ]

    # Override model path if specified
    if model_path:
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model not found: {model_path_obj}")

        cmd.extend(['-o', f'Global.pretrained_model={model_path}'])
        logger.info(f"Model: {model_path}")

    logger.info(f"\nCommand: {' '.join(cmd)}\n")

    # Run evaluation
    try:
        result = subprocess.run(cmd, check=True)
        logger.info("\nDone!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        logger.error(f"\nEvaluation failed with exit code {e.returncode}")
        return e.returncode


def main():
    parser = argparse.ArgumentParser(description='Evaluate Romanian OCR model')
    parser.add_argument('--config', type=str,
                        default='PaddleOCR/configs/rec/romanian/romanian_ppocrv3_rec.yml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model checkpoint (overrides config)')

    args = parser.parse_args()

    config_path = Path(args.config)

    # Pre-flight checks
    logger.info("Checking setup...")

    # Check PaddleOCR exists
    if not Path('PaddleOCR/tools/eval.py').exists():
        logger.error("PaddleOCR not found!")
        sys.exit(1)

    # Check validation data exists
    val_data = Path('data/document_pages/val_lines')
    if not val_data.exists():
        logger.error(f"Validation data not found: {val_data}")
        logger.error("Run: python scripts/generate_training_data.py")
        sys.exit(1)

    logger.info("All good\n")

    # Start evaluation
    exit_code = evaluate(config_path, args.model)
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
