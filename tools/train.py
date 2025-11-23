"""
Train Romanian PP-OCRv3 text recognition model.

Usage:
    python tools/train.py [--config CONFIG] [--epochs EPOCHS] [--resume CHECKPOINT]
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train(config_path: Path, epochs: int = None, resume: str = None):
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    logger.info("Starting training")
    logger.info(f"Config: {config_path}")

    # Build training command
    cmd = [
        sys.executable,
        'PaddleOCR/tools/train.py',
        '-c', str(config_path),
    ]

    # Override epochs if specified
    if epochs is not None:
        cmd.extend(['-o', f'Global.epoch_num={epochs}'])
        logger.info(f"Epochs: {epochs}")

    # Resume from checkpoint if specified
    if resume:
        resume_path = Path(resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

        cmd.extend(['-o', f'Global.checkpoints={resume}'])
        logger.info(f"Resuming from: {resume}")

    logger.info(f"\nCommand: {' '.join(cmd)}\n")
    logger.info("Training - logs below")

    # Run training
    try:
        result = subprocess.run(cmd, check=True)
        logger.info("\nDone!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        logger.error(f"\nTraining failed with exit code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
        return 1


def main():
    parser = argparse.ArgumentParser(description='Train Romanian OCR model')
    parser.add_argument('--config', type=str,
                        default='PaddleOCR/configs/rec/romanian/romanian_ppocrv3_rec.yml',
                        help='Path to training config file')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')

    args = parser.parse_args()

    config_path = Path(args.config)

    # Pre-flight checks
    logger.info("Checking setup...")

    # Check PaddleOCR exists
    if not Path('PaddleOCR/tools/train.py').exists():
        logger.error("PaddleOCR not found!")
        sys.exit(1)

    # Check training data exists
    train_data = Path('data/document_pages/train_lines')
    if not train_data.exists():
        logger.error(f"Training data not found: {train_data}")
        logger.error("Run: python scripts/generate_training_data.py")
        sys.exit(1)

    # Check pretrained model exists
    pretrained = Path('models/pretrained/en_PP-OCRv3_rec_train')
    if not pretrained.exists():
        logger.warning(f"Pretrained model not found: {pretrained}")
        logger.warning("Run: python tools/download_pretrained.py --models en_train")
        logger.warning("Will start from scratch (not recommended)\n")

    logger.info("All good\n")

    # Start training
    exit_code = train(config_path, args.epochs, args.resume)
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
