"""
Romanian OCR inference utility.

Usage:
    python tools/infer.py --image path/to/image.jpg [--model MODEL]
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def infer(image_path: Path, model_dir: Path, char_dict: Path):
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not model_dir.exists():
        raise FileNotFoundError(f"Model not found: {model_dir}")

    if not char_dict.exists():
        raise FileNotFoundError(f"Character dictionary not found: {char_dict}")

    logger.info("Running inference")
    logger.info(f"Model: {model_dir}")
    logger.info(f"Image: {image_path}\n")

    # Build inference command
    cmd = [
        sys.executable,
        'PaddleOCR/tools/infer/predict_rec.py',
        f'--image_dir={image_path}',
        f'--rec_model_dir={model_dir}',
        f'--rec_char_dict_path={char_dict}',
    ]

    # Run inference
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        logger.error(f"\nInference failed with exit code {e.returncode}")
        return e.returncode


def main():
    parser = argparse.ArgumentParser(description='Romanian OCR inference')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to image file or directory')
    parser.add_argument('--model', type=str,
                        default='models/romanian_ppocrv3_infer',
                        help='Path to inference model directory')
    parser.add_argument('--char-dict', type=str,
                        default='config/romanian_charset.txt',
                        help='Path to character dictionary')

    args = parser.parse_args()

    image_path = Path(args.image)
    model_dir = Path(args.model)
    char_dict = Path(args.char_dict)

    # Pre-flight checks
    if not Path('PaddleOCR/tools/infer/predict_rec.py').exists():
        logger.error("PaddleOCR not found!")
        sys.exit(1)

    # Start inference
    exit_code = infer(image_path, model_dir, char_dict)
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
