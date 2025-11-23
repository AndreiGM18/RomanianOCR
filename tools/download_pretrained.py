"""
Download pretrained models for baselines and transfer learning.

Usage:
    python tools/download_pretrained.py              # all models
    python tools/download_pretrained.py --models en_train
"""

import argparse
import logging
import tarfile
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


MODELS = {
    # Training checkpoints
    'en_train': {
        'url': 'https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_train.tar',
        'output_dir': 'PaddleOCR/en_PP-OCRv3_rec_train',
        'description': 'EN PP-OCRv3 training checkpoint (for transfer learning)',
    },

    # Recognition models - Inference
    'en_v3_rec': {
        'url': 'https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar',
        'output_dir': 'PaddleOCR/en_PP-OCRv3_rec_infer',
        'description': 'EN PP-OCRv3 recognition inference (baseline)',
    },
    'en_v4_mobile_rec': {
        'url': 'https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_mobile_rec_infer.tar',
        'output_dir': 'baselines/en_PP-OCRv4_mobile_rec_infer',
        'description': 'EN PP-OCRv4 mobile recognition (baseline)',
    },
    'v4_server_rec': {
        'url': 'https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_server_rec_infer.tar',
        'output_dir': 'baselines/PP-OCRv4_server_rec_infer',
        'description': 'PP-OCRv4 server recognition (baseline)',
    },
    'v5_latin': {
        'url': 'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/latin_ppocr_mobile_v2.0_rec_infer.tar',
        'output_dir': 'PaddleOCR/latin_ppocrv5_infer',
        'description': 'Latin PP-OCRv5 multilingual recognition (baseline)',
    },

    # Detection models
    'en_v3_det': {
        'url': 'https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar',
        'output_dir': 'PaddleOCR/en_PP-OCRv3_det_infer',
        'description': 'EN PP-OCRv3 detection',
    },
    'ch_v3_det': {
        'url': 'https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar',
        'output_dir': 'baselines/ch_PP-OCRv3_det',
        'description': 'Chinese PP-OCRv3 detection',
    },
}


def download_with_progress(url: str, output_path: Path):
    def _progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        mb_downloaded = (count * block_size) / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f'\r  {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)', end='')

    urllib.request.urlretrieve(url, output_path, _progress_hook)
    print()


def extract_tar(tar_path: Path, output_dir: Path):
    logger.info(f"Extracting {tar_path.name}...")

    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(output_dir.parent)

    tar_path.unlink()
    logger.info("Done")


def download_model(model_key: str):
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}")

    model_info = MODELS[model_key]
    output_dir = Path(model_info['output_dir'])

    if output_dir.exists():
        logger.info(f"Already have: {output_dir}")
        return

    logger.info(f"Downloading {model_info['description']}...")

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    tar_path = output_dir.parent / f'{model_key}.tar'
    download_with_progress(model_info['url'], tar_path)

    extract_tar(tar_path, output_dir)

    logger.info(f"Saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Download pretrained PaddleOCR models')
    parser.add_argument('--models', type=str, default='all',
                        help=f'Which model(s) to download. Options: all, {", ".join(MODELS.keys())}')

    args = parser.parse_args()

    if args.models == 'all':
        models_to_download = list(MODELS.keys())
    else:
        models_to_download = [args.models]

    logger.info(f"Downloading {len(models_to_download)} models\n")

    for model_key in models_to_download:
        try:
            download_model(model_key)
            print()
        except Exception as e:
            logger.error(f"Failed: {model_key} - {e}")

    logger.info("\nDone!")


if __name__ == '__main__':
    main()
