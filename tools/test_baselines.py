"""
Test baseline models and save metrics.

Usage:
    python tools/test_baselines.py
    python tools/test_baselines.py --sample 1000
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple
import logging
from collections import Counter
import sys
import os
import subprocess
import re
import tempfile

# Add PaddleOCR to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'PaddleOCR'))
os.environ['FLAGS_allocator_strategy'] = 'auto_growth'

import paddle
from ppocr.utils.utility import get_image_file_list
from ppocr.utils.logging import get_logger
from tools.infer.utility import draw_ocr_box_txt, get_rotate_crop_image
from tools.infer.predict_rec import TextRecognizer

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def calculate_metrics(predictions: List[Tuple[str, str, str]]) -> dict:
    total = len(predictions)
    exact_matches = 0
    total_chars = 0
    total_char_errors = 0
    total_words = 0
    total_word_errors = 0

    # Track diacritic-specific errors
    diacritic_map = {
        'ă': 'a', 'â': 'a', 'î': 'i', 'ș': 's', 'ț': 't',
        'Ă': 'A', 'Â': 'A', 'Î': 'I', 'Ș': 'S', 'Ț': 'T'
    }
    diacritic_errors = Counter()
    diacritic_total = Counter()

    for img_path, label, pred in predictions:
        # Exact match
        if label == pred:
            exact_matches += 1

        # Character-level metrics
        total_chars += len(label)
        char_errors = levenshtein_distance(label, pred)
        total_char_errors += char_errors

        # Word-level metrics
        label_words = label.split()
        pred_words = pred.split()
        total_words += len(label_words)
        total_word_errors += levenshtein_distance(label_words, pred_words)

        # Diacritic-specific tracking
        for i, char in enumerate(label):
            if char in diacritic_map:
                diacritic_total[char] += 1
                # Check if this diacritic was preserved in prediction
                if i < len(pred):
                    if pred[i] != char:
                        diacritic_errors[char] += 1
                else:
                    diacritic_errors[char] += 1

    # Calculate rates
    exact_accuracy = exact_matches / total if total > 0 else 0
    char_error_rate = total_char_errors / total_chars if total_chars > 0 else 0
    char_accuracy = 1 - char_error_rate
    word_error_rate = total_word_errors / total_words if total_words > 0 else 0
    word_accuracy = 1 - word_error_rate

    # Diacritic accuracy
    diacritic_accuracy = {}
    for char in diacritic_map.keys():
        if diacritic_total[char] > 0:
            errors = diacritic_errors[char]
            total_count = diacritic_total[char]
            accuracy = 1 - (errors / total_count)
            diacritic_accuracy[char] = {
                'accuracy': accuracy,
                'errors': errors,
                'total': total_count
            }

    return {
        'total_samples': total,
        'exact_matches': exact_matches,
        'exact_accuracy': exact_accuracy,
        'character_accuracy': char_accuracy,
        'character_error_rate': char_error_rate,
        'word_accuracy': word_accuracy,
        'word_error_rate': word_error_rate,
        'diacritic_accuracy': diacritic_accuracy
    }


def load_dataset(data_dir: Path, label_file: Path, sample_size: int = None) -> List[Tuple[Path, str]]:
    samples = []

    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t', 1)
            if len(parts) != 2:
                continue

            img_rel_path, text = parts
            img_path = data_dir / img_rel_path

            if img_path.exists():
                samples.append((img_path, text))

    # Sample if requested
    if sample_size and sample_size < len(samples):
        import random
        random.seed(42)
        samples = random.sample(samples, sample_size)

    return samples


def create_recognizer(model_dir: str, char_dict_path: str = None, use_space_char_val: bool = True, algorithm: str = 'SVTR_LCNet'):
    class Args:
        use_gpu = True
        use_xpu = False
        use_npu = False
        use_mlu = False
        use_gcu = False
        ir_optim = True
        use_tensorrt = False
        min_subgraph_size = 15
        precision = 'fp32'
        gpu_mem = 500
        gpu_id = 0
        rec_model_dir = model_dir
        rec_image_shape = "3, 48, 320"
        rec_batch_num = 6
        rec_char_dict_path = char_dict_path  # None will use model's embedded dict
        use_space_char = use_space_char_val
        rec_algorithm = algorithm
        rec_image_inverse = False
        max_text_length = 1000  # Latin model uses max_text_length=1000
        use_mp = False
        benchmark = False
        warmup = False
        use_onnx = False
        enable_mkldnn = False
        cpu_threads = 10
        return_word_box = False
        use_pdserving = False
        show_log = False

    return TextRecognizer(Args())


def run_paddleocr_cli(img_path: Path, model_dir: str) -> str:
    try:
        cmd = [
            sys.executable,
            'PaddleOCR/tools/infer/predict_rec.py',
            f'--image_dir={img_path}',
            f'--rec_model_dir={model_dir}',
            '--use_gpu=True'
        ]

        # Set UTF-8 encoding for subprocess to fix Windows charmap issues
        # Force GPU usage
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['CUDA_VISIBLE_DEVICES'] = '0'
        env['FLAGS_use_cuda'] = '1'

        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30, env=env)

        # Parse output - look for: "Predicts of ...:('text', confidence)"
        output_text = result.stdout + result.stderr

        # Try to find in normal log lines
        for line in output_text.split('\n'):
            if 'Predicts of' in line and ':(' in line:
                tuple_part = line.split(':(', 1)[1]
                try:
                    import ast
                    result_tuple = ast.literal_eval('(' + tuple_part)
                    if isinstance(result_tuple, tuple) and len(result_tuple) > 0:
                        return result_tuple[0]
                except:
                    pass

        # If not found, check for logging error message format
        for line in output_text.split('\n'):
            if line.startswith('Message:') and 'Predicts of' in line and ':(' in line:
                if '"' in line or "'" in line:
                    msg_part = line.split('Message:', 1)[1].strip().strip('"').strip("'")
                    if ':(' in msg_part:
                        tuple_part = msg_part.split(':(', 1)[1]
                        try:
                            import ast
                            result_tuple = ast.literal_eval('(' + tuple_part)
                            if isinstance(result_tuple, tuple) and len(result_tuple) > 0:
                                return result_tuple[0]
                        except:
                            pass

        return ''
    except:
        return ''


def test_model(model_name: str, recognizer, dataset: List[Tuple[Path, str]], output_file: Path, use_cli: bool = False):
    logger.info(f"\nTesting {model_name}")
    logger.info(f"Testing {len(dataset)} images\n")

    predictions = []

    for idx, (img_path, label) in enumerate(dataset):
        if (idx + 1) % 100 == 0:
            logger.info(f"  Processed {idx + 1}/{len(dataset)} images")

        # Run recognition and extract text
        pred_text = ''

        if use_cli:
            # Use PaddleOCR CLI (recognizer is model_dir string)
            pred_text = run_paddleocr_cli(img_path, recognizer)
        else:
            # Use direct API
            try:
                import cv2
                img = cv2.imread(str(img_path))
                if img is None:
                    logger.warning(f"  Skipped {img_path.name} (couldn't read)")
                    predictions.append((str(img_path), label, ''))
                    continue

                rec_res, _ = recognizer([img])

                # Extract text
                if rec_res and len(rec_res) > 0:
                    result = rec_res[0]
                    # Handle both tuple and list formats
                    if isinstance(result, (tuple, list)) and len(result) > 0:
                        pred_text = result[0]
                    elif isinstance(result, str):
                        pred_text = result
                    else:
                        pred_text = str(result)

            except Exception as e:
                if idx < 3:
                    logger.error(f"  ERROR [{img_path.name}]: {e}")
                pred_text = ''

        predictions.append((str(img_path), label, pred_text))

    logger.info(f"  Done testing\n")

    # Calculate metrics
    logger.info("Calculating metrics...")
    metrics = calculate_metrics(predictions)

    # Save predictions
    logger.info("Saving predictions...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for img_path, label, pred in predictions:
            f.write(json.dumps({
                'image': img_path,
                'label': label,
                'prediction': pred
            }, ensure_ascii=False) + '\n')

    # Save metrics
    metrics_file = output_file.with_suffix('.metrics.json')
    logger.info("Saving metrics...")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Display metrics
    logger.info(f"\n{model_name} Results")
    logger.info(f"Total samples:        {metrics['total_samples']:,}")
    logger.info(f"Exact matches:        {metrics['exact_matches']:,}")
    logger.info(f"Exact accuracy:       {metrics['exact_accuracy']:.2%}")
    logger.info(f"Character accuracy:   {metrics['character_accuracy']:.2%}")
    logger.info(f"Character error rate: {metrics['character_error_rate']:.2%}")
    logger.info(f"Word accuracy:        {metrics['word_accuracy']:.2%}")
    logger.info(f"Word error rate:      {metrics['word_error_rate']:.2%}\n")

    # Diacritic accuracy
    if metrics['diacritic_accuracy']:
        logger.info("Diacritic-specific accuracy:")
        for char, stats in sorted(metrics['diacritic_accuracy'].items()):
            logger.info(f"  {char}: {stats['accuracy']:.2%} "
                       f"({stats['total'] - stats['errors']}/{stats['total']})")
        logger.info("")

    return metrics


def calculate_metrics_from_file(prediction_file: Path) -> dict:
    predictions = []

    with open(prediction_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            predictions.append((data['image'], data['label'], data['prediction']))

    return calculate_metrics(predictions)


def main():
    parser = argparse.ArgumentParser(description='Test baseline OCR models')
    parser.add_argument('--data-dir', type=Path,
                       default=Path('data/document_pages'),
                       help='Data directory')
    parser.add_argument('--output-dir', type=Path,
                       default=Path('baselines'),
                       help='Output directory for predictions')
    parser.add_argument('--sample', type=int, default=None,
                       help='Sample N images from each dataset (for quick testing)')
    parser.add_argument('--model', type=str, default='both',
                       choices=['v3', 'v5', 'both'],
                       help='Which model to test (v3=EN PP-OCRv3, v5=PP-OCRv5 Latin, both=both models)')
    parser.add_argument('--calc-metrics-only', action='store_true',
                       help='Only calculate metrics from existing prediction files')

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(exist_ok=True)

    logger.info("Baseline Model Testing")
    logger.info(f"\nData directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    if args.sample:
        logger.info(f"Sample size: {args.sample} images per dataset")
    logger.info("")

    # If only calculating metrics from existing files
    if args.calc_metrics_only:
        logger.info("Recalculating metrics...\n")

        for pred_file in args.output_dir.glob('*.jsonl'):
            logger.info(f"Processing {pred_file.name}...")
            metrics = calculate_metrics_from_file(pred_file)

            # Save metrics
            metrics_file = pred_file.with_suffix('.metrics.json')
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)

            # Display metrics
            logger.info(f"Total samples:        {metrics['total_samples']:,}")
            logger.info(f"Exact accuracy:       {metrics['exact_accuracy']:.2%}")
            logger.info(f"Character accuracy:   {metrics['character_accuracy']:.2%}")
            logger.info(f"Saved to {metrics_file.name}\n")

        return

    # Load datasets
    logger.info("Loading datasets...")
    train_data = load_dataset(
        args.data_dir / 'train_lines',
        args.data_dir / 'train_lines' / 'train.txt',
        args.sample
    )
    val_data = load_dataset(
        args.data_dir / 'val_lines',
        args.data_dir / 'val_lines' / 'val.txt',
        args.sample
    )

    logger.info(f"  Training set: {len(train_data):,} images")
    logger.info(f"  Validation set: {len(val_data):,} images\n")

    # Test EN PP-OCRv3
    if args.model in ['v3', 'both']:
        logger.info("Loading EN PP-OCRv3...")
        try:
            en_v3_recognizer = create_recognizer(
                model_dir='PaddleOCR/en_PP-OCRv3_rec_infer',
                char_dict_path='PaddleOCR/ppocr/utils/en_dict.txt',
                use_space_char_val=True
            )

            # Test on training set
            test_model(
                'EN PP-OCRv3 (Train)',
                en_v3_recognizer,
                train_data,
                args.output_dir / 'en_ppocrv3_train_predictions.jsonl'
            )

            # Test on validation set
            test_model(
                'EN PP-OCRv3 (Val)',
                en_v3_recognizer,
                val_data,
                args.output_dir / 'en_ppocrv3_val_predictions.jsonl'
            )

        except Exception as e:
            logger.error(f"Error with EN PP-OCRv3: {e}")
            import traceback
            traceback.print_exc()
            logger.info("Skipping EN PP-OCRv3\n")

    # Test PP-OCRv5 Latin
    if args.model in ['v5', 'both']:
        logger.info("Loading PP-OCRv5 Latin...")
        try:
            latin_v5_recognizer = create_recognizer(
                model_dir='PaddleOCR/latin_ppocrv5_infer',
                char_dict_path='config/latin_v5_charset.txt',
                use_space_char_val=True,
                algorithm='SVTR_LCNet'
            )

            # Test on training set
            test_model(
                'PP-OCRv5 Latin (Train)',
                latin_v5_recognizer,
                train_data,
                args.output_dir / 'ppocrv5_latin_train_predictions.jsonl'
            )

            # Test on validation set
            test_model(
                'PP-OCRv5 Latin (Val)',
                latin_v5_recognizer,
                val_data,
                args.output_dir / 'ppocrv5_latin_val_predictions.jsonl'
            )

        except Exception as e:
            logger.error(f"Error with PP-OCRv5 Latin: {e}")
            import traceback
            traceback.print_exc()
            logger.info("Skipping PP-OCRv5 Latin\n")

    logger.info("\nDone!")
    logger.info(f"Results in: {args.output_dir}\n")


if __name__ == '__main__':
    main()
