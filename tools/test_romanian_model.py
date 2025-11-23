"""
Test Romanian OCR model on validation data and save to baselines.

Usage:
    python tools/test_romanian_model.py
    python tools/test_romanian_model.py --sample 100
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple
import logging
import sys
import os

# Add PaddleOCR to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'PaddleOCR'))
os.environ['FLAGS_allocator_strategy'] = 'auto_growth'

from tools.infer.predict_rec import TextRecognizer

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Import metrics calculation from test_baselines
sys.path.insert(0, str(Path(__file__).parent))
from test_baselines import calculate_metrics, load_dataset


def create_romanian_recognizer(model_dir: str, char_dict_path: str, use_gpu: bool = True):
    # Capture parameters in outer scope for class definition
    _use_gpu = use_gpu
    _model_dir = model_dir
    _char_dict_path = char_dict_path

    class Args:
        use_gpu = _use_gpu
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
        rec_model_dir = _model_dir
        rec_image_shape = "3, 48, 640"  # Romanian model uses 640 width
        rec_batch_num = 6
        rec_char_dict_path = _char_dict_path
        use_space_char = True
        rec_algorithm = 'SVTR_LCNet'
        rec_image_inverse = False
        max_text_length = 75
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


def test_romanian_model(recognizer, dataset: List[Tuple[Path, str]], output_file: Path, model_name: str):
    logger.info(f"\nTesting {model_name}")
    logger.info(f"Testing {len(dataset)} images\n")

    predictions = []

    # Open output file for writing immediately
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, (img_path, label) in enumerate(dataset):
            if (idx + 1) % 10 == 0:
                logger.info(f"  Processed {idx + 1}/{len(dataset)} images")

            try:
                import cv2
                img = cv2.imread(str(img_path))
                if img is None:
                    logger.warning(f"  Skipped {img_path.name} (couldn't read)")
                    pred_text = ''
                else:
                    rec_res, _ = recognizer([img])

                    # Extract text
                    pred_text = ''
                    if rec_res and len(rec_res) > 0:
                        result = rec_res[0]
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

            # Write immediately
            f.write(json.dumps({
                'image': str(img_path),
                'label': label,
                'prediction': pred_text
            }, ensure_ascii=False) + '\n')
            f.flush()  # Flush to disk immediately

            predictions.append((str(img_path), label, pred_text))

    logger.info(f"  Done testing\n")

    # Calculate metrics
    logger.info("Calculating metrics...")
    metrics = calculate_metrics(predictions)

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


def main():
    parser = argparse.ArgumentParser(description='Test Romanian OCR model')
    parser.add_argument('--data-dir', type=Path,
                       default=Path('data/document_pages'),
                       help='Data directory')
    parser.add_argument('--model-dir', type=Path,
                       default=Path('baselines/romanian_ppocrv3_infer_final'),
                       help='Romanian model directory (inference format)')
    parser.add_argument('--char-dict', type=Path,
                       default=Path('config/romanian_charset.txt'),
                       help='Character dictionary path')
    parser.add_argument('--output-dir', type=Path,
                       default=Path('baselines'),
                       help='Output directory for predictions')
    parser.add_argument('--sample', type=int, default=None,
                       help='Sample N images (for quick testing)')

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(exist_ok=True)

    logger.info("Romanian Model Testing (GPU)")
    logger.info(f"\nModel: {args.model_dir}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    if args.sample:
        logger.info(f"Sample size: {args.sample} images")
    logger.info("")

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

    # Initialize Romanian model (GPU)
    logger.info("Loading Romanian model (GPU)...")
    try:
        ro_recognizer = create_romanian_recognizer(
            model_dir=str(args.model_dir),
            char_dict_path=str(args.char_dict),
            use_gpu=True)

        # Test on training set
        test_romanian_model(
            ro_recognizer,
            train_data,
            args.output_dir / 'romanian_ppocrv3_final_train_predictions.jsonl',
            'Romanian PP-OCRv3 Final (Train)'
        )

        # Test on validation set
        test_romanian_model(
            ro_recognizer,
            val_data,
            args.output_dir / 'romanian_ppocrv3_final_val_predictions.jsonl',
            'Romanian PP-OCRv3 Final (Val)'
        )

    except Exception as e:
        logger.error(f"Error with Romanian model: {e}")
        import traceback
        traceback.print_exc()

    logger.info("\nDone!")
    logger.info(f"Results in: {args.output_dir}\n")


if __name__ == '__main__':
    main()
