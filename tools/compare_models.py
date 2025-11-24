import argparse
import sys
from pathlib import Path
import os
import cv2

# Add PaddleOCR to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'PaddleOCR'))
os.environ['FLAGS_allocator_strategy'] = 'auto_growth'

from tools.infer.predict_rec import TextRecognizer


def create_recognizer(model_dir, char_dict_path, max_text_len=75):
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
        rec_char_dict_path = char_dict_path
        use_space_char = True
        rec_algorithm = 'SVTR_LCNet'
        rec_image_inverse = False
        max_text_length = max_text_len
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


def main():
    parser = argparse.ArgumentParser(description='Compare all models on a single line image')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to line image file')

    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    print(f"\nTesting image: {image_path.name}")
    print("Note: This tool expects LINE images (cropped text lines), not full pages.\n")

    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to read image: {image_path}")
        sys.exit(1)

    # Load models
    print("Loading models...")

    print("  EN PP-OCRv3...")
    rec_en = create_recognizer(
        'PaddleOCR/en_PP-OCRv3_rec_infer',
        'PaddleOCR/ppocr/utils/en_dict.txt'
    )

    print("  PP-OCRv5 Latin...")
    rec_latin = create_recognizer(
        'PaddleOCR/latin_ppocrv5_infer',
        'config/latin_v5_charset.txt',
        max_text_len=1000
    )

    print("  Romanian PP-OCRv3 (Epoch 10)...")
    rec_ro_e10 = create_recognizer(
        'models/romanian_ppocrv3_epoch10_infer',
        'config/romanian_charset.txt'
    )

    print("  Romanian PP-OCRv3 (Epoch 50)...")
    rec_ro_e50 = create_recognizer(
        'models/romanian_ppocrv3_epoch50_infer',
        'config/romanian_charset.txt'
    )

    print("\nRunning recognition...\n")

    # Run recognition
    res_en, _ = rec_en([img])
    res_latin, _ = rec_latin([img])
    res_ro_e10, _ = rec_ro_e10([img])
    res_ro_e50, _ = rec_ro_e50([img])

    # Extract text
    def get_text(result):
        if result and len(result) > 0:
            r = result[0]
            if isinstance(r, (tuple, list)) and len(r) > 0:
                return r[0]
            return str(r)
        return '(no result)'

    text_en = get_text(res_en)
    text_latin = get_text(res_latin)
    text_ro_e10 = get_text(res_ro_e10)
    text_ro_e50 = get_text(res_ro_e50)

    # Display results
    print("EN PP-OCRv3:")
    print(f"  {text_en}\n")

    print("PP-OCRv5 Latin:")
    print(f"  {text_latin}\n")

    print("Romanian PP-OCRv3 (Epoch 10):")
    print(f"  {text_ro_e10}\n")

    print("Romanian PP-OCRv3 (Epoch 50):")
    print(f"  {text_ro_e50}\n")


if __name__ == '__main__':
    main()
