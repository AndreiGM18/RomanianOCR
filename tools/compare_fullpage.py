import argparse
import sys
from pathlib import Path
import os
import cv2

# Add PaddleOCR to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'PaddleOCR'))
os.environ['FLAGS_allocator_strategy'] = 'auto_growth'

from tools.infer.predict_det import TextDetector
from tools.infer.predict_rec import TextRecognizer
from tools.infer.utility import get_rotate_crop_image


def create_detector():
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
        det_algorithm = 'DB'
        det_model_dir = 'baselines/ch_PP-OCRv3_det'
        det_limit_side_len = 960
        det_limit_type = 'max'
        det_box_type = 'quad'
        det_db_thresh = 0.3
        det_db_box_thresh = 0.6
        det_db_unclip_ratio = 1.5
        max_batch_size = 10
        use_dilation = False
        det_db_score_mode = 'fast'
        det_east_score_thresh = 0.8
        det_east_cover_thresh = 0.1
        det_east_nms_thresh = 0.2
        det_sast_score_thresh = 0.5
        det_sast_nms_thresh = 0.2
        det_pse_thresh = 0
        det_pse_box_thresh = 0.85
        det_pse_min_area = 16
        det_pse_scale = 1
        scales = [8, 16, 32]
        alpha = 1.0
        beta = 1.0
        fourier_degree = 5
        use_mp = False
        benchmark = False
        warmup = False
        use_onnx = False
        enable_mkldnn = False
        cpu_threads = 10

    return TextDetector(Args())


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


def compare_on_fullpage(image_path):
    print(f"\nProcessing full page: {image_path.name}\n")

    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to read image: {image_path}")
        sys.exit(1)

    # Initialize detector
    print("Loading detector...")
    detector = create_detector()

    # Initialize recognizers
    print("Loading recognizers...")
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

    print("\nDetecting text regions...")
    dt_boxes, _ = detector(img)

    if dt_boxes is None or len(dt_boxes) == 0:
        print("No text detected!")
        return

    print(f"Found {len(dt_boxes)} text regions\n")
    print("Running recognition...\n")

    # Extract text with all models
    results_en = []
    results_latin = []
    results_ro_e10 = []
    results_ro_e50 = []

    for box in dt_boxes:
        # Crop text region
        img_crop = get_rotate_crop_image(img, box)

        # Run recognition with all models
        res_en, _ = rec_en([img_crop])
        res_latin, _ = rec_latin([img_crop])
        res_ro_e10, _ = rec_ro_e10([img_crop])
        res_ro_e50, _ = rec_ro_e50([img_crop])

        # Extract text
        def get_text(result):
            if result and len(result) > 0:
                r = result[0]
                if isinstance(r, (tuple, list)) and len(r) > 0:
                    return r[0]
                return str(r)
            return ''

        results_en.append(get_text(res_en))
        results_latin.append(get_text(res_latin))
        results_ro_e10.append(get_text(res_ro_e10))
        results_ro_e50.append(get_text(res_ro_e50))

    # Display comparison
    print("=" * 170)
    print(f"{'Line':<5} | {'EN PP-OCRv3':<30} | {'PP-OCRv5 Latin':<30} | {'Romanian E10':<30} | {'Romanian E50':<30}")
    print("=" * 170)

    for i in range(len(results_en)):
        en_text = results_en[i][:30]
        latin_text = results_latin[i][:30]
        ro_e10_text = results_ro_e10[i][:30]
        ro_e50_text = results_ro_e50[i][:30]
        print(f"{i+1:<5} | {en_text:<30} | {latin_text:<30} | {ro_e10_text:<30} | {ro_e50_text:<30}")

    print("=" * 170)
    print(f"\nTotal lines: {len(results_en)}")
    print()


def main():
    parser = argparse.ArgumentParser(description='Compare all models on a full page image')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to full page image')

    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    # Check required models
    required_models = [
        'baselines/ch_PP-OCRv3_det',
        'PaddleOCR/en_PP-OCRv3_rec_infer',
        'PaddleOCR/latin_ppocrv5_infer',
        'models/romanian_ppocrv3_epoch10_infer',
        'models/romanian_ppocrv3_epoch50_infer'
    ]

    missing = [m for m in required_models if not Path(m).exists()]
    if missing:
        print("Missing required models:")
        for m in missing:
            print(f"  - {m}")
        print("\nRun: python tools/download_pretrained.py")
        sys.exit(1)

    compare_on_fullpage(image_path)


if __name__ == '__main__':
    main()
