import argparse
import subprocess
import sys
from pathlib import Path


def run_inference(image_path, model_dir, char_dict=None):
    cmd = [
        sys.executable,
        'PaddleOCR/tools/infer/predict_rec.py',
        f'--image_dir={image_path}',
        f'--rec_model_dir={model_dir}',
        '--use_gpu=True'
    ]

    if char_dict:
        cmd.append(f'--rec_char_dict_path={char_dict}')

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=30
        )

        output = result.stdout + result.stderr

        # Parse prediction from output
        for line in output.split('\n'):
            if 'Predicts of' in line and ':(' in line:
                tuple_part = line.split(':(', 1)[1]
                try:
                    import ast
                    result_tuple = ast.literal_eval('(' + tuple_part)
                    if isinstance(result_tuple, tuple) and len(result_tuple) > 0:
                        return result_tuple[0]
                except:
                    pass

        return ''
    except Exception as e:
        return f'ERROR: {e}'


def main():
    parser = argparse.ArgumentParser(description='Compare all models on a single image')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to image file')

    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    if not Path('PaddleOCR/tools/infer/predict_rec.py').exists():
        print("PaddleOCR not found!")
        sys.exit(1)

    print(f"\nTesting image: {image_path.name}\n")

    # Test EN PP-OCRv3
    print("EN PP-OCRv3:")
    en_pred = run_inference(
        image_path,
        'PaddleOCR/en_PP-OCRv3_rec_infer',
        'PaddleOCR/ppocr/utils/en_dict.txt'
    )
    print(f"  {en_pred}\n")

    # Test PP-OCRv5 Latin
    print("PP-OCRv5 Latin:")
    latin_pred = run_inference(
        image_path,
        'PaddleOCR/latin_ppocrv5_infer',
        'config/latin_v5_charset.txt'
    )
    print(f"  {latin_pred}\n")

    # Test Romanian PP-OCRv3
    print("Romanian PP-OCRv3:")
    romanian_pred = run_inference(
        image_path,
        'models/romanian_ppocrv3_infer',
        'config/romanian_charset.txt'
    )
    print(f"  {romanian_pred}\n")


if __name__ == '__main__':
    main()
