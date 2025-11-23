"""
Generate synthetic training data for Romanian OCR.

Usage:
    python scripts/generate_training_data.py [--num-train 100] [--num-val 10]
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DocumentPageGenerator:
    def __init__(self, font_size_range=(12, 20), line_spacing_range=(1.2, 1.8)):
        self.font_size_range = font_size_range
        self.line_spacing_range = line_spacing_range
        self.fonts = self._find_fonts()

        if not self.fonts:
            raise ValueError("No .ttf fonts found! Check font directories.")

        logger.info(f"Loaded {len(self.fonts)} fonts")

    def _find_fonts(self) -> List[Path]:
        validated_fonts_file = Path('config/validated_fonts.txt')

        if not validated_fonts_file.exists():
            raise ValueError(
                "\nNo validated fonts found!\n"
                "Please run: python tools/validate_fonts.py\n"
                "This will let you visually approve fonts that render Romanian correctly."
            )

        # Read validated fonts
        with open(validated_fonts_file, 'r', encoding='utf-8') as f:
            font_paths = [Path(line.strip()) for line in f if line.strip()]

        # Filter to fonts that still exist
        existing_fonts = [f for f in font_paths if f.exists()]

        if not existing_fonts:
            raise ValueError(
                "\nNone of the validated fonts exist on this system!\n"
                "Please re-run: python tools/validate_fonts.py"
            )

        logger.info(f"Loaded {len(existing_fonts)} validated fonts")
        return existing_fonts

    def generate_page(
        self,
        texts: List[str],
        width=800,
        height=1200,
        margin=50
    ) -> Tuple[Image.Image, List[dict]]:
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)

        y = margin
        regions = []

        for text in texts:
            if y >= height - margin:
                break

            # Random font and size
            font_path = random.choice(self.fonts)
            font_size = random.randint(*self.font_size_range)

            try:
                font = ImageFont.truetype(str(font_path), font_size)
            except Exception:
                continue

            # Truncate text if it's too long to fit on page
            x = margin
            max_width = width - margin - x

            # Binary search to find how much text fits
            if len(text) > 10:  # Only truncate if text is reasonably long
                bbox = draw.textbbox((x, y), text, font=font)
                if bbox[2] > width - margin:
                    # Text too long, find truncation point
                    left, right = 0, len(text)
                    truncated_text = text

                    while left < right - 1:
                        mid = (left + right) // 2
                        test_text = text[:mid]
                        test_bbox = draw.textbbox((x, y), test_text, font=font)

                        if test_bbox[2] <= width - margin:
                            left = mid
                            truncated_text = test_text
                        else:
                            right = mid

                    text = truncated_text

            # Now draw the (possibly truncated) text
            draw.text((x, y), text, fill='black', font=font)

            # Get bounding box of what was actually drawn
            bbox = draw.textbbox((x, y), text, font=font)
            x1, y1, x2, y2 = bbox

            # Skip if text didn't render (bbox too small)
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue

            # Validate Romanian diacritics if present in text
            if any(c in text for c in 'ăâîșțĂÂÎȘȚ'):
                # Quick check: ensure bbox is reasonable for text length
                expected_width = len(text) * (font_size * 0.3)
                actual_width = x2 - x1
                # If width is way too small, font likely didn't render diacritics
                if actual_width < expected_width * 0.5:
                    continue

            # Store region
            regions.append({
                'transcription': text,
                'points': [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            })

            # Move to next line
            line_spacing = random.uniform(*self.line_spacing_range)
            y = y2 + int(font_size * (line_spacing - 1))

        return img, regions

    def generate_dataset(
        self,
        corpus_file: Path,
        output_dir: Path,
        num_pages: int,
        split: str = 'train'
    ):
        # Load corpus and filter for lines with Romanian diacritics
        with open(corpus_file, 'r', encoding='utf-8') as f:
            all_texts = [line.strip() for line in f if line.strip()]

        logger.info(f"Loaded {len(all_texts):,} lines from corpus")

        romanian_diacritics = 'ăâîșțĂÂÎȘȚ'
        texts_with_diacritics = [
            text for text in all_texts
            if any(c in text for c in romanian_diacritics)
        ]

        logger.info(f"Found {len(texts_with_diacritics):,} lines with diacritics ({len(texts_with_diacritics)/len(all_texts)*100:.1f}%)")

        if not texts_with_diacritics:
            raise ValueError("No lines with Romanian diacritics found in corpus!")

        # Use only texts with diacritics
        all_texts = texts_with_diacritics

        # Create output directories
        split_dir = output_dir / split
        img_dir = split_dir / 'images'
        img_dir.mkdir(parents=True, exist_ok=True)

        # Generate pages without duplicate lines
        annotations = []
        available_texts = all_texts.copy()
        random.shuffle(available_texts)

        for page_idx in range(num_pages):
            # Sample random texts for this page (no duplicates)
            num_lines = random.randint(20, 40)

            # Check if enough texts available
            if len(available_texts) < num_lines:
                logger.warning(f"Not enough unique texts for page {page_idx}. "
                             f"Need {num_lines}, have {len(available_texts)}. "
                             f"Reshuffling corpus...")
                available_texts = all_texts.copy()
                random.shuffle(available_texts)

            # Take texts from available pool
            page_texts = available_texts[:num_lines]
            available_texts = available_texts[num_lines:]

            # Generate page image
            img, regions = self.generate_page(page_texts)

            # Save image
            img_path = img_dir / f'{split}_{page_idx:06d}.jpg'
            img.save(img_path, 'JPEG', quality=95)

            # Store annotation
            annotations.append({
                'img_path': f'images/{split}_{page_idx:06d}.jpg',
                'regions': regions
            })

            if (page_idx + 1) % 10 == 0:
                logger.info(f"Generated {page_idx + 1}/{num_pages} pages")

        # Save annotations
        ann_file = split_dir / f'{split}.txt'
        with open(ann_file, 'w', encoding='utf-8') as f:
            for ann in annotations:
                f.write(json.dumps(ann, ensure_ascii=False) + '\n')

        logger.info(f"Saved {num_pages} pages to {split_dir}")
        return annotations


class LineImageExtractor:
    def extract_lines(
        self,
        input_dir: Path,
        output_dir: Path,
        split: str = 'train'
    ):
        split_in = input_dir / split
        split_out = output_dir / f'{split}_lines'
        img_out_dir = split_out / 'images'
        img_out_dir.mkdir(parents=True, exist_ok=True)

        # Read annotations
        ann_file = split_in / f'{split}.txt'
        with open(ann_file, 'r', encoding='utf-8') as f:
            annotations = [json.loads(line) for line in f]

        # Extract lines
        line_labels = []
        line_count = 0

        for ann in annotations:
            img_path = split_in / ann['img_path']
            img = Image.open(img_path)

            for region in ann['regions']:
                points = region['points']
                transcription = region['transcription']

                # Get bounding box
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                x1, y1 = min(x_coords), min(y_coords)
                x2, y2 = max(x_coords), max(y_coords)

                # Crop line image
                line_img = img.crop((x1, y1, x2, y2))

                # Save line image
                line_img_path = img_out_dir / f'{split}_{line_count:06d}.jpg'
                line_img.save(line_img_path, 'JPEG', quality=95)

                # Store label
                line_labels.append(f'images/{split}_{line_count:06d}.jpg\t{transcription}')
                line_count += 1

        # Save labels
        label_file = split_out / f'{split}.txt'
        with open(label_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(line_labels))

        logger.info(f"Created {line_count} line images in {split_out}")
        return line_count


def main():
    parser = argparse.ArgumentParser(description='Generate Romanian OCR training data')
    parser.add_argument('--corpus', type=str, default='config/romanian_corpus.txt',
                        help='Path to Romanian text corpus')
    parser.add_argument('--output', type=str, default='data/document_pages',
                        help='Output directory')
    parser.add_argument('--num-train', type=int, default=100,
                        help='Number of training pages')
    parser.add_argument('--num-val', type=int, default=10,
                        help='Number of validation pages')
    parser.add_argument('--regenerate', action='store_true',
                        help='Delete existing data and regenerate')

    args = parser.parse_args()

    corpus_file = Path(args.corpus)
    output_dir = Path(args.output)

    if not corpus_file.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_file}")

    # Delete existing data if regenerating
    if args.regenerate and output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
        logger.info("Deleted existing data")

    # Generate pages
    generator = DocumentPageGenerator()

    logger.info(f"Generating {args.num_train} training pages...")
    generator.generate_dataset(corpus_file, output_dir, args.num_train, 'train')

    logger.info(f"Generating {args.num_val} validation pages...")
    generator.generate_dataset(corpus_file, output_dir, args.num_val, 'val')

    # Extract line images
    extractor = LineImageExtractor()

    logger.info("Extracting training line images...")
    train_lines = extractor.extract_lines(output_dir, output_dir, 'train')

    logger.info("Extracting validation line images...")
    val_lines = extractor.extract_lines(output_dir, output_dir, 'val')

    logger.info(f"\nDone:")
    logger.info(f"  Train: {train_lines}")
    logger.info(f"  Val: {val_lines}")
    logger.info(f"  Output: {output_dir.absolute()}")


if __name__ == '__main__':
    main()
