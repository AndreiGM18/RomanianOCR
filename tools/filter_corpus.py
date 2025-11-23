"""
Filter Romanian corpus to only lines containing diacritics.

Reads compressed .txt.xz file, extracts N lines with Romanian diacritics,
and saves to config/romanian_corpus.txt for training data generation.

Usage:
    python tools/filter_corpus.py
    python tools/filter_corpus.py --max-lines 100000
    python tools/filter_corpus.py --input path/to/corpus.txt.xz --max-lines 200000
"""

import lzma
import argparse
from pathlib import Path
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def filter_corpus(input_path: Path, output_path: Path, max_lines: int):
    logger.info("Romanian Corpus Filter")
    logger.info(f"\nInput:  {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Target: {max_lines:,} lines\n")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Romanian diacritics (main 5)
    main_diacritics = 'ăâîșț'
    all_diacritics = 'ăâîșțĂÂÎȘȚ'

    logger.info("Extracting lines with diacritics...")

    output_path.parent.mkdir(exist_ok=True)
    written = 0
    scanned = 0
    diacritic_counts = Counter()

    with lzma.open(input_path, 'rt', encoding='utf-8') as f_in:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                scanned += 1

                if scanned % 100000 == 0:
                    logger.info(f"  Scanned {scanned:,} lines, extracted {written:,}")

                line = line.strip()

                # Skip empty or lines without diacritics
                if not line or not any(c in line for c in all_diacritics):
                    continue

                # Write line
                f_out.write(line + '\n')
                written += 1

                # Count diacritics for stats
                for char in main_diacritics:
                    if char in line.lower():
                        diacritic_counts[char] += 1

                # Stop when we reach target
                if written >= max_lines:
                    logger.info(f"  Reached target of {max_lines:,} lines")
                    break

    logger.info("\nExtraction Complete")
    logger.info(f"Scanned:   {scanned:,} lines")
    logger.info(f"Extracted: {written:,} lines\n")

    # Show diacritic distribution
    logger.info("Diacritic distribution:")
    total_diacritics = sum(diacritic_counts.values())
    for char in main_diacritics:
        count = diacritic_counts[char]
        percentage = (count / total_diacritics * 100) if total_diacritics > 0 else 0
        logger.info(f"  {char}: {count:,} ({percentage:.1f}%)")

    logger.info(f"\nSaved to: {output_path}")
    logger.info(f"Done!")


def main():
    parser = argparse.ArgumentParser(description='Filter Romanian corpus for diacritics')
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('C:/Users/andre/Downloads/ro.txt.xz'),
        help='Input corpus file (.txt.xz)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('config/romanian_corpus.txt'),
        help='Output filtered corpus file'
    )
    parser.add_argument(
        '--max-lines',
        type=int,
        default=100000,
        help='Maximum number of lines to extract (default: 100000)'
    )

    args = parser.parse_args()

    filter_corpus(args.input, args.output, args.max_lines)


if __name__ == '__main__':
    main()
