"""
Clean corpus: normalize diacritics, remove unwanted characters.

- Normalizes old Romanian diacritics (ţ→ț, ş→ș)
- Removes emojis, foreign scripts, encoding errors
- Keeps only valid Romanian text characters

Usage:
    python tools/clean_corpus.py
    python tools/clean_corpus.py --input config/romanian_corpus.txt --output config/romanian_corpus_clean.txt
"""

import argparse
from pathlib import Path
from collections import Counter
import unicodedata
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_charset(charset_path: Path) -> set:
    """Load character set from file."""
    with open(charset_path, 'r', encoding='utf-8') as f:
        charset = set(line.strip() for line in f if line.strip())
    # Add space (critical for OCR)
    charset.add(' ')
    return charset


def normalize_diacritics(text: str) -> str:
    """Normalize old Romanian diacritics to modern forms."""
    # Old cedilla forms → modern comma-below forms
    replacements = {
        '\u0163': '\u021B',  # ţ → ț (t-cedilla → t-comma)
        '\u0162': '\u021A',  # Ţ → Ț
        '\u015F': '\u0219',  # ş → ș (s-cedilla → s-comma)
        '\u015E': '\u0218',  # Ş → Ș
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def clean_line(line: str, charset: set) -> tuple[str, Counter]:
    """Clean a single line and return cleaned text + removed char counts."""

    # Normalize diacritics first
    line = normalize_diacritics(line)

    # Track removed characters
    removed = Counter()

    # Filter characters
    cleaned_chars = []
    for char in line:
        if char in charset:
            cleaned_chars.append(char)
        else:
            removed[char] += 1

    return ''.join(cleaned_chars), removed


def clean_corpus(input_path: Path, output_path: Path, charset_path: Path):
    """Clean corpus file."""

    logger.info("=" * 70)
    logger.info("Corpus Cleaning")
    logger.info("=" * 70)
    logger.info(f"\nInput:   {input_path}")
    logger.info(f"Output:  {output_path}")
    logger.info(f"Charset: {charset_path}\n")

    # Load charset
    charset = load_charset(charset_path)
    logger.info(f"Loaded charset: {len(charset)} characters (including space)\n")

    # Read all input first (in case input == output)
    logger.info("Reading input corpus...")
    with open(input_path, 'r', encoding='utf-8') as f:
        input_lines = f.readlines()

    logger.info(f"Loaded {len(input_lines):,} lines\n")

    # Clean corpus
    logger.info("Cleaning corpus...")

    total_removed = Counter()
    lines_processed = 0
    lines_written = 0
    normalized_count = 0
    cleaned_lines = []

    for line in input_lines:
        lines_processed += 1

        if lines_processed % 10000 == 0:
            logger.info(f"  Processed {lines_processed:,} lines")

        original_line = line.strip()
        if not original_line:
            continue

        # Check if normalization needed
        if any(c in original_line for c in '\u0163\u0162\u015F\u015E'):
            normalized_count += 1

        # Clean line
        cleaned_line, removed = clean_line(original_line, charset)

        # Update removed counts
        total_removed.update(removed)

        # Store cleaned line
        if cleaned_line.strip():  # Only keep non-empty lines
            cleaned_lines.append(cleaned_line)
            lines_written += 1

    # Write output
    logger.info("\nWriting cleaned corpus...")
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for line in cleaned_lines:
            f_out.write(line + '\n')

    logger.info(f"\n{'=' * 70}")
    logger.info("Cleaning Complete")
    logger.info(f"{'=' * 70}")
    logger.info(f"Lines processed:        {lines_processed:,}")
    logger.info(f"Lines written:          {lines_written:,}")
    logger.info(f"Lines normalized:       {normalized_count:,}")
    logger.info(f"Unique chars removed:   {len(total_removed)}\n")

    # Show top removed characters
    if total_removed:
        logger.info("Top 20 removed characters:")
        for char, count in total_removed.most_common(20):
            code_point = f"U+{ord(char):04X}"
            # Get unicode name if possible
            try:
                name = unicodedata.name(char)
            except ValueError:
                name = "UNKNOWN"
            logger.info(f"  '{char}' ({code_point}) {name[:40]:<40} - {count:,}x")

    logger.info(f"\n{'=' * 70}")
    logger.info(f"Saved to: {output_path}")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Clean Romanian corpus')
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('config/romanian_corpus.txt'),
        help='Input corpus file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('config/romanian_corpus.txt'),
        help='Output cleaned corpus file (default: overwrites input)'
    )
    parser.add_argument(
        '--charset',
        type=Path,
        default=Path('config/romanian_charset.txt'),
        help='Character set file'
    )

    args = parser.parse_args()

    clean_corpus(args.input, args.output, args.charset)


if __name__ == '__main__':
    main()
