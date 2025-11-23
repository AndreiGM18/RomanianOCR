"""
Check if all characters in corpus are in the character dictionary.

Usage:
    python tools/check_charset.py
"""

from pathlib import Path
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def check_charset():
    corpus_path = Path('config/romanian_corpus.txt')
    charset_path = Path('config/romanian_charset.txt')

    logger.info("Character Set Validation")

    # Load charset
    with open(charset_path, 'r', encoding='utf-8') as f:
        charset = set(line.strip() for line in f if line.strip())

    logger.info(f"\nCharset size: {len(charset)} characters\n")

    # Find all unique characters in corpus
    corpus_chars = set()
    line_count = 0

    logger.info("Scanning corpus...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_count += 1
            corpus_chars.update(line.strip())

            if line_count % 10000 == 0:
                logger.info(f"  Scanned {line_count:,} lines, found {len(corpus_chars)} unique chars")

    logger.info(f"\nTotal lines scanned: {line_count:,}")
    logger.info(f"Unique characters in corpus: {len(corpus_chars)}\n")

    # Find missing characters
    missing_chars = corpus_chars - charset

    if missing_chars:
        logger.info("\nWARNING: Characters in corpus NOT in charset:")

        # Count occurrences
        char_counts = Counter()
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                for char in missing_chars:
                    char_counts[char] += line.count(char)

        # Sort by frequency
        sorted_missing = sorted(missing_chars, key=lambda c: char_counts[c], reverse=True)

        for char in sorted_missing:
            # Show char, unicode code point, and count
            code_point = f"U+{ord(char):04X}"
            count = char_counts[char]
            logger.info(f"  '{char}' ({code_point}) - {count:,} occurrences")

        logger.info(f"\nTotal missing: {len(missing_chars)} characters")
        logger.info("\nAction needed: Add these to config/romanian_charset.txt or clean corpus.")

    else:
        logger.info("\nSUCCESS: All corpus characters are in charset!")

    # Also show unused charset characters
    unused_chars = charset - corpus_chars
    if unused_chars:
        logger.info(f"\nNote: {len(unused_chars)} charset characters not used in corpus")


if __name__ == '__main__':
    check_charset()
