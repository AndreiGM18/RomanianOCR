"""
Review validated fonts - show all approved fonts in one view.

Press ESC or Q to close.

Usage:
    python tools/review_validated_fonts.py
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def render_font_sample(font_path: Path, compact=True):
    """Render compact Romanian diacritic sample."""
    try:
        font_size = 20 if compact else 28
        font = ImageFont.truetype(str(font_path), font_size)

        # Create sample image
        width = 400 if compact else 600
        height = 80 if compact else 120
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)

        # Draw font name (smaller)
        try:
            font_small = ImageFont.truetype(str(font_path), 10)
        except:
            font_small = None

        if font_small:
            draw.text((5, 3), font_path.name[:40], fill='gray', font=font_small)

        # Draw Romanian text
        text = "ă â î ș ț  România București"
        y_pos = 20 if compact else 35
        draw.text((5, y_pos), text, fill='black', font=font)

        return img

    except Exception as e:
        # Return error placeholder
        img = Image.new('RGB', (400, 80), color='#ffeeee')
        draw = ImageDraw.Draw(img)
        draw.text((5, 30), f"Error: {font_path.name}", fill='red')
        return img


def main():
    validated_fonts_file = Path('config/validated_fonts.txt')

    if not validated_fonts_file.exists():
        logger.error("No validated fonts found!")
        logger.error("Run: python tools/validate_fonts.py")
        return

    # Load validated fonts
    with open(validated_fonts_file, 'r', encoding='utf-8') as f:
        font_paths = [Path(line.strip()) for line in f if line.strip()]

    # Filter to existing fonts
    existing_fonts = [f for f in font_paths if f.exists()]

    if not existing_fonts:
        logger.error("None of the validated fonts exist!")
        return

    logger.info("=" * 70)
    logger.info("Validated Fonts Review")
    logger.info("=" * 70)
    logger.info(f"\nShowing {len(existing_fonts)} validated fonts\n")
    logger.info("Press ESC or Q to close window\n")

    # Determine grid layout
    fonts_per_page = 20
    num_pages = (len(existing_fonts) + fonts_per_page - 1) // fonts_per_page

    for page in range(num_pages):
        start_idx = page * fonts_per_page
        end_idx = min(start_idx + fonts_per_page, len(existing_fonts))
        page_fonts = existing_fonts[start_idx:end_idx]

        # Create figure
        fig = plt.figure(figsize=(14, 12))
        fig.suptitle(
            f'Validated Fonts - Page {page + 1}/{num_pages} '
            f'({len(existing_fonts)} total)\n\n'
            f'Press ESC or Q to close, Arrow keys to navigate',
            fontsize=14
        )

        # Create grid
        rows = (len(page_fonts) + 1) // 2
        cols = 2

        for idx, font_path in enumerate(page_fonts):
            # Render sample
            img = render_font_sample(font_path, compact=True)

            # Add to grid
            ax = plt.subplot(rows, cols, idx + 1)
            ax.imshow(img)
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    logger.info("\n" + "=" * 70)
    logger.info("Review complete!")
    logger.info("=" * 70)
    logger.info(f"\nValidated fonts: {len(existing_fonts)}")
    logger.info(f"Location: {validated_fonts_file}")


if __name__ == '__main__':
    main()
