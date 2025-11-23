"""
Interactive font validation tool for Romanian diacritics.

Shows visual preview of each font. Press keys to approve/reject:
- Y or ENTER: Approve font
- N or SPACE: Reject font
- Q or ESC: Quit and save

Usage:
    python tools/validate_fonts.py
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Interactive backend
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def find_all_fonts():
    import os

    search_dirs = [
        Path('/usr/share/fonts'),
        Path.home() / '.fonts',
        Path('C:/Windows/Fonts'),
        Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts',
    ]

    all_fonts = []
    for font_dir in search_dirs:
        if font_dir.exists():
            all_fonts.extend(font_dir.rglob('*.ttf'))
            all_fonts.extend(font_dir.rglob('*.TTF'))

    # Remove duplicates (keep first occurrence)
    seen = set()
    unique_fonts = []
    for font in all_fonts:
        font_key = (font.name.lower(), font.stat().st_size)
        if font_key not in seen:
            seen.add(font_key)
            unique_fonts.append(font)

    return sorted(unique_fonts, key=lambda x: x.name)


def render_font_sample(font_path: Path):
    try:
        font = ImageFont.truetype(str(font_path), 28)
        font_small = ImageFont.truetype(str(font_path), 18)

        # Create sample image
        img = Image.new('RGB', (800, 300), color='white')
        draw = ImageDraw.Draw(img)

        # Draw font name
        draw.text((20, 15), f"Font: {font_path.name}", fill='black')

        # Draw Romanian text samples
        samples = [
            ("Diacritics:", "ă  â  î  ș  ț    Ă  Â  Î  Ș  Ț"),
            ("Words:", "română  București  țară  întrebare"),
            ("Sentence:", "Această întrebare foarte importantă"),
        ]

        y = 70
        for label, text in samples:
            draw.text((20, y), label, fill='gray', font=font_small)
            draw.text((20, y + 25), text, fill='black', font=font)
            y += 75

        return img

    except Exception as e:
        logger.error(f"Failed to render {font_path.name}: {e}")
        return None


class FontValidator:
    def __init__(self):
        self.fonts = find_all_fonts()
        self.validated_fonts = []
        self.current_idx = 0
        self.fig = None
        self.ax = None
        self.quit_requested = False
        self.processing = False  # Prevent duplicate key events

    def on_key(self, event):
        if self.processing:
            return

        key = event.key.lower() if event.key else ''

        if key in ['y', 'enter']:
            # Approve
            self.processing = True
            font_path = self.fonts[self.current_idx]
            self.validated_fonts.append(str(font_path))
            logger.info(f"  APPROVED: {font_path.name}")
            self.current_idx += 1
            self.show_next()
            self.processing = False

        elif key in ['n', ' ', 'space']:
            # Reject
            self.processing = True
            logger.info(f"  REJECTED: {self.fonts[self.current_idx].name}")
            self.current_idx += 1
            self.show_next()
            self.processing = False

        elif key in ['q', 'escape']:
            # Quit
            self.quit_requested = True
            plt.close(self.fig)

    def show_next(self):
        if self.current_idx >= len(self.fonts):
            logger.info("\nAll fonts reviewed!")
            plt.close(self.fig)
            return

        if self.quit_requested:
            return

        font_path = self.fonts[self.current_idx]

        # Render sample
        img = render_font_sample(font_path)
        if img is None:
            logger.info(f"  SKIPPED (render failed): {font_path.name}")
            self.current_idx += 1
            self.show_next()
            return

        # Display
        self.ax.clear()
        self.ax.imshow(img)
        self.ax.axis('off')
        self.ax.set_title(
            f"Font {self.current_idx + 1}/{len(self.fonts)}\n\n"
            f"Y/Enter = Approve  |  N/Space = Reject  |  Q/Esc = Quit",
            fontsize=12, pad=20
        )
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def run(self):
        logger.info("Romanian Font Validation Tool")
        logger.info(f"\nFound {len(self.fonts)} fonts to validate\n")
        logger.info("Controls:")
        logger.info("  Y or ENTER = Approve font")
        logger.info("  N or SPACE = Reject font")
        logger.info("  Q or ESC   = Quit and save\n")

        # Setup matplotlib
        self.fig, self.ax = plt.subplots(figsize=(12, 5))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        plt.subplots_adjust(top=0.85)

        # Show first font
        self.show_next()

        # Start event loop
        plt.show()

        # Save results
        self.save_validated_fonts()

    def save_validated_fonts(self):
        if not self.validated_fonts:
            logger.info("\nNo fonts validated!")
            return

        output_file = Path('config/validated_fonts.txt')
        output_file.parent.mkdir(exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            for font_path in self.validated_fonts:
                f.write(font_path + '\n')

        logger.info(f"\nSaved {len(self.validated_fonts)} validated fonts!")
        logger.info(f"Output: {output_file}")
        logger.info("\nValidated fonts:")
        for font_path in self.validated_fonts:
            logger.info(f"  - {Path(font_path).name}")


def main():
    validator = FontValidator()
    validator.run()


if __name__ == '__main__':
    main()
