"""NLE (NetHack Learning Environment) TTY rendering utilities.

This module provides proper ASCII rendering for NetHack and MiniHack environments
using a pre-rendered texture atlas approach for efficient rendering.

Based on the rendering approach from BALROG:
https://github.com/balrog-ai/BALROG
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# NetHack ANSI color palette (16 colors)
NLE_COLORS = [
    "#000000",  # 0: Black
    "#800000",  # 1: Dark Red
    "#008000",  # 2: Dark Green
    "#808000",  # 3: Brown/Olive
    "#000080",  # 4: Dark Blue
    "#800080",  # 5: Dark Magenta
    "#008080",  # 6: Dark Cyan
    "#808080",  # 7: Gray (flipped with 8)
    "#C0C0C0",  # 8: Light Gray (flipped with 7)
    "#FF0000",  # 9: Bright Red
    "#00FF00",  # 10: Bright Green
    "#FFFF00",  # 11: Bright Yellow
    "#0000FF",  # 12: Bright Blue
    "#FF00FF",  # 13: Bright Magenta
    "#00FFFF",  # 14: Bright Cyan
    "#FFFFFF",  # 15: White
]


def _get_monospace_font(size: int = 12) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a monospace font, with fallback to default."""
    font_candidates = [
        "DejaVuSansMono.ttf",
        "LiberationMono-Regular.ttf",
        "UbuntuMono-R.ttf",
        "Consolas.ttf",
        "Courier New.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
    ]
    for font_path in font_candidates:
        try:
            return ImageFont.truetype(font_path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _create_texture_atlas(font_size: int = 12) -> tuple[np.ndarray, int, int]:
    """Create a texture atlas with all ASCII characters in all 16 colors.

    Returns:
        Tuple of (atlas array, cell_width, cell_height)
        Atlas shape: (4096, cell_height, cell_width, 3)
        where 4096 = 16 colors × 256 characters
    """
    font = _get_monospace_font(font_size)

    # Measure character cell size
    dummy_img = Image.new("RGB", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)

    # Find max character dimensions
    cell_width = 0
    cell_height = 0
    for i in range(256):
        try:
            bbox = dummy_draw.textbbox((0, 0), chr(i), font=font)
            cell_width = max(cell_width, bbox[2] - bbox[0])
            cell_height = max(cell_height, bbox[3] - bbox[1])
        except Exception:
            pass

    # Ensure minimum cell size
    cell_width = max(cell_width, 8)
    cell_height = max(cell_height, 14)

    # Create atlas image: 64x64 grid (16 colors × 4 rows, 16 chars × 4 cols per color block)
    img_width = cell_width * 64
    img_height = cell_height * 64
    img = Image.new("RGB", (img_width, img_height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    for color_idx, color in enumerate(NLE_COLORS):
        # Each color block is 16x16 characters
        x_offset = (color_idx % 4) * (cell_width * 16)
        y_offset = (color_idx // 4) * (cell_height * 16)

        for char_code in range(256):
            x = (char_code % 16) * cell_width + x_offset
            y = (char_code // 16) * cell_height + y_offset

            try:
                char = chr(char_code)
                bbox = draw.textbbox((0, 0), char, font=font)
                text_width = bbox[2] - bbox[0]
                # Center the character
                draw.text(
                    (x + (cell_width - text_width) // 2, y),
                    char,
                    font=font,
                    fill=color,
                )
            except Exception:
                pass

    # Convert to numpy and reshape into atlas
    atlas_img = np.array(img, dtype=np.uint8)

    # Reshape: (64*cell_h, 64*cell_w, 3) -> (4096, cell_h, cell_w, 3)
    # Layout: 4 color rows × 4 color cols, each containing 16×16 char grid
    atlas = (
        atlas_img.reshape(4, 16, cell_height, 4, 16, cell_width, 3)
        .transpose(0, 3, 1, 4, 2, 5, 6)  # (color_row, color_col, char_row, char_col, h, w, 3)
        .reshape(4096, cell_height, cell_width, 3)
    )

    return atlas, cell_width, cell_height


# Global cached texture atlas (lazy loaded)
_TEXTURE_ATLAS: np.ndarray | None = None
_CELL_WIDTH: int = 0
_CELL_HEIGHT: int = 0


def _get_texture_atlas() -> tuple[np.ndarray, int, int]:
    """Get the cached texture atlas, creating it if needed."""
    global _TEXTURE_ATLAS, _CELL_WIDTH, _CELL_HEIGHT
    if _TEXTURE_ATLAS is None:
        _TEXTURE_ATLAS, _CELL_WIDTH, _CELL_HEIGHT = _create_texture_atlas()
    return _TEXTURE_ATLAS, _CELL_WIDTH, _CELL_HEIGHT


def render_tty_to_rgb(
    tty_chars: np.ndarray,
    tty_colors: np.ndarray | None = None,
) -> np.ndarray:
    """Render TTY characters and colors to an RGB image.

    This is the main rendering function for NetHack/MiniHack TTY output.

    Args:
        tty_chars: Shape (rows, cols) uint8 array of ASCII character codes
        tty_colors: Shape (rows, cols) int array of color indices (0-15).
                   If None, defaults to white (15).

    Returns:
        RGB image array of shape (rows * cell_height, cols * cell_width, 3)
    """
    atlas, cell_width, cell_height = _get_texture_atlas()

    rows, cols = tty_chars.shape

    # Default to white if no colors provided
    if tty_colors is None:
        tty_colors = np.full_like(tty_chars, 15, dtype=np.int32)

    # Mask colors to valid range (0-15)
    colors_masked = tty_colors & 0x0F

    # Compute atlas indices: color * 256 + char_code
    atlas_indices = colors_masked.astype(np.int32) * 256 + tty_chars.astype(np.int32)

    # Look up tiles from atlas and reshape into image
    # atlas[indices] gives shape (rows, cols, cell_h, cell_w, 3)
    tiles = atlas[atlas_indices]

    # Reshape: (rows, cols, cell_h, cell_w, 3) -> (rows*cell_h, cols*cell_w, 3)
    image = (
        tiles
        .transpose(0, 2, 1, 3, 4)  # (rows, cell_h, cols, cell_w, 3)
        .reshape(rows * cell_height, cols * cell_width, 3)
    )

    return image


def render_chars_to_rgb(
    chars: np.ndarray,
    colors: np.ndarray | None = None,
) -> np.ndarray:
    """Render chars/colors arrays to RGB (same as render_tty_to_rgb).

    This is an alias for environments that use 'chars' instead of 'tty_chars'.
    """
    return render_tty_to_rgb(chars, colors)
