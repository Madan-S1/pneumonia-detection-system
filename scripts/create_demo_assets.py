from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"


def base_chest(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    h, w = 360, 360
    y, x = np.ogrid[:h, :w]
    center_x = w / 2
    chest = np.full((h, w), 28, dtype=np.float32)
    body = ((x - center_x) ** 2 / (150**2) + (y - 190) ** 2 / (175**2)) < 1
    chest[body] = 70
    left_lung = ((x - 132) ** 2 / (68**2) + (y - 185) ** 2 / (112**2)) < 1
    right_lung = ((x - 228) ** 2 / (68**2) + (y - 185) ** 2 / (112**2)) < 1
    chest[left_lung | right_lung] = 116
    spine = np.abs(x - center_x) < 14
    chest[spine & body] = 158
    ribs = (np.sin((y - 70) / 13.0) > 0.92) & (left_lung | right_lung)
    chest[ribs] = 150
    chest += rng.normal(0, 5, size=(h, w))
    return Image.fromarray(np.clip(chest, 0, 255).astype(np.uint8), mode="L").filter(ImageFilter.GaussianBlur(0.4))


def make_normal() -> Image.Image:
    image = base_chest(7).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    draw.text((18, 18), "NORMAL SAMPLE", fill=(230, 250, 247, 210))
    return image


def make_pneumonia() -> Image.Image:
    image = base_chest(13).convert("RGB")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.ellipse((86, 135, 180, 260), fill=(225, 225, 210, 70))
    draw.ellipse((190, 118, 286, 270), fill=(225, 225, 210, 85))
    draw.ellipse((132, 210, 250, 304), fill=(225, 225, 210, 65))
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=130, threshold=3))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.text((18, 18), "PNEUMONIA SAMPLE", fill=(255, 246, 230, 220))
    return image


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    make_normal().save(ASSETS / "sample_normal.png")
    make_pneumonia().save(ASSETS / "sample_pneumonia.png")


if __name__ == "__main__":
    main()

