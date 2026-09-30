from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter

from .config import IMAGE_SIZE


@dataclass
class DemoScore:
    pneumonia_probability: float
    normal_probability: float
    texture_score: float
    contrast_score: float
    brightness_score: float


class DemoRadiologyModel:
    """Lightweight image-feature demo predictor for the UI.

    This is not a medical model. It keeps the app runnable before a real
    MobileNetV2 model is trained and saved to artifacts/model.keras.
    """

    mode = "demo"

    def predict(self, image_bytes: bytes) -> DemoScore:
        image = Image.open(BytesIO(image_bytes)).convert("L")
        image = image.resize(IMAGE_SIZE)
        image = image.filter(ImageFilter.MedianFilter(size=3))

        pixels = np.asarray(image, dtype=np.float32) / 255.0
        gy, gx = np.gradient(pixels)
        edges = np.sqrt((gx * gx) + (gy * gy))

        texture_score = float(np.clip(edges.mean() * 8.0, 0.0, 1.0))
        contrast_score = float(np.clip(pixels.std() * 3.0, 0.0, 1.0))
        brightness_score = float(np.clip(pixels.mean(), 0.0, 1.0))

        centered_brightness = 1.0 - abs(brightness_score - 0.48) * 2.0
        raw_score = (
            0.15
            + 4.00 * max(texture_score - 0.08, 0.0)
            + 2.00 * max(contrast_score - 0.50, 0.0)
            + 0.08 * np.clip(centered_brightness, 0.0, 1.0)
        )
        pneumonia_probability = float(np.clip(raw_score, 0.02, 0.98))
        normal_probability = 1.0 - pneumonia_probability

        return DemoScore(
            pneumonia_probability=pneumonia_probability,
            normal_probability=normal_probability,
            texture_score=texture_score,
            contrast_score=contrast_score,
            brightness_score=brightness_score,
        )
