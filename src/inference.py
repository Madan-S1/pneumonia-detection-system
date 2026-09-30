from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from io import BytesIO
from typing import Dict, Optional

import numpy as np
from PIL import Image

from .config import CLASS_NAMES_PATH, DEFAULT_CLASS_NAMES, IMAGE_SIZE, MODEL_PATH
from .demo_model import DemoRadiologyModel


@dataclass
class PredictionResult:
    label: str
    confidence: float
    probabilities: Dict[str, float]
    mode: str
    note: str
    details: Dict[str, float]

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["confidence_percent"] = round(self.confidence * 100.0, 2)
        return data


class TensorFlowPredictor:
    mode = "trained_model"

    def __init__(self, model_path=MODEL_PATH, class_names_path=CLASS_NAMES_PATH):
        import tensorflow as tf

        self.tf = tf
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = self._load_class_names(class_names_path)

    @staticmethod
    def _load_class_names(path) -> list[str]:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return DEFAULT_CLASS_NAMES

    def predict(self, image_bytes: bytes) -> PredictionResult:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image = image.resize(IMAGE_SIZE)
        arr = np.asarray(image, dtype=np.float32)
        arr = self.tf.keras.applications.mobilenet_v2.preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)

        output = self.model.predict(arr, verbose=0)
        values = np.asarray(output).reshape(-1)

        if len(values) == 1:
            pneumonia_probability = float(values[0])
            probabilities = {
                "NORMAL": 1.0 - pneumonia_probability,
                "PNEUMONIA": pneumonia_probability,
            }
        else:
            exp_values = np.exp(values - np.max(values))
            softmax = exp_values / exp_values.sum()
            probabilities = {
                self.class_names[i]: float(softmax[i])
                for i in range(min(len(self.class_names), len(softmax)))
            }

        label = max(probabilities, key=probabilities.get)
        confidence = probabilities[label]
        return PredictionResult(
            label=label,
            confidence=confidence,
            probabilities=probabilities,
            mode=self.mode,
            note="Prediction generated using the trained MobileNetV2 model.",
            details={},
        )


class Predictor:
    def __init__(self):
        self.tensorflow_predictor: Optional[TensorFlowPredictor] = None
        self.demo_predictor = DemoRadiologyModel()
        self._try_load_tensorflow_model()

    def _try_load_tensorflow_model(self) -> None:
        if not MODEL_PATH.exists():
            return
        try:
            self.tensorflow_predictor = TensorFlowPredictor()
        except Exception:
            self.tensorflow_predictor = None

    def predict(self, image_bytes: bytes) -> PredictionResult:
        if self.tensorflow_predictor is not None:
            return self.tensorflow_predictor.predict(image_bytes)

        score = self.demo_predictor.predict(image_bytes)
        probabilities = {
            "NORMAL": score.normal_probability,
            "PNEUMONIA": score.pneumonia_probability,
        }
        label = max(probabilities, key=probabilities.get)
        confidence = probabilities[label]
        return PredictionResult(
            label=label,
            confidence=confidence,
            probabilities=probabilities,
            mode="demo",
            note=(
                "Demo mode is active because no trained TensorFlow model was found. "
                "Train MobileNetV2 to enable real model inference."
            ),
            details={
                "texture_score": score.texture_score,
                "contrast_score": score.contrast_score,
                "brightness_score": score.brightness_score,
            },
        )


_PREDICTOR: Optional[Predictor] = None


def get_predictor() -> Predictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = Predictor()
    return _PREDICTOR


def predict_image_bytes(image_bytes: bytes) -> PredictionResult:
    return get_predictor().predict(image_bytes)

