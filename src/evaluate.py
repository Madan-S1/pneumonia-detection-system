from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ARTIFACTS_DIR, CLASS_NAMES_PATH, IMAGE_SIZE, MODEL_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the trained pneumonia model.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Dataset root containing test/ folder.")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

    test_dir = args.data_dir / "test"
    if not test_dir.exists():
        raise FileNotFoundError("Dataset must contain a test/ directory.")
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Train the model first. Missing artifacts/model.keras.")

    class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8")) if CLASS_NAMES_PATH.exists() else None
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
    )
    test_data = datagen.flow_from_directory(
        test_dir,
        target_size=IMAGE_SIZE,
        batch_size=args.batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    if class_names is None:
        class_names = [name for name, _ in sorted(test_data.class_indices.items(), key=lambda item: item[1])]

    model = tf.keras.models.load_model(MODEL_PATH)
    probabilities = model.predict(test_data, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)
    y_true = test_data.classes

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    matrix = confusion_matrix(y_true, y_pred)

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    (ARTIFACTS_DIR / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
    display.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "confusion_matrix.png", dpi=160)

    print(json.dumps(report, indent=2))
    print(f"Saved confusion matrix to {ARTIFACTS_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()

