from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ARTIFACTS_DIR, CLASS_NAMES_PATH, IMAGE_SIZE, MODEL_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MobileNetV2 for pneumonia detection.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Dataset root containing train/val/test folders.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--fine-tune-at", type=int, default=120, help="Unfreeze layers from this index onward.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    from sklearn.utils.class_weight import compute_class_weight

    train_dir = args.data_dir / "train"
    val_dir = args.data_dir / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError("Dataset must contain train/ and val/ directories.")

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        rotation_range=12,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.12,
        shear_range=0.08,
        horizontal_flip=True,
    )
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
    )

    train_data = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=args.batch_size,
        class_mode="categorical",
    )
    val_data = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMAGE_SIZE,
        batch_size=args.batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    class_names = [name for name, _ in sorted(train_data.class_indices.items(), key=lambda item: item[1])]
    num_classes = len(class_names)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = True
    for layer in base_model.layers[: args.fine_tune_at]:
        layer.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_data.classes),
        y=train_data.classes,
    )
    class_weight_map = {index: float(weight) for index, weight in enumerate(class_weights)}

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy", mode="max"),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.3),
    ]

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=args.epochs,
        class_weight=class_weight_map,
        callbacks=callbacks,
    )

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    model.save(MODEL_PATH)
    CLASS_NAMES_PATH.write_text(json.dumps(class_names, indent=2), encoding="utf-8")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Training")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Training")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "training_history.png", dpi=160)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved class names to {CLASS_NAMES_PATH}")


if __name__ == "__main__":
    main()

