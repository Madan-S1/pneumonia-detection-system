from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ASSETS_DIR = PROJECT_ROOT / "assets"

IMAGE_SIZE = (224, 224)
MODEL_PATH = ARTIFACTS_DIR / "model.keras"
CLASS_NAMES_PATH = ARTIFACTS_DIR / "class_names.json"

DEFAULT_CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

