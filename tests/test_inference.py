from io import BytesIO
import unittest

from PIL import Image

from src.inference import predict_image_bytes


def png_bytes(color: int = 120) -> bytes:
    image = Image.new("L", (224, 224), color=color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class InferenceTest(unittest.TestCase):
    def test_prediction_result_has_required_fields(self):
        result = predict_image_bytes(png_bytes()).to_dict()

        self.assertIn(result["label"], {"NORMAL", "PNEUMONIA"})
        self.assertGreaterEqual(result["confidence"], 0)
        self.assertLessEqual(result["confidence"], 1)
        self.assertIn("NORMAL", result["probabilities"])
        self.assertIn("PNEUMONIA", result["probabilities"])
        self.assertIn(result["mode"], {"demo", "trained_model"})


if __name__ == "__main__":
    unittest.main()
