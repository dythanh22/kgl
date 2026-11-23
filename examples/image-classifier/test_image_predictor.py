import base64
import io
import unittest

from PIL import Image
from image_predictor import handle


class ImageTests(unittest.TestCase):
    def test_input_validation(self):
        for value in ("", "!", "AAAA", base64.b64encode(bytes(350 * 1024 + 1)).decode()):
            with self.subTest(value=value[:4]), self.assertRaises(ValueError):
                handle(("test", None), {"image": value})

    def test_png_and_jpeg(self):
        for kind in ("PNG", "JPEG"):
            stream = io.BytesIO()
            Image.new("L", (2, 2)).save(stream, format=kind)
            request = {"image": base64.b64encode(stream.getvalue()).decode()}

            def classify(image):
                self.assertEqual(image.mode, "RGB")
                return [{"label": "sample", "score": 1.0}]

            result = handle(("test", classify), request)
            self.assertEqual(result, {"model": "test", "predictions": [{"label": "sample", "score": 1.0}]})
