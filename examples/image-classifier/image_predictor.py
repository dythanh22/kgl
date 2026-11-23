"""Transformers or red/blue checkpoint inference for the Python handler example."""
import base64
import io
from pathlib import Path

from PIL import Image

LABELS = ("red", "blue")


def load(checkpoint, device):
    import torch

    if not Path(checkpoint).is_file():
        from transformers import pipeline

        classifier = pipeline("image-classification", model=checkpoint, device=device)

        def classify(image):
            return classifier(image, top_k=3)
    else:
        model = torch.nn.Linear(3, len(LABELS)).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model.eval()

        @torch.inference_mode()
        def classify(image):
            rgb = image.resize((1, 1)).getpixel((0, 0))
            features = torch.tensor(rgb, dtype=torch.float32, device=device) / 255
            scores = model(features).softmax(dim=0).tolist()
            return [{"label": label, "score": score}
                    for label, score in sorted(zip(LABELS, scores), key=lambda item: item[1], reverse=True)]

    return checkpoint, classify


def handle(loaded, request):
    checkpoint, classify = loaded
    if not isinstance(request, dict) or not isinstance(request.get("image"), str):
        raise ValueError("image must be a base64 string")
    data = base64.b64decode(request["image"], validate=True)
    if not 0 < len(data) <= 350 * 1024:
        raise ValueError("image must be between 1 byte and 350 KiB")
    try:
        with Image.open(io.BytesIO(data), formats=("JPEG", "PNG")) as image:
            if image.width * image.height > 16_000_000:
                raise ValueError("image exceeds 16 megapixels")
            rgb = image.convert("RGB")
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("expected a JPEG or PNG image") from exc

    predictions = classify(rgb)
    if not isinstance(predictions, list) or any(
        not isinstance(item, dict) or not isinstance(item.get("label"), str)
        or type(item.get("score")) not in (int, float) or not 0 <= item["score"] <= 1
        for item in predictions
    ):
        raise RuntimeError("classifier must return labels and numeric scores between 0 and 1")
    return {"model": checkpoint, "predictions": predictions}
