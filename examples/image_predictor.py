"""Example custom classifier for kgl. Match architecture and preprocessing to training."""

import torch

LABELS = ("red", "blue")


def load(checkpoint, device):
    model = torch.nn.Linear(3, len(LABELS)).to(device)
    weights = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(weights)
    return model.eval()


def predict(model, image):
    # The demo checkpoint uses RGB channel averages scaled to [0, 1].
    rgb = image.resize((1, 1)).getpixel((0, 0))
    features = torch.tensor(rgb, dtype=torch.float32, device=model.weight.device) / 255
    scores = model(features).softmax(dim=0).tolist()
    return [
        {"label": label, "score": score}
        for label, score in sorted(zip(LABELS, scores), key=lambda item: item[1], reverse=True)
    ]
