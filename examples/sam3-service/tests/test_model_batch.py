from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
from PIL import Image

from sam3_service.config import Settings
from sam3_service.model import Sam3Model


def test_predict_instances_encodes_image_once_and_preserves_box_order():
    class Processor:
        confidence_threshold = 0.5
        calls = 0

        def set_image(self, image):
            self.calls += 1
            return {"original_height": image.height, "original_width": image.width}

    class Predictor:
        reset_calls = 0

        def reset_predictor(self):
            self.reset_calls += 1

    class FakeSam3:
        def __init__(self):
            self.inst_interactive_predictor = Predictor()
            self.boxes = []

        def predict_inst(self, state, *, point_coords, point_labels, box, multimask_output):
            self.boxes.append(tuple(box.tolist()))
            mask = np.zeros((1, state["original_height"], state["original_width"]), dtype=bool)
            x1, y1, x2, y2 = map(int, box)
            mask[0, y1:y2, x1:x2] = True
            return mask, np.array([0.9]), None

    model = Sam3Model(Settings(auth_disabled=True, device="cpu"))
    processor = Processor()
    sam3 = FakeSam3()
    model._torch = SimpleNamespace(inference_mode=nullcontext)
    model._model = sam3
    model._processor = processor
    image = Image.new("RGB", (64, 48))
    boxes = [(2.0, 3.0, 12.0, 15.0), (30.0, 20.0, 50.0, 40.0)]

    results = model.predict_instances(image, boxes)

    assert processor.calls == 1
    assert sam3.boxes == boxes
    assert [result.prompt_id for result in results] == [0, 1]
    assert [int(result.mask.sum()) for result in results] == [120, 400]
    assert sam3.inst_interactive_predictor.reset_calls == 1
