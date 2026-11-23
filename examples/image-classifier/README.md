# Image classification

Upload the handler and start the worker from the repository root.

```bash
kgl upload examples/image-classifier/image_predictor.py kgl-image.py
kgl serve start --backend python --handler /kaggle/working/kgl-image.py --model google/vit-base-patch16-224 --require-gpu
kgl serve proxy --static examples/chat-demo
```

Open the printed URL and select a JPEG or PNG. The handler accepts a JSON object with a base64 `image` field and returns the model name and its predictions. It loads a Transformers pipeline for a Hub ID or model directory, and a red/blue linear classifier for a checkpoint file. [Custom checkpoint walkthrough](../demos.md#use-your-own-checkpoint)

Run `python -m unittest discover -s examples/image-classifier` with Pillow installed to check input validation and JPEG/PNG decoding.
