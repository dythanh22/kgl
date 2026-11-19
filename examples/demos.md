# Run a model demo from your laptop

Open a Kaggle notebook, enable a GPU and start **Kaggle Jupyter Server**. Copy its **VS Code Compatible URL**. On the laptop, use Python 3.10 or later, install kgl from this checkout with `python -m pip install .`, then run `kgl connect` and paste the URL at the hidden prompt.

The demo page runs on your laptop. The model runs on the notebook GPU and is available while that notebook session is running.

## Text chat

The notebook needs PyTorch and Transformers. To fetch the example model by Hub ID, enable notebook internet. Alternatively, use the directory of an attached checkpoint that includes a tokenizer with a chat template.

In terminal A on the laptop:

```bash
kgl serve start --require-gpu --model HuggingFaceTB/SmolLM2-360M-Instruct
kgl serve status
kgl infer 'Name one ocean.' --max-tokens 16 --temperature 0
kgl serve proxy --demo
```

Open [the local demo](http://127.0.0.1:8001/), enter a question and select **Send question**. The reply appears as the model generates it. If the proxy uses a bearer key, enter it in the page and select **Check connection**. The key stays in the open page.

If model loading takes a while, use `--no-wait` on the start command to open the proxy during startup. Select **Check connection** again when the worker is ready.

## Image classification

If chat is running, press Ctrl-C in the proxy terminal and run `kgl serve stop` before switching backends.

The notebook needs PyTorch, Pillow and, for the built-in classifier, Transformers and its vision dependencies. Start the default ViT model from the laptop:

```bash
kgl serve start --backend image-classification --require-gpu
kgl serve status
kgl serve proxy --demo
```

Open [the local demo](http://127.0.0.1:8001/), select a JPEG or PNG and choose **Classify image**. The page displays labels and confidence scores. Try two images in sequence. The model stays loaded between requests. Images are limited to 350 KiB and 16 megapixels. Resize large photos before selecting them.

A local application can also send `POST /predict` with `Content-Type: application/json` and the base64-encoded bytes in the `image` field. It receives a JSON object with `model` and a `predictions` list of label and score pairs.

## Use your own checkpoint

For a complete Transformers checkpoint, pass its Hub ID or absolute VM directory with `--model`. To keep the files private across sessions, attach a private Kaggle Dataset and use the directory shown inside the notebook. For small artifacts, use `kgl upload` to copy them from the laptop. Relative upload destinations use `/kaggle/working` by default.

For a custom PyTorch classifier, upload a Python file that provides `load(checkpoint, device)` and `predict(model, image)`. The worker passes an RGB Pillow image to `predict`. Return a list of dictionaries with string `label` and numeric `score` between 0 and 1. The Python file runs in your notebook with its installed dependencies. Use the same architecture, preprocessing and label order as in training.

[The custom predictor example](image_predictor.py) demonstrates the contract with a tiny red/blue classifier. Create its demo weights and start it from the laptop after stopping any previous worker:

```bash
kgl exec <<'PY'
import torch

torch.save({
    "weight": torch.tensor([[3., 0., 0.], [0., 0., 3.]]),
    "bias": torch.zeros(2),
}, "/kaggle/working/kgl-demo-color.pt")
PY
kgl upload examples/image_predictor.py kgl-demo-predictor.py
kgl serve start --backend image-classification --require-gpu \
  --model /kaggle/working/kgl-demo-color.pt \
  --predictor /kaggle/working/kgl-demo-predictor.py
kgl serve proxy --demo
```

Try an image dominated by red and one dominated by blue. Replace the weights and predictor with your own to use the same browser page for your project.

## Troubleshoot and stop

- **Worker busy.** Let the current request finish, then select **Check connection**. Each demo serves one request at a time.
- **Loading fails or CUDA runs out of memory.** Run `kgl serve logs` in another terminal. Check that Kaggle assigned a GPU and try a smaller checkpoint if the model exceeds available VRAM.
- **Session ends.** Start a notebook session again, reconnect with its VS Code Compatible URL, make the checkpoint available and restart the worker and proxy.
- **After the demo.** Press Ctrl-C in the proxy terminal, run `kgl serve stop`, then stop the Kaggle notebook from its interface.
