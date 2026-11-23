# Running other workloads

Start a Kaggle GPU notebook and run `kgl connect` locally. The examples below need the checkpoint files on the VM. Enable notebook internet for Hub IDs, or substitute the full VM path of an attached checkpoint. These recipes have been checked against the CLI and model cards. GPU acceptance results for Qwen are in [the test report](../docs/reports/2026-11-19-v2.md).

## Serve another text chat model

Choose a checkpoint supported by the notebook's Transformers version through AutoTokenizer and AutoModelForCausalLM. Its tokenizer needs a chat template, and weights plus generation must fit in GPU memory. SmolLM2-360M-Instruct uses all three interfaces. [Model card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)

With the previous worker stopped (`kgl serve stop` if switching models), run locally:

```bash
kgl serve start --require-gpu --model HuggingFaceTB/SmolLM2-360M-Instruct
kgl serve status
kgl infer 'Name one ocean.' --max-tokens 16 --temperature 0
```

Check the model ID and CUDA device in status. To build a demo UI, run the frontend on your machine and have its local application backend call the [chat proxy](../README.md#call-the-model-from-an-application). Stop the worker with `kgl serve stop` before switching workloads.

## Classify an image on the notebook

Use a local JPEG of your choice and a notebook with PyTorch, Transformers and their vision dependencies. This example uploads one image, runs a ViT image classifier on GPU 0 and prints its top prediction as JSON. The path below is reserved for the example and reused on subsequent uploads. [Model card](https://huggingface.co/google/vit-base-patch16-224)

```bash
kgl upload ./photo.jpg kgl-examples/photo.jpg
kgl exec --timeout 300 <<'PY'
import json
from transformers import pipeline

classifier = pipeline("image-classification", model="google/vit-base-patch16-224", device=0)
prediction = classifier("/kaggle/working/kgl-examples/photo.jpg", top_k=1)
print(json.dumps(prediction))
PY
```

The result contains a label and score. The classifier stays loaded in the connected kernel. For another image, repeat the upload and call the existing classifier:

```bash
kgl upload ./second.jpg kgl-examples/photo.jpg
kgl exec 'import json; print(json.dumps(classifier("/kaggle/working/kgl-examples/photo.jpg", top_k=1)))'
```

For a persistent image endpoint, upload the example handler and start the Python backend. [Browser demo and custom checkpoint instructions](demos.md)

## Serve SAM3 segmentation on Kaggle

Run the SAM3 FastAPI service in the notebook and use its local browser demo through a kgl forward. The example covers a pinned Python/CUDA environment, point and text prompts, and PNG masks. [SAM3 on Kaggle](serving-sam3-on-kaggle.md)

## Other inference engines and web services

The managed serving backends are Transformers text chat, llama.cpp, Python handlers and echo. vLLM runs its own server and needs an engine adapter for managed startup, readiness, streaming, cancellation and shutdown. Use `kgl forward --remote-port PORT` for buffered HTTP routes and binary responses. The current local proxy handles one active inference at a time.
