# kgl

Use a running Kaggle GPU notebook from your terminal.

## What you can do

- Run Python code or a local script on the notebook and stream its output back to your terminal.
- Upload and download files with SHA-256 verification.
- Run interactive inference on one or more GPUs and stream responses through an OpenAI-compatible chat endpoint.
- Open a local browser demo for text chat or image classification, including your own classifier checkpoint.

kgl connects to a notebook you start in Kaggle. Traffic goes through Jupyter, and the model server listens on loopback inside the notebook. Serving is tied to the life of that session and handles one active inference request at a time.

## Requirements

- Python 3.10 or later on your machine. The CLI has no third-party runtime dependencies.
- A running Kaggle notebook with **Kaggle Jupyter Server** enabled. Enable a GPU for CUDA inference.
- PyTorch and Transformers on the notebook for model serving. Multi-GPU placement also requires Accelerate. To load a model, enable notebook internet access or attach the model files as a Kaggle dataset.

## Quick start

Install from source.

```bash
git clone https://github.com/dythanh22/kgl.git
cd kgl
python -m pip install .
```

In Kaggle, open **Run → Kaggle Jupyter Server → Manually Connect** and copy the **VS Code Compatible URL**. Paste it at the hidden prompt from `kgl connect`.

```bash
kgl connect
kgl exec 'print(42)'
kgl serve start --require-gpu --model Qwen/Qwen2.5-0.5B-Instruct
kgl infer "Explain GPU memory in one sentence." --timing
```

If you have a model attached as a dataset, pass its directory on the notebook to `--model`.

## Call the model from an application

Keep `kgl serve proxy` running in one terminal. In another, call the local endpoint. Requests that omit `model` use the model started above.

```bash
curl --fail http://127.0.0.1:8001/ready
curl --fail-with-body http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello in Vietnamese."}]}'
```

Set `"stream": true` and use `curl -N` for server-sent events. For an OpenAI client, use `http://127.0.0.1:8001/v1` as the base URL and get the model ID from `/v1/models`. If you configure `KGL_PROXY_KEY`, send that value as a bearer token. [Agent example](examples/agent_infer.py)

When finished, stop the local proxy with Ctrl-C, run `kgl stop`, and stop the notebook session in Kaggle.

## Documentation

- [Browser demos for chat, image classification, and your own checkpoint](examples/demos.md)
- [Other text models and image classification](examples/workloads.md)
- [Setup, file transfer, and multi-GPU serving](docs/TESTING.md)
- [Architecture, connection behavior, and configuration](docs/ARCHITECTURE.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Kaggle validation results and roadmap](docs/PLAN_V2.md)
