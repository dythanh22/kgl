# kgl

Run Python, move files, and serve models on a running Kaggle notebook from your own terminal. The local CLI uses the Python standard library. Your code, model weights, and GPU workloads run on Kaggle.

**How a request travels** · A local application calls the kgl proxy on your machine. The proxy sends the request through the notebook’s Jupyter connection to a model worker listening on the notebook’s loopback interface. The CLI can also run code and call the worker directly. The worker serves one inference request at a time and lives for the duration of the notebook session.

[Install and connect](#install-and-connect) · [Run a model](#run-a-model) · [Call the API](#call-the-api) · [Stop or switch sessions](#stop-or-switch-sessions) · [Examples](#examples)

## Install and connect

1. **On your machine**, activate a Python 3.10+ environment and install kgl from the checkout. Run the commands in this guide from the repository root.

   ```bash
   git clone https://github.com/dythanh22/kgl.git
   cd kgl
   python -m pip install .
   kgl --version
   ```

2. **In Kaggle**, start a notebook. Enable a GPU if your workload needs CUDA. Open **Run → Kaggle Jupyter Server → Manually Connect** and copy the **VS Code Compatible URL**. Treat the URL as a credential and paste it at the hidden prompt in the next step.

3. **Back in your terminal**, connect and run a remote smoke check.

   ```bash
   kgl connect
   kgl exec --timeout 15 'print(2)'
   kgl status
   ```

   The second command should print `2`. The status command shows the selected kernel, connection and available resources. Subsequent commands reuse the saved connection. To keep independent local session records, set `KGL_HOME` to a different directory for each notebook **before** connecting, and use the same setting in every terminal serving that notebook.

## Run code and transfer files

The following round trip creates a local script, runs it on the notebook, then downloads its output. Relative remote file paths resolve under `/kaggle/working` by default. Transfers verify the file with SHA-256.

```bash
echo 'from pathlib import Path; Path("/kaggle/working/result.txt").write_text("Hello from Kaggle\n")' > hello_kgl.py
kgl upload hello_kgl.py hello_kgl.py
kgl exec --shell 'python /kaggle/working/hello_kgl.py'
kgl download result.txt ./result.txt
```

For a standalone local script, `kgl exec -f hello_kgl.py` sends its contents directly to the kernel. Use `kgl exec --help` for execution timeouts and interrupt behavior.

## Run a model

Choose one worker for the connected notebook. Backend dependencies belong in the notebook environment.

### Transformers

The notebook needs PyTorch and Transformers. Multi-GPU placement also requires Accelerate. Enable notebook Internet access for a Hub model, or pass the remote directory of an attached checkpoint as `--model`.

```bash
kgl serve start --require-gpu --model Qwen/Qwen2.5-0.5B-Instruct
kgl serve status
kgl infer "Explain GPU memory in one sentence." --timing
```

To use a browser or another application, start the foreground proxy below and leave its terminal open. Use another terminal for API calls. [Local browser demo](http://127.0.0.1:8001/)

```bash
kgl serve proxy --static examples/chat-demo
```

### Qwen GGUF on two Tesla T4 GPUs

Select **GPU T4 ×2**, attach the Q4_K_M GGUF specified in the preset, and grant the notebook access to the pinned llama.cpp build dataset. With the notebook connected, run the following from the same local Python environment as kgl. The proxy management script requires Linux.

```bash
python examples/qwen35b-llama/ready.py
python examples/qwen35b-llama/proxy.py
python examples/qwen35b-llama/check_answer.py
```

`ready.py` prepares the pinned build and checks the worker on both GPUs. `proxy.py` starts or reuses a **background** proxy on port 8001. The final check should print `Answer: '4'`. The Qwen preset uses a 32,768-token context. In the browser, the default reply limit is 2,048 tokens and 4,096 is available for longer reports. [Local chat](http://127.0.0.1:8001/) · [Qwen operator guide](examples/qwen35b-llama/QWEN35B-KAGGLE.md) · [Build and recovery](examples/qwen35b-llama/BUILD.md)

The Linux desktop launcher can attach a notebook URL, start Qwen, and reopen chat. [Desktop instructions](examples/qwen35b-llama/QWEN35B-KAGGLE.md#launcher-desktop-trên-linux)

## Call the API

An application on the proxy machine can use **OpenAI-compatible Chat Completions** at `http://127.0.0.1:8001/v1`. It can run in its own environment while the notebook, worker and proxy are running. Use `GET /v1/models` to discover the model ID. The managed Qwen example exposes the alias `qwen-local`.

```bash
curl --fail http://127.0.0.1:8001/ready
curl --fail http://127.0.0.1:8001/v1/models
curl --fail-with-body http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-local","messages":[{"role":"user","content":"Say hello in Vietnamese."}],"max_tokens":2048,"stream":false}'
```

Read the answer from `choices[0].message.content`. Replace the alias for a different worker, or omit `model` to use the active model. For SSE, set `"stream":true` and use `curl -N`. The proxy listens on loopback by default. If you set `KGL_PROXY_KEY` before starting it, include `Authorization: Bearer <key>` on client requests. On `503 worker_busy`, respect `Retry-After` before retrying. [API examples, authentication and error handling](examples/qwen35b-llama/API.md)

## Stop or switch sessions

| What you want to do | Steps |
|---|---|
| Check what is running | Use `kgl status` for the notebook connection, `kgl serve status` for the worker, and `kgl serve logs --lines 50` for startup or runtime errors. |
| Stop inference, keep using the notebook | Wait for the current request to finish, then run `kgl serve stop`. You can still use `kgl exec`, upload and download. |
| Switch models on the same notebook | Stop a foreground proxy with Ctrl-C, run `kgl serve stop`, then start the new worker and its proxy. For managed Qwen, rerun `ready.py` and `proxy.py` using the commands above. |
| Finish a session and release the GPU | Download any outputs you need. Stop the foreground proxy if applicable, run `kgl stop` to stop the tracked worker and clear the local connection, then **stop the notebook in Kaggle** and confirm it has ended. |
| Reconnect after a notebook restart | Copy the new compatible URL, run `kgl connect` again, and repeat the `print(2)` check. On a new VM, check attached model inputs, then start the worker and proxy again. |
| Move to a different notebook | For a planned switch, finish the old session first. Connect to the new notebook and start its worker. If keeping both session records, use separate `KGL_HOME` directories. |

**Keep the same Qwen chat address after a VM change.** The managed proxy can stay on port 8001 between sessions. If the disconnected proxy belongs to a different session directory, run the following after the new worker is ready, with `KGL_HOME` set to the **new** session directory. The script checks ownership and readiness before handing over the port. Using the same browser profile and origin preserves local chat history. [VM handoff guide](examples/qwen35b-llama/QWEN35B-KAGGLE.md)

```bash
python examples/qwen35b-llama/proxy.py --from-home /path/to/previous/session
```

For explicit kernel deletion, `kgl stop --kill-kernel` also requests deletion of the connected kernel. This can interrupt notebook work. If deletion cannot be confirmed, kgl reports an error and retains the local session record for a retry. The Kaggle notebook itself is stopped from the Kaggle UI.

## Examples

| Workload | Start here |
|---|---|
| Qwen 35B GGUF on two GPUs | [Setup and daily operation](examples/qwen35b-llama/QWEN35B-KAGGLE.md) · [API integration](examples/qwen35b-llama/API.md) · [Evaluation cases](examples/qwen35b-llama/eval/README.md) |
| Browser chat | [Chat demo, history, attachments and prompt controls](examples/chat-demo/README.md) |
| Image classification and custom checkpoints | [Run the browser demo](examples/demos.md) · [Python handler example](examples/image-classifier/README.md) |
| SAM3 segmentation service | [Kaggle walkthrough](examples/serving-sam3-on-kaggle.md) · [Service project](examples/sam3-service/README.md) |
| Script or agent integration | [Python inference example](examples/agent/agent_infer.py) |
| Other models and workloads | [Text and image serving recipes](examples/workloads.md) |

## Reference

- [Architecture, Jupyter transport and configuration](docs/ARCHITECTURE.md)
- [Setup checks, file transfer and multi-GPU validation](docs/TESTING.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Daily Qwen design](docs/DESIGN-v3.md) · [Chat UI implementation plan](docs/PLAN-chat-ui.md)
- [Kaggle validation results and roadmap](docs/reports/2026-11-19-v2.md)

## Migrating an existing checkout

- Import client operations directly from `kgl`. Examples can use `Settings`, `SessionStore`, `Runtime`, `upload_file`, `download_file`, `infer_chat` and `KglError`.
- Run `kgl serve proxy --static examples/chat-demo` from the repository root to open the browser demo. This replaces `--demo`.
- Use `kgl model install` and `kgl model plan` for the former `serve install` and `serve plan` commands. Pass a local JSON path to `--preset`. A short name resolves to `NAME.json` in the current directory or a unique match under `examples/*/` when run from the checkout root. [Qwen preset](examples/qwen35b-llama/qwen3.8-35b-a3b.json)
- Stop the previous image worker before starting a Python handler. Replace `--backend image-classification --predictor FILE` with `--backend python --handler FILE` and send JSON to `POST /handle`. The module implements `load(path, device)` and `handle(model, request_json)`. [Classifier example](examples/image-classifier/README.md)
- Use `kgl forward --remote-port 8013` for a notebook HTTP service listening on port 8013. The local endpoint defaults to `127.0.0.1:8014`. Forwarding buffers request and response bodies and applies `KGL_REQUEST_BYTES` to each.

## License

[MIT](LICENSE)
