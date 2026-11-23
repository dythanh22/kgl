"""Resolve model files and settings for a llama.cpp launch."""
from .model_fetch import fetch_model, read_preset
from ..core.errors import ServeError


def prepare_llama(rt, a, session_manifest, write):
    model = a.model
    manifest = a.llama_manifest or session_manifest
    if not manifest or not manifest.startswith("/"):
        raise ServeError("run kgl model install llama or pass --llama-manifest with an absolute VM path")
    preset = read_preset(a.preset) if a.preset else None
    if a.hf_repo or (preset and not a.model):
        if a.model:
            raise ServeError("use either --model or --hf-repo")
        model = fetch_model(rt, preset, a.hf_repo, a.quant, a.model_dir, a.hf_token_file,
                            a.wait_timeout, write)["path"]
    elif not a.model or a.quant:
        raise ServeError("supply a GGUF path or choose a model with --preset or --hf-repo")
    context = a.ctx if a.ctx is not None else (preset or {}).get("context", 4096)
    return model, {
        "llama_manifest": manifest, "ctx": context,
        "settings": {key: preset.get(key, {}) for key in ("capabilities", "defaults")} if preset else {},
    }
