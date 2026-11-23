"""Local entry points for provisioning llama.cpp and GGUF files on the VM."""
import importlib.resources
import json
import re
from pathlib import Path, PurePosixPath

from ..core.errors import ProtocolError, ServeError
from ..remote import new_marker, render

INSTALL_DIR = "/tmp/kgl/llama"
MODEL_DIR = "/tmp/kgl/models"


def read_preset(name):
    try:
        path = Path(name)
        if not path.is_file() and re.fullmatch(r"[a-z0-9.-]+", str(name)):
            filename = f"{name}.json"
            path = Path(filename)
            if not path.is_file():
                matches = sorted(Path("examples").glob(f"*/{filename}"))
                if len(matches) != 1:
                    raise ValueError("supply a path to a unique preset JSON file")
                path = matches[0]
        preset = json.loads(path.read_text(encoding="utf-8"))
        required = {"repo", "revision", "default_quant", "files", "checksum_file", "checksums"}
        if not isinstance(preset, dict) or required - preset.keys():
            raise ValueError("missing preset fields")
        for key in ("files", "checksums", "capabilities", "defaults"):
            if not isinstance(preset.get(key, {}), dict):
                raise ValueError(f"{key} must be an object")
        if preset["default_quant"] not in preset["files"] or preset["files"].keys() - preset["checksums"].keys():
            raise ValueError("incomplete quantization metadata")
        for filename in (*preset["files"].values(), preset["checksum_file"]):
            if not isinstance(filename, str) or filename in ("", ".", "..") or PurePosixPath(filename).name != filename:
                raise ValueError("preset files must be plain filenames")
        if not all(isinstance(v, str) and re.fullmatch(r"[a-f0-9]{64}", v) for v in preset["checksums"].values()):
            raise ValueError("invalid preset SHA-256")
        if type(preset.get("context", 4096)) is not int or preset.get("context", 4096) <= 0:
            raise ValueError("context must be a positive integer")
        return preset
    except (OSError, ValueError, TypeError) as exc:
        raise ServeError(f"cannot load preset {name}: {exc}") from exc


def vm_operation(rt, operation, options, timeout, write):
    source = importlib.resources.files("kgl").joinpath("vm").joinpath("llama_tools.py").read_text(encoding="utf-8")
    marker = new_marker()
    code = render("exec(__SOURCE__)\n_emit(globals()[__OPERATION__](**__OPTIONS__))",
                  marker=marker, SOURCE=source, OPERATION=operation, OPTIONS=options)
    _, events = rt.run_markers(code, write, marker=marker, total_timeout=timeout,
                               check=True, capture_output=False)
    if not events:
        raise ProtocolError(f"{operation} completed without a result")
    return events[-1]


def fetch_model(rt, preset, repo, quant, directory, token_file, timeout, write):
    if preset and repo:
        raise ServeError("use either --preset or --hf-repo")
    options = {"directory": directory, "token_file": token_file}
    if preset:
        quant = quant or preset["default_quant"]
        if quant not in preset["files"]:
            raise ServeError(f"quant {quant} is unavailable in this preset")
        options.update(repo=preset["repo"], revision=preset["revision"], filename=preset["files"][quant],
                       checksum_file=preset["checksum_file"], expected_hash=preset["checksums"][quant])
    elif repo and quant:
        options["repo"] = repo
    else:
        raise ServeError("supply --preset or --hf-repo with --quant")
    if token_file and not PurePosixPath(token_file).is_absolute():
        raise ServeError("--hf-token-file must be an absolute VM path")
    return vm_operation(rt, "fetch", dict(options, quant=quant), timeout, write)
