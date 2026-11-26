"""Provisioning and planning helpers executed on the notebook VM."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import struct
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request


UPSTREAM = "https://github.com/ggml-org/llama.cpp.git"
HF = "https://huggingface.co"


def capture(args, **kwargs):
    return subprocess.run(args, text=True, capture_output=True, check=True, **kwargs).stdout.strip()


def inspect_server(package, argv):
    package = Path(package)
    command = [str(package / argv[0]), *argv[1:]]
    version = subprocess.run([*command, "--version"], text=True, capture_output=True, check=True, timeout=15)
    help_result = subprocess.run([*command, "--help"], text=True, capture_output=True, check=True, timeout=15)
    help_text = help_result.stdout + help_result.stderr
    options = {}
    for key, flag in {
        "model": "--model", "host": "--host", "port": "--port", "ctx": "--ctx-size",
        "gpu_layers": "--gpu-layers", "split_mode": "--split-mode", "parallel": "--parallel",
        "jinja": "--jinja", "reasoning_format": "--reasoning-format",
        "tensor_split": "--tensor-split",
    }.items():
        if re.search(r"(?<![\w-])" + re.escape(flag) + r"(?![\w-])", help_text):
            options[key] = flag
    if not {"model", "host", "port", "ctx", "gpu_layers", "split_mode", "parallel"} <= options.keys():
        raise ValueError("pinned llama.cpp build is missing required server options")
    return {"argv": argv, "options": options, "version": (version.stdout + version.stderr).strip()}


def install(revision, directory, jobs=2):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("llama.cpp revision must be a full Git commit SHA")
    if not 1 <= jobs <= 8:
        raise ValueError("build jobs must be between 1 and 8")
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".install.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        destination = root / revision
        manifest = destination / "manifest.json"
        if manifest.is_file():
            saved = json.loads(manifest.read_text())
            capture([str(destination / saved["argv"][0]), *saved["argv"][1:], "--version"], timeout=15)
            return {"manifest": str(manifest), **saved, "cached": True}
        started = time.monotonic()
        with tempfile.TemporaryDirectory(dir=root) as working:
            source, build = Path(working) / "source", Path(working) / "build"
            subprocess.run(["git", "init", "-q", str(source)], check=True)
            subprocess.run(["git", "-C", str(source), "fetch", "--depth=1", UPSTREAM, revision], check=True)
            subprocess.run(["git", "-C", str(source), "checkout", "--detach", "FETCH_HEAD"], check=True)
            if capture(["git", "-C", str(source), "rev-parse", "HEAD"]) != revision:
                raise ValueError("llama.cpp checkout does not match the requested revision")
            flags = ["-DGGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=75", "-DCMAKE_BUILD_TYPE=Release",
                     "-DCMAKE_INSTALL_RPATH=$ORIGIN", "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON"]
            subprocess.run(["cmake", "-S", str(source), "-B", str(build), *flags], check=True)
            targets = capture(["cmake", "--build", str(build), "--target", "help"]).splitlines()
            app = any(line.split()[-1:] == ["llama-app"] for line in targets)
            target = "llama-app" if app else "llama-server"
            subprocess.run(["cmake", "--build", str(build), "--parallel", str(jobs), "--target", target], check=True)
            package = Path(working) / "package"
            shutil.copytree(build / "bin", package)
            for library in build.rglob("lib*.so*"):
                dest = package / library.name
                if library.is_file() and not dest.exists():
                    shutil.copy2(library, dest)
            argv = ["llama", "serve"] if app else ["llama-server"]
            saved = {"revision": revision, "cuda_arch": 75, "cmake_flags": flags,
                     **inspect_server(package, argv),
                     "build_bytes": sum(path.stat().st_size for path in package.iterdir() if path.is_file()),
                     "build_seconds": round(time.monotonic() - started, 2)}
            (package / "manifest.json").write_text(json.dumps(saved, indent=2) + "\n")
            package.rename(destination)
        return {"manifest": str(manifest), **saved, "cached": False}


def _open(url, token_file=None, **headers):
    request = urllib.request.Request(url, headers=headers)
    if token_file:
        path = Path(token_file)
        if not path.is_file() or stat.S_IMODE(path.stat().st_mode) != 0o600:
            raise ValueError("HF token file on the VM must have permissions 0600")
        request.add_unredirected_header("Authorization", "Bearer " + path.read_text().strip())
    return urllib.request.urlopen(request, timeout=30)


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for data in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def _input_digest_marker(path, info, expected_hash):
    # Kaggle Input is versioned and mounted read-only. Treat every other path
    # as mutable and verify its contents on each explicit launch.
    resolved = path.resolve()
    if (not resolved.is_relative_to("/kaggle/input")
            or not os.statvfs(resolved).f_flag & os.ST_RDONLY):
        return None
    directory = Path(tempfile.gettempdir()) / f"kgl-gguf-digests-{os.getuid()}"
    directory.mkdir(mode=0o700, exist_ok=True)
    metadata = directory.lstat()
    if (not stat.S_ISDIR(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o700
            or metadata.st_uid != os.getuid()):
        return None
    identity = (str(resolved), info.st_dev, info.st_ino, info.st_size,
                info.st_mtime_ns, info.st_ctime_ns, expected_hash)
    return directory / hashlib.sha256(repr(identity).encode()).hexdigest()


def verify_model(model, quant, expected_hash):
    path = Path(model)
    start = time.monotonic()
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"GGUF is not a regular file for {quant}: {path}")
    try:
        marker = _input_digest_marker(path, info, expected_hash)
    except OSError:
        marker = None
    cached = marker is not None and marker.is_file()
    if not cached and _hash(path) != expected_hash:
        raise ValueError(f"SHA-256 mismatch for {quant}: {path}")
    after = path.stat()
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) != (
            info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns):
        raise ValueError(f"GGUF changed during SHA-256 verification for {quant}: {path}")
    if marker is not None and not cached:
        marker.touch(mode=0o600, exist_ok=True)
    elapsed = time.monotonic() - start
    print(f"Verified {quant} SHA-256 in {elapsed:.2f}s" + (" (cached)" if cached else ""), flush=True)
    return {"sha256": expected_hash, "elapsed_s": elapsed, "cached": cached}


def fetch(repo, quant, directory, revision="main", filename=None, checksum_file="SHA256SUMS",
          token_file=None, input_root="/kaggle/input", expected_hash=None):
    if not re.fullmatch(r"[A-Za-z0-9_][\w.-]*/[A-Za-z0-9_][\w.-]*", repo) or not re.fullmatch(r"[\w.-]+", revision):
        raise ValueError("invalid Hugging Face repository or revision")
    revision = urllib.parse.quote(revision, safe="")

    def matches(name):
        return name == filename if filename else name.lower().endswith("-" + quant.lower() + ".gguf")

    def checksum(lines, name):
        for line in lines.splitlines():
            fields = line.split(maxsplit=1)
            if len(fields) == 2 and fields[1].lstrip("*") == name and re.fullmatch(r"[0-9a-fA-F]{64}", fields[0]):
                return fields[0].lower()
        raise ValueError(f"no SHA-256 in {checksum_file} for {name}")

    visited = set()
    for directory_name, children, files in os.walk(input_root, followlinks=True):
        directory_path = Path(directory_name)
        real_path = directory_path.resolve()
        if real_path in visited:
            children.clear()
            continue
        visited.add(real_path)
        sidecar = directory_path / checksum_file
        for name in sorted(files):
            if matches(name) and (expected_hash or sidecar.is_file()):
                path = directory_path / name
                expected = expected_hash or checksum(sidecar.read_text(), name)
                if _hash(path) != expected:
                    raise ValueError(f"checksum mismatch for attached file {path.name}")
                return {"path": str(path), "source": "dataset", "sha256": expected, "size": path.stat().st_size}

    root = Path(directory).expanduser().resolve()
    if root == Path("/kaggle/working") or Path("/kaggle/working") in root.parents:
        raise ValueError("choose a model cache outside /kaggle/working")
    if expected_hash and filename and re.fullmatch(r"[0-9a-f]{40}", revision):
        cached = root / repo / revision / filename
        if cached.is_file() and _hash(cached) == expected_hash:
            return {"path": str(cached), "source": "cache", "revision": revision,
                    "size": cached.stat().st_size, "sha256": expected_hash}

    with _open(f"{HF}/api/models/{repo}/revision/{revision}?blobs=true", token_file) as reply:
        model = json.load(reply)
    pinned = model["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", pinned):
        raise ValueError("Hugging Face returned an invalid revision")
    if re.fullmatch(r"[0-9a-f]{40}", revision) and pinned != revision:
        raise ValueError("Hugging Face returned a different revision from the requested commit")
    base = f"{HF}/{repo}/resolve/{pinned}"
    siblings = {item["rfilename"]: item for item in model["siblings"]}
    names = [name for name in siblings if "/" not in name and matches(name)]
    if len(names) != 1:
        raise ValueError("expected one matching single-file GGUF, found: " + ", ".join(names))
    name = names[0]
    if checksum_file in siblings:
        with _open(f"{base}/{urllib.parse.quote(checksum_file, safe='')}", token_file) as reply:
            expected = checksum(reply.read(1024 * 1024).decode(), name)
    else:
        expected = (siblings[name].get("lfs") or {}).get("sha256", "")
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"no SHA-256 available for {name}")
    if expected_hash and expected != expected_hash:
        raise ValueError("remote SHA-256 does not match the pinned preset")
    dest = root / repo / pinned / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".gguf.part")
    with dest.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not dest.is_file() or _hash(dest) != expected:
            offset = part.stat().st_size if part.is_file() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            url = f"{base}/{urllib.parse.quote(name, safe='')}"
            if not offset or _hash(part) != expected:
                try:
                    with _open(url, token_file, **headers) as reply:
                        resume = offset and reply.status == 206
                        if resume and not reply.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                            raise ValueError("unexpected response to range request")
                        if not resume:
                            offset = 0
                        remaining = int(reply.headers.get("Content-Length", "0"))
                        if remaining > shutil.disk_usage(dest.parent).free:
                            raise ValueError("insufficient disk space for GGUF")
                        written = offset
                        next_report = written
                        with part.open("ab" if resume else "wb") as output:
                            for data in iter(lambda: reply.read(4 * 1024 * 1024), b""):
                                written += output.write(data)
                                if written >= next_report:
                                    print(f"GGUF {written:,}/{offset + remaining:,} bytes", flush=True)
                                    next_report = written + 128 * 1024 * 1024
                        if remaining and written != offset + remaining:
                            raise OSError("incomplete GGUF download; retry to resume")
                except urllib.error.HTTPError as exc:
                    if exc.code == 416 and offset:
                        part.unlink()
                        raise ValueError("invalid partial download; run fetch again") from None
                    raise ValueError(f"GGUF download returned HTTP {exc.code}") from None
            if _hash(part) != expected:
                part.unlink()
                raise ValueError("GGUF checksum mismatch")
            part.replace(dest)
    return {"path": str(dest), "source": "huggingface", "revision": pinned,
            "size": dest.stat().st_size, "sha256": expected}


def _gguf_metadata(path):
    formats = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
               7: "?", 10: "Q", 11: "q", 12: "d"}
    size = path.stat().st_size
    with path.open("rb") as file:
        if file.read(4) != b"GGUF":
            raise ValueError("model does not have a GGUF header")
        def read(fmt):
            length = struct.calcsize("<" + fmt)
            data = file.read(length)
            if len(data) != length:
                raise ValueError("truncated GGUF metadata")
            return struct.unpack("<" + fmt, data)[0]

        def skip(length):
            if file.tell() + length > size:
                raise ValueError("invalid GGUF metadata length")
            file.seek(length, os.SEEK_CUR)

        def string(keep):
            length = read("Q")
            if keep and length <= 4096:
                data = file.read(length)
                if len(data) != length:
                    raise ValueError("truncated GGUF string")
                return data.decode("utf-8")
            skip(length)
            return None

        def value(kind, keep):
            if kind in formats:
                return read(formats[kind])
            if kind == 8:
                return string(keep)
            if kind == 9:
                element, count = read("I"), read("Q")
                if element in formats:
                    if keep and count <= 1024:
                        return [read(formats[element]) for _ in range(count)]
                    skip(struct.calcsize("<" + formats[element]) * count)
                elif element == 8 and count <= 2_000_000:
                    for _ in range(count):
                        string(False)
                else:
                    raise ValueError("unsupported GGUF array")
                return None
            raise ValueError("unsupported GGUF metadata type")

        if read("I") not in (2, 3):
            raise ValueError("unsupported GGUF version")
        read("Q")
        count = read("Q")
        if count > 16384:
            raise ValueError("too many GGUF metadata entries")
        metadata = {}
        for _ in range(count):
            key = string(True)
            keep = bool(key and not key.startswith("tokenizer."))
            item = value(read("I"), keep)
            if keep and item is not None:
                metadata[key] = item
        return metadata


def _hybrid_memory(metadata):
    arch = metadata.get("general.architecture")
    if arch not in ("qwen35", "qwen35moe"):
        return {}

    def field(name, default=None):
        return metadata.get(f"{arch}.{name}", default)

    blocks = field("block_count")
    interval = field("full_attention_interval", 4)
    if not isinstance(blocks, int) or not isinstance(interval, int) or not (0 < interval <= blocks <= 1024):
        return {}
    recurrent = field("attention.recurrent_layers")
    if not isinstance(recurrent, list) or len(recurrent) != blocks:
        recurrent = [(index + 1) % interval != 0 for index in range(blocks)]
    counts = field("attention.head_count_kv")
    counts = counts if isinstance(counts, list) else [counts] * blocks
    key_length = field("attention.key_length")
    value_length = field("attention.value_length", key_length)
    state, inner = field("ssm.state_size"), field("ssm.inner_size")
    conv, groups = field("ssm.conv_kernel"), field("ssm.group_count")
    parameters = [key_length, value_length, state, inner, conv, groups, *counts]
    if len(counts) != blocks or any(type(item) is not int or item < 0 for item in parameters):
        return {}
    kv_heads = sum(heads for heads, is_recurrent in zip(counts, recurrent) if not is_recurrent)
    recurrence = sum(bool(item) for item in recurrent)
    return {"attention_layers": blocks - recurrence, "recurrent_layers": recurrence,
            "kv_bytes_per_token": 2 * kv_heads * (key_length + value_length),
            "recurrent_bytes": 4 * recurrence * (state * inner + max(0, conv - 1) * (inner + 2 * groups * state))}


def plan(model, ctx, log=None):
    path = Path(model)
    size = path.stat().st_size
    warnings = ["Validate placement and context against the measured llama.cpp buffers on each GPU."]
    try:
        metadata = _gguf_metadata(path)
    except ValueError as exc:
        metadata = {}
        warnings.append(f"Cannot read GGUF metadata: {exc}")
    raw = capture(["nvidia-smi", "--query-gpu=index,memory.free,memory.total", "--format=csv,noheader,nounits"],
                  timeout=10)
    devices = []
    for line in raw.splitlines():
        index, free, total = map(int, line.split(","))
        devices.append({"index": index, "free_mib": free, "total_mib": total})
    available = [max(0, device["free_mib"] - 1024) for device in devices]
    weight_mib = size / (1024 * 1024)
    if weight_mib >= sum(available):
        warnings.append("File size exceeds aggregate free VRAM after reserving 1 GiB per GPU.")
    memory = _hybrid_memory(metadata)
    recommended = None
    if memory.get("kv_bytes_per_token"):
        remaining = sum(available) * 1024 * 1024 - size - memory["recurrent_bytes"]
        upper_bound = max(0, remaining // memory["kv_bytes_per_token"])
        training_context = metadata.get(f"{metadata['general.architecture']}.context_length", ctx)
        if isinstance(training_context, int) and training_context > 0:
            upper_bound = min(upper_bound, training_context)
        recommended = min(ctx, upper_bound)
        memory.update(kv_bytes_at_requested_context=ctx * memory["kv_bytes_per_token"],
                      aggregate_context_bound=upper_bound)
        if recommended < ctx:
            warnings.append("The requested context exceeds the aggregate memory estimate.")
    else:
        warnings.append("Use the pinned build's model-loading log to measure KV and recurrent buffers.")
    if log:
        with Path(log).open("rb") as stream:
            stream.seek(max(0, os.fstat(stream.fileno()).st_size - 1024 * 1024))
            report = stream.read().decode(errors="replace")
        buffers = re.findall(r"\b(CUDA\d+|CPU(?:_Mapped|_Host)?)\s+"
                             r"(?:(model|KV|RS|compute|recurrent|output)\s+)?buffer size\s*=\s*"
                             r"([0-9.]+) MiB", report, re.IGNORECASE)
        memory["logged_buffers"] = [{"device": device, "kind": kind or "unspecified", "size_mib": float(value)}
                                    for device, kind, value in buffers]
    return {"model": str(path), "weight_bytes": size, "gpus": devices, "requested_context": ctx,
            "architecture": metadata.get("general.architecture"), "memory_estimate": memory,
            "recommended_context": recommended, "tentative_tensor_split": available, "warnings": warnings}
