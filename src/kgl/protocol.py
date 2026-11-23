"""Standard-library contracts shared by the CLI and VM gateway."""

INSTANCE_HEADER = "X-Kgl-Instance"
REQUEST_ID_HEADER = "X-Kgl-Request-Id"
TIMEOUT_HEADER = "X-Kgl-Timeout"


def server_health(backend, model, instance, status, error):
    return {"status": status, "model": model, "instance": instance, "device": backend.device,
            "devices": backend.devices, "dtype": backend.dtype, "backend": backend.name, "error": error,
            **backend.health()}


LLAMA_FIELDS = {"model", "messages", "n", "stream", "stream_options", "max_tokens",
                "max_completion_tokens", "temperature", "top_p", "top_k", "stop", "seed",
                "tools", "tool_choice"}

MAX_REQUEST_BYTES = 512 * 1024


def gpu_ready(health):
    return any(str(device).startswith("cuda") for device in
               (health.get("devices") or [health.get("device")]))


def chat_args(req, model_id, capabilities=None):
    if not isinstance(req, dict):
        raise ValueError("request must be a JSON object")
    if req.get("model", model_id) != model_id:
        raise ValueError("requested model does not match the running model")
    if capabilities is not None:
        extra = set(req) - LLAMA_FIELDS
        if extra:
            raise ValueError(f"{min(extra)} is unsupported")
    for key in ("tools", "tool_choice", "functions", "function_call", "stop", "seed", "top_k"):
        capability = "tools" if key == "tool_choice" else key
        if req.get(key) is not None and not (capabilities or {}).get(capability):
            raise ValueError(f"{key} is unsupported")
    if req.get("seed") is not None and type(req["seed"]) is not int:
        raise ValueError("seed must be an integer")
    if req.get("top_k") is not None and (type(req["top_k"]) is not int or req["top_k"] < 0):
        raise ValueError("top_k must be a non-negative integer")
    stop = req.get("stop")
    if stop is not None and not (isinstance(stop, str) or (
            isinstance(stop, list) and all(isinstance(item, str) for item in stop))):
        raise ValueError("stop must be a string or an array of strings")
    if type(req.get("n", 1)) is not int or req.get("n", 1) != 1:
        raise ValueError("n must be 1")
    if type(req.get("stream", False)) is not bool:
        raise ValueError("stream must be a boolean")
    stream_options = req.get("stream_options")
    if stream_options is not None and (
            not req.get("stream") or not isinstance(stream_options, dict)
            or set(stream_options) - {"include_usage"}
            or type(stream_options.get("include_usage", False)) is not bool):
        raise ValueError("stream_options only supports include_usage for streaming requests")
    if "max_completion_tokens" in req and "max_tokens" in req:
        raise ValueError("supply only one of max_tokens and max_completion_tokens")
    maximum = req.get("max_completion_tokens", req.get("max_tokens", 256))
    if type(maximum) is not int or maximum <= 0:
        raise ValueError("completion token limit must be a positive integer")
    temperature, top_p = req.get("temperature", 0.7), req.get("top_p", 0.95)
    if type(temperature) not in (int, float) or not 0 <= temperature <= 2:
        raise ValueError("temperature must be between 0 and 2")
    if type(top_p) not in (int, float) or not 0 < top_p <= 1:
        raise ValueError("top_p must be greater than 0 and at most 1")
    messages = req.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    normalized = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in ("system", "user", "assistant"):
            raise ValueError("messages must have system, user or assistant roles")
        content = message.get("content")
        if isinstance(content, list):
            if not content or any(not isinstance(part, dict) or part.get("type") != "text"
                                  or not isinstance(part.get("text"), str) for part in content):
                raise ValueError("only text message parts are supported")
            content = "".join(part["text"] for part in content)
        if not isinstance(content, str):
            raise ValueError("message content must be text")
        normalized.append({"role": message["role"], "content": content})
    return normalized, maximum, temperature, top_p


def chat_defaults(request, defaults):
    result = dict(request)
    for key in ("temperature", "top_p", "top_k"):
        if key in defaults:
            result.setdefault(key, defaults[key])
    if "max_completion_tokens" not in result and "max_tokens" in defaults:
        result.setdefault("max_tokens", defaults["max_tokens"])
    return result
