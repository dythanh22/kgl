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
                "tools", "tool_choice", "chat_template_kwargs", "presence_penalty",
                "repeat_penalty", "response_format"}

MAX_REQUEST_BYTES = 512 * 1024


def gpu_ready(health):
    return any(str(device).startswith("cuda") for device in
               (health.get("devices") or [health.get("device")]))


def _response_format(value, capabilities):
    if not isinstance(value, dict):
        raise ValueError("response_format must be an object")
    if value == {"type": "json_object"}:
        return
    if value.get("type") != "json_schema" or not capabilities.get("json_schema"):
        raise ValueError("unsupported response_format type")
    wrapper = value.get("json_schema")
    if (set(value) != {"type", "json_schema"} or not isinstance(wrapper, dict)
            or set(wrapper) - {"name", "strict", "schema"}
            or not isinstance(wrapper.get("name"), str) or not wrapper["name"].strip()
            or type(wrapper.get("strict", False)) is not bool or not isinstance(wrapper.get("schema"), dict)):
        raise ValueError("json_schema requires a name and a schema object, with optional boolean strict")
    nested = [wrapper["schema"]]
    while nested:
        item = nested.pop()
        if isinstance(item, dict):
            for key, child in item.items():
                if key == "$ref" and (not isinstance(child, str) or not child.startswith("#/")):
                    raise ValueError("json_schema references must be local JSON pointers")
                nested.append(child)
        elif isinstance(item, list):
            nested.extend(item)


def _llama_options(req, capabilities):
    for field, capability in (("chat_template_kwargs", "enable_thinking"),
                              ("response_format", "response_format"),
                              ("presence_penalty", "presence_penalty"), ("repeat_penalty", "repeat_penalty")):
        if field in req and not capabilities.get(capability):
            raise ValueError(f"{field} is unsupported")
    if "chat_template_kwargs" in req:
        options = req["chat_template_kwargs"]
        if (not isinstance(options, dict) or set(options) != {"enable_thinking"}
                or type(options["enable_thinking"]) is not bool):
            raise ValueError("chat_template_kwargs only supports boolean enable_thinking")
    if "presence_penalty" in req:
        value = req["presence_penalty"]
        if type(value) not in (int, float) or not -2 <= value <= 2:
            raise ValueError("presence_penalty must be between -2 and 2")
    if "repeat_penalty" in req:
        value = req["repeat_penalty"]
        # llama.cpp stores this multiplier as a single-precision float.
        if type(value) not in (int, float) or not 1e-38 <= value <= 3.4e38:
            raise ValueError("repeat_penalty must be a positive finite float32 multiplier")
    if "response_format" in req:
        _response_format(req["response_format"], capabilities)


def chat_args(req, model_id, capabilities=None):
    if not isinstance(req, dict):
        raise ValueError("request must be a JSON object")
    if req.get("model", model_id) != model_id:
        raise ValueError("requested model does not match the running model")
    if capabilities is not None:
        extra = set(req) - LLAMA_FIELDS
        if extra:
            raise ValueError(f"{min(extra)} is unsupported")
    _llama_options(req, capabilities or {})
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
