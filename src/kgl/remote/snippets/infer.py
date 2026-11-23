import base64, json, urllib.request, urllib.error
payload = base64.b64decode(__PAYLOAD__)
options = __OPTIONS__
stream = not options.get("custom") and json.loads(payload).get("stream") is True
headers = {"Content-Type": "application/json", __TIMEOUT_HEADER__: str(__TIMEOUT__)}
if options.get("request_id"):
    headers[__REQUEST_ID_HEADER__] = options["request_id"]
instance = options.get("instance")
route = "/v1/chat/completions"
if instance:
    headers[__INSTANCE_HEADER__] = instance
    route = "/internal/handle" if options.get("custom") else "/internal/chat/completions"
req = urllib.request.Request(f"http://127.0.0.1:{__PORT__}" + route, data=payload, headers=headers)
try:
    with urllib.request.urlopen(req, timeout=__TIMEOUT__) as r:
        if stream:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if line.startswith("data:"):
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        _emit({"status": 200, "done": True})
                        break
                    obj = json.loads(chunk)
                    if "error" in obj:
                        _emit({"status": 0, "error": obj["error"]["message"]})
                        break
                    _emit({"chunk": chunk})
            else:
                raise ValueError("incomplete stream: upstream closed before [DONE]")
        else:
            _emit({"status": 200, "body": json.loads(r.read().decode())})
except urllib.error.HTTPError as e:
    raw = e.read(2000).decode("utf-8", "replace")
    try:
        raw = json.loads(raw)["error"]["message"]
    except (ValueError, KeyError, TypeError):
        pass
    if instance and e.code == 404:
        raw = "running server needs upgrading; run kgl serve stop and kgl serve start"
    _emit({"status": e.code, "error": raw})
except Exception as e:
    _emit({"status": 0, "error": repr(e)[:500]})
