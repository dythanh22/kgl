import json, urllib.error, urllib.request
req = urllib.request.Request(f"http://127.0.0.1:{__PORT__}/cancel", data=b"",
    headers={__INSTANCE_HEADER__: __INSTANCE__, __REQUEST_ID_HEADER__: __REQUEST_ID__})
try:
    with urllib.request.urlopen(req, timeout=2.5) as reply:
        _emit(json.load(reply))
except (OSError, ValueError):
    _emit({"cancelled": False})
