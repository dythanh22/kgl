import base64
import os
from http.client import HTTPConnection
from pathlib import Path

conn = HTTPConnection("127.0.0.1", __PORT__, timeout=90)
try:
    body = Path(__FILE__).read_bytes() if __FILE__ else base64.b64decode(__BODY__)
    headers = __HEADERS__
    conn.putrequest(__METHOD__, __PATH__, skip_accept_encoding=True)
    for name, value in headers:
        conn.putheader(name, value)
    if body or __METHOD__ in ("POST", "PUT", "PATCH"):
        conn.putheader("Content-Length", str(len(body)))
    conn.endheaders(body or None)
    reply = conn.getresponse()
    length = reply.length
    data = reply.read(__LIMIT__ + 1)
    if len(data) > __LIMIT__:
        _emit({"oversize": True})
    elif length is not None and length != len(data):
        raise OSError("incomplete HTTP response")
    else:
        headers = reply.getheaders()
        if len(data) > __CHUNK__:
            with os.fdopen(os.open(__RESULT__, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as output:
                output.write(data)
            _emit({"status": reply.status, "headers": headers, "file": True})
        else:
            _emit({"status": reply.status, "headers": headers, "body": base64.b64encode(data).decode()})
finally:
    conn.close()
