"""Host, Origin, and access checks for the local HTTP proxy."""

import hmac
import ipaddress

from ..core.errors import ServeError


def is_authorized(headers, api_key: str | None) -> bool:
    return not api_key or hmac.compare_digest(headers.get("Authorization", "").encode(),
                                              f"Bearer {api_key}".encode())


def validate_host_origin(headers, address, listen_host: str) -> tuple[int, str] | None:
    values = headers.get_all("Host", [])
    if len(values) == 1:
        host = values[0].lower()
        bound_ip, bound_port = address[:2]
        names = {listen_host.lower(), bound_ip}
        if ipaddress.ip_address(bound_ip).is_loopback:
            names.add("localhost")
        names -= {"0.0.0.0", "::"}
        allowed = {f"{name}:{bound_port}" for name in names}
        if bound_port == 80:
            allowed.update(names)
        if host in allowed:
            origins = headers.get_all("Origin", [])
            if not origins or origins == [f"http://{host}"]:
                return None
            return 403, "untrusted Origin"
    return 400, "untrusted Host"


def validate_bind(host: str, api_key: str | None, static: bool) -> str:
    if host.lower() == "localhost":
        host = "127.0.0.1"
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = False
    if not loopback and not api_key:
        raise ServeError("binding the proxy outside loopback requires KGL_PROXY_KEY or --api-key")
    if static and not loopback:
        raise ServeError("--static requires a loopback address")
    return host
