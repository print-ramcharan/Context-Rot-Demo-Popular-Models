"""Small, dependency-free bearer authentication for the demo API.

Configure APP_ENV, AUTH_SECRET_KEY, DEMO_USERNAME and DEMO_PASSWORD through
the deployment environment. The 123456 default exists only in development.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Optional

APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
if APP_ENV != "development" and not os.getenv("AUTH_SECRET_KEY"):
    raise RuntimeError("AUTH_SECRET_KEY must be set in production")

_SECRET = os.getenv(
    "AUTH_SECRET_KEY",
    "local-development-only-change-this-secret-before-deployment",
).encode("utf-8")
if APP_ENV != "development" and (
    len(_SECRET) < 32 or b"replace-this" in _SECRET
):
    raise RuntimeError("Set AUTH_SECRET_KEY to a unique random value of at least 32 bytes in production")
USERNAME = os.getenv("DEMO_USERNAME", "demo")
if APP_ENV == "development":
    PASSWORD = os.getenv("DEMO_PASSWORD", "123456")
else:
    PASSWORD = os.getenv("DEMO_PASSWORD")
    if not os.getenv("DEMO_USERNAME"):
        raise RuntimeError("DEMO_USERNAME must be set outside development")
    if not PASSWORD or len(PASSWORD) < 12:
        raise RuntimeError("DEMO_PASSWORD must be at least 12 characters outside development")

TOKEN_TTL_SECONDS = max(300, int(os.getenv("AUTH_TOKEN_TTL_SECONDS", "3600")))
_PASSWORD_SALT = os.getenv("AUTH_PASSWORD_SALT", "context-rot-demo-v1").encode()
_PASSWORD_HASH = hashlib.pbkdf2_hmac(
    "sha256", PASSWORD.encode("utf-8"), _PASSWORD_SALT, 310_000
)


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def authenticate(username: str, password: str) -> bool:
    candidate = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), _PASSWORD_SALT, 310_000
    )
    return hmac.compare_digest(username.encode(), USERNAME.encode()) and hmac.compare_digest(
        candidate, _PASSWORD_HASH
    )


def create_access_token(user_id: str) -> str:
    now = int(time.time())
    payload = _b64(json.dumps({
        "sub": user_id,
        "iat": now,
        "exp": now + TOKEN_TTL_SECONDS,
    }, separators=(",", ":")).encode())
    signature = _b64(hmac.new(_SECRET, payload.encode(), hashlib.sha256).digest())
    return payload + "." + signature


def verify_access_token(token: str) -> Optional[str]:
    try:
        payload, supplied_signature = token.split(".", 1)
        expected = _b64(hmac.new(_SECRET, payload.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(supplied_signature, expected):
            return None
        claims = json.loads(_unb64(payload))
        user_id = claims.get("sub")
        if not isinstance(user_id, str) or not user_id:
            return None
        if int(claims.get("exp", 0)) <= int(time.time()):
            return None
        return user_id
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None
