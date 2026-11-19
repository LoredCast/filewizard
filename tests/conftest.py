"""Shared fixtures and mock OIDC provider for FileWizard tests."""

import base64
import hashlib
import re
import sys
import time
import types
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from joserfc import jwt
from joserfc.jwk import RSAKey

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def install_heavy_dependency_stubs() -> None:
    """Stub optional/heavy converter dependencies that main.py imports at module level."""
    for mod_name in ("ocrmypdf", "pypdf", "pytesseract", "faster_whisper", "piper", "PIL"):
        if mod_name in sys.modules:
            continue
        mod = types.ModuleType(mod_name)
        if mod_name == "pytesseract":
            mod.TesseractNotFoundError = type("TesseractNotFoundError", (Exception,), {})
            mod.get_languages = lambda config="": ["eng", "deu", "fra", "osd"]
        elif mod_name == "faster_whisper":
            mod.WhisperModel = object
        elif mod_name == "piper":
            mod.PiperVoice = None
        elif mod_name == "PIL":
            img_mod = types.ModuleType("PIL.Image")
            img_mod.Image = object
            mod.Image = img_mod
            mod.UnidentifiedImageError = type("UnidentifiedImageError", (Exception,), {})
            sys.modules["PIL.Image"] = img_mod
        sys.modules[mod_name] = mod


class MockOIDCProvider:
    """In-memory OIDC provider supporting PKCE S256, public clients, and RS256 ID tokens."""

    def __init__(
        self,
        issuer: str = "https://idp.example.test",
        require_pkce: bool = True,
        reject_pkce: bool = False,
        public_client: bool = False,
        expose_userinfo: bool = True,
        extra_id_token_claims: dict | None = None,
        userinfo_payload: dict | None = None,
    ):
        self.issuer = issuer
        self.require_pkce = require_pkce
        self.reject_pkce = reject_pkce
        self.public_client = public_client
        self.expose_userinfo = expose_userinfo
        self.extra_id_token_claims = extra_id_token_claims or {}
        self.client_id = "filewiz-test"
        self.client_secret = "" if public_client else "super-secret-value"
        self.rsa_key = RSAKey.generate_key(2048, auto_kid=True)
        self.jwks = {"keys": [self.rsa_key.as_dict(private=False)]}
        self.codes: dict[str, dict] = {}
        self.token_requests: list[dict] = []
        self.access_token = "access-tok-123"
        self.userinfo_payload = userinfo_payload or {
            "sub": "user-123",
            "email": "admin@example.com",
            "email_verified": True,
            "name": "Test Admin",
            "groups": ["admins", "staff", "engineering"],
        }

    @property
    def metadata_url(self) -> str:
        return f"{self.issuer}/.well-known/openid-configuration"

    def issue_code_from_authorize_url(self, authorize_url: str) -> tuple[str, str]:
        parsed = urlparse(authorize_url)
        qs = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        assert qs["response_type"] == "code"
        assert qs["client_id"] == self.client_id

        challenge = qs.get("code_challenge")
        method = qs.get("code_challenge_method")

        if self.require_pkce:
            assert challenge, "IdP requires code_challenge"
            assert method == "S256", f"IdP requires S256, got {method!r}"
            assert re.fullmatch(r"[A-Za-z0-9\-_]{43}", challenge), f"Bad S256 challenge: {challenge!r}"
        if self.reject_pkce:
            assert challenge is None, f"Legacy IdP rejects code_challenge, got {challenge!r}"
            assert method is None, f"Legacy IdP rejects code_challenge_method, got {method!r}"

        code = f"code-{len(self.codes) + 1}"
        self.codes[code] = {
            "redirect_uri": qs["redirect_uri"],
            "nonce": qs.get("nonce"),
            "code_challenge": challenge,
            "code_challenge_method": method,
        }
        return code, qs["state"]

    def make_httpx_transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if url == self.metadata_url:
                doc = {
                    "issuer": self.issuer,
                    "authorization_endpoint": f"{self.issuer}/authorize",
                    "token_endpoint": f"{self.issuer}/token",
                    "jwks_uri": f"{self.issuer}/jwks",
                    "end_session_endpoint": f"{self.issuer}/logout",
                    "code_challenge_methods_supported": ["S256"],
                    "id_token_signing_alg_values_supported": ["RS256"],
                }
                if self.expose_userinfo:
                    doc["userinfo_endpoint"] = f"{self.issuer}/userinfo"
                return httpx.Response(200, json=doc)
            if url == f"{self.issuer}/jwks":
                return httpx.Response(200, json=self.jwks)
            if url == f"{self.issuer}/token":
                body = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
                self.token_requests.append({"headers": dict(request.headers), "body": body})

                if self.public_client:
                    if request.headers.get("authorization"):
                        return httpx.Response(401, json={"error": "invalid_client"})
                    if body.get("client_id") != self.client_id:
                        return httpx.Response(401, json={"error": "invalid_client"})
                else:
                    auth_header = request.headers.get("authorization", "")
                    expected_basic = base64.b64encode(
                        f"{self.client_id}:{self.client_secret}".encode()
                    ).decode()
                    if auth_header != f"Basic {expected_basic}":
                        return httpx.Response(401, json={"error": "invalid_client"})

                code = body.get("code")
                record = self.codes.pop(code, None)
                if not record:
                    return httpx.Response(400, json={"error": "invalid_grant"})

                verifier = body.get("code_verifier")
                if self.require_pkce or record["code_challenge"]:
                    if not verifier:
                        return httpx.Response(
                            400, json={"error": "invalid_grant", "error_description": "Missing code_verifier"}
                        )
                    digest = hashlib.sha256(verifier.encode("ascii")).digest()
                    computed = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
                    if computed != record["code_challenge"]:
                        return httpx.Response(
                            400, json={"error": "invalid_grant", "error_description": "PKCE verification failed"}
                        )
                elif self.reject_pkce and verifier is not None:
                    return httpx.Response(
                        400, json={"error": "invalid_request", "error_description": "Unexpected code_verifier"}
                    )

                now = int(time.time())
                claims = {
                    "iss": self.issuer,
                    "sub": self.userinfo_payload["sub"],
                    "aud": self.client_id,
                    "iat": now,
                    "exp": now + 3600,
                    "email": self.userinfo_payload.get("email"),
                    "email_verified": self.userinfo_payload.get("email_verified", True),
                    "name": self.userinfo_payload.get("name"),
                }
                claims.update(self.extra_id_token_claims)
                if record["nonce"]:
                    claims["nonce"] = record["nonce"]
                id_token = jwt.encode({"alg": "RS256", "kid": self.rsa_key.kid}, claims, self.rsa_key)
                return httpx.Response(
                    200,
                    json={
                        "access_token": self.access_token,
                        "token_type": "Bearer",
                        "expires_in": 3600,
                        "id_token": id_token,
                    },
                )
            if url == f"{self.issuer}/userinfo":
                if request.headers.get("authorization") != f"Bearer {self.access_token}":
                    return httpx.Response(401, json={"error": "invalid_token"})
                return httpx.Response(200, json=self.userinfo_payload)
            return httpx.Response(404, json={"error": f"unhandled {url}"})

        return httpx.MockTransport(handler)


@pytest.fixture
def load_app(tmp_path, monkeypatch):
    """Factory fixture that imports main.py isolated inside tmp_path with custom env vars."""

    def _load(*, local_only: bool = False, extra_env: dict | None = None):
        install_heavy_dependency_stubs()
        uploads = tmp_path / "uploads"
        processed = tmp_path / "processed"
        data = tmp_path / "data"
        config = tmp_path / "config"
        for d in (uploads, processed, data, config):
            d.mkdir(exist_ok=True)

        monkeypatch.setenv("LOCAL_ONLY", "True" if local_only else "False")
        monkeypatch.setenv("SECRET_KEY", "test-secret-key-for-filewizard-tests")
        monkeypatch.setenv("UPLOADS_DIR", str(uploads))
        monkeypatch.setenv("PROCESSED_DIR", str(processed))
        monkeypatch.setenv("DATA_DIR", str(data))
        for key, value in (extra_env or {}).items():
            monkeypatch.setenv(key, value)

        sys.modules.pop("main", None)
        import main as app_main

        app_main.PATHS.CONFIG_DIR = config
        app_main.PATHS.SETTINGS_FILE = config / "settings.yml"
        app_main._registered_oidc_config = None
        app_main._config_mtimes = None
        app_main._ocr_languages_cache = None
        app_main.oauth._clients.clear()
        app_main.oauth._registry.clear()
        return app_main, config / "settings.yml"

    return _load


@pytest.fixture
def patch_oidc_transport(monkeypatch):
    def _patch(provider: MockOIDCProvider):
        orig_init = httpx.AsyncClient.__init__
        transport = provider.make_httpx_transport()

        def patched_init(self, *args, **kwargs):
            kwargs.setdefault("transport", transport)
            return orig_init(self, *args, **kwargs)

        monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)

    return _patch
