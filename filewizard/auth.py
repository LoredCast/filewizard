"""OIDC configuration, PKCE, session compaction, and user authorization helpers."""

import json
import logging
from typing import Any, Optional
from urllib.parse import urlparse

from starlette.requests import Request

logger = logging.getLogger(__name__)

# Untouched example values from older settings.default.yml files must not be treated
# as a real OIDC provider now that client_secret is optional for PKCE public clients.
PLACEHOLDER_OIDC_URLS = frozenset({
    "https://accounts.example.com/oidc/.well-known/openid-configuration",
    "https://accounts.example.com/oidc/me",
    "https://accounts.example.com/oidc/session/end",
})

# Keep signed SessionMiddleware cookies comfortably under the 4096-byte browser limit
# after base64 + timestamp/HMAC + cookie attribute overhead.
MAX_SESSION_JSON_BYTES = 2500


def _clean_oidc_url(value: Any) -> Optional[str]:
    text = str(value).strip() if isinstance(value, str) else ""
    if not text or text in PLACEHOLDER_OIDC_URLS:
        return None
    return text


def oidc_pkce_enabled(oidc_cfg: Optional[dict]) -> bool:
    """Returns whether PKCE (S256) should be used for OIDC login. Enabled by default."""
    value = (oidc_cfg or {}).get("oidc_pkce", True)
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in ("false", "0", "f", "no", "n", "off")
    return bool(value)


def oidc_config_signature(oidc_cfg: Optional[dict]) -> tuple:
    cfg = oidc_cfg or {}
    return (
        (cfg.get("oidc_client_id") or "").strip(),
        (cfg.get("oidc_client_secret") or "").strip(),
        _clean_oidc_url(cfg.get("oidc_server_metadata_url")),
        _clean_oidc_url(cfg.get("oidc_userinfo_endpoint")),
        _clean_oidc_url(cfg.get("oidc_end_session_endpoint")),
        oidc_pkce_enabled(cfg),
    )


def oidc_settings_complete(oidc_cfg: Optional[dict]) -> bool:
    """
    OIDC requires client_id and a non-placeholder server_metadata_url.
    client_secret is optional when PKCE is enabled (public client) and required when PKCE is off.
    """
    client_id, client_secret, metadata_url, _userinfo, _logout, use_pkce = oidc_config_signature(oidc_cfg)
    if not client_id or not metadata_url:
        return False
    if not use_pkce and not client_secret:
        return False
    return True


def register_oidc_client(oauth, oidc_cfg: Optional[dict]) -> bool:
    """
    Re-registers Authlib's 'oidc' client from oidc_cfg.
    Returns True if registration succeeded, False if settings are incomplete or registration failed.
    """
    # Authlib caches created client instances in oauth._clients; evict the old one so
    # changes to auth_settings (such as toggling PKCE) take effect without a restart.
    oauth._clients.pop("oidc", None)
    oauth._registry.pop("oidc", None)

    client_id, client_secret, metadata_url, userinfo_url, end_session_url, use_pkce = oidc_config_signature(oidc_cfg)
    if not oidc_settings_complete(oidc_cfg):
        logger.warning("OIDC auth settings are incomplete. Auth will be disabled.")
        return False

    client_kwargs = {"scope": "openid email profile"}
    if use_pkce:
        client_kwargs["code_challenge_method"] = "S256"

    register_kwargs: dict[str, Any] = {
        "name": "oidc",
        "client_id": client_id,
        "client_secret": client_secret or None,
        "server_metadata_url": metadata_url,
        "client_kwargs": client_kwargs,
    }
    if userinfo_url:
        register_kwargs["userinfo_endpoint"] = userinfo_url
    if end_session_url:
        register_kwargs["end_session_endpoint"] = end_session_url

    try:
        oauth.register(**register_kwargs)
        mode = "confidential" if client_secret else "public"
        logger.info("OAuth registered successfully (%s client, PKCE: %s).", mode, "S256" if use_pkce else "disabled")
        return True
    except Exception as e:
        logger.error(f"Failed to register OIDC OAuth provider: {e}. Authentication will be disabled.")
        return False


def merge_oidc_user_claims(id_token_claims: Optional[dict], userinfo_claims: Optional[dict]) -> dict:
    """Merges id_token and userinfo claims without carrying email_verified across different emails."""
    merged = dict(id_token_claims or {})
    userinfo = dict(userinfo_claims or {})
    if "email" in userinfo and "email" in merged and userinfo["email"] != merged["email"] and "email_verified" not in userinfo:
        merged.pop("email_verified", None)
    merged.update(userinfo)
    return merged


def compact_session_user(claims: Optional[dict]) -> dict:
    """
    Keeps only the OIDC claims FileWizard uses so the signed session cookie stays small
    and extra IdP claims (groups, roles, etc.) are not embedded in the cookie or HTML.
    """
    claims = claims or {}
    sub = claims.get("sub")
    email = claims.get("email")
    name = claims.get("name") or claims.get("preferred_username") or email or sub or ""
    compact: dict[str, Any] = {"sub": sub, "name": name}
    if email is not None:
        compact["email"] = email
    if "email_verified" in claims:
        compact["email_verified"] = claims["email_verified"]
    return compact


def fit_session_id_token(user: dict, id_token: Optional[str], max_json_bytes: int = MAX_SESSION_JSON_BYTES) -> Optional[str]:
    """
    Returns id_token only if storing it alongside user keeps the session JSON small enough
    for the signed session cookie to stay below browser 4 KB limits.
    """
    if not id_token or not isinstance(id_token, str):
        return None
    encoded = json.dumps({"user": user, "id_token": id_token}, separators=(",", ":")).encode("utf-8")
    if len(encoded) > max_json_bytes:
        logger.warning(
            "OIDC id_token (%d bytes) omitted from session cookie to stay within browser cookie limits; "
            "RP-initiated logout will use client_id instead.",
            len(id_token),
        )
        return None
    return id_token


def external_url(request: Request, route_name: str, app_public_url: Optional[str]) -> str:
    """
    Builds an absolute URL for a named route, preferring app_public_url when configured so
    OIDC redirect_uri and post_logout_redirect_uri work behind reverse proxies.
    """
    public = (app_public_url or "").strip()
    if public:
        parsed = urlparse(public)
        if parsed.scheme in ("http", "https") and parsed.netloc:
            route_path = request.app.url_path_for(route_name)
            return f"{public.rstrip('/')}/{route_path.lstrip('/')}"
    return str(request.url_for(route_name))


def normalized_email_set(values) -> set:
    if not isinstance(values, (list, tuple, set)):
        return set()
    return {v.strip().lower() for v in values if isinstance(v, str) and v.strip()}


def verified_email(user: Optional[dict]) -> Optional[str]:
    """The user's email, unless the identity provider explicitly marked it as unverified."""
    if not user:
        return None
    email = user.get("email")
    if not isinstance(email, str) or not email.strip() or user.get("email_verified") is False:
        return None
    return email.strip().lower()


def is_user_allowed(user: Optional[dict], auth_settings: Optional[dict], local_only_mode: bool = False) -> bool:
    """
    Authorization for OIDC users. With auth_settings.allowed_users and allowed_domains both
    empty, every account the identity provider authenticates may use the app.
    """
    if local_only_mode:
        return True
    if not user:
        return False
    auth_settings = auth_settings or {}
    allowed_users = normalized_email_set(auth_settings.get("allowed_users"))
    allowed_domains = {d.lstrip("@") for d in normalized_email_set(auth_settings.get("allowed_domains"))}
    if not allowed_users and not allowed_domains:
        return True
    email = verified_email(user)
    if not email:
        return False
    if email in allowed_users or email in normalized_email_set(auth_settings.get("admin_users")):
        return True
    return email.rsplit("@", 1)[-1] in allowed_domains


def is_user_admin(user: Optional[dict], auth_settings: Optional[dict], local_only_mode: bool = False) -> bool:
    if local_only_mode:
        return True
    email = verified_email(user)
    if not email:
        return False
    return email in normalized_email_set((auth_settings or {}).get("admin_users", []))
