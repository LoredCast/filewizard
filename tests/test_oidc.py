"""End-to-end and unit tests for OIDC PKCE, public clients, session compaction, and API auth."""

import json
import os
import re
from urllib.parse import parse_qs, urlparse

import yaml
from fastapi.testclient import TestClient

from conftest import MockOIDCProvider
from filewizard.auth import (
    compact_session_user,
    fit_session_id_token,
    is_user_allowed,
    merge_oidc_user_claims,
    oidc_pkce_enabled,
    oidc_settings_complete,
    verified_email,
)


def test_pkce_enabled_by_default_and_end_to_end_login(load_app, patch_oidc_transport):
    app_main, settings_file = load_app(local_only=False)
    provider = MockOIDCProvider(require_pkce=True)
    patch_oidc_transport(provider)

    # Deliberately omit oidc_pkce to verify existing configs upgrade with PKCE S256 enabled
    settings_file.write_text(
        yaml.safe_dump(
            {
                "auth_settings": {
                    "oidc_client_id": provider.client_id,
                    "oidc_client_secret": provider.client_secret,
                    "oidc_server_metadata_url": provider.metadata_url,
                    "admin_users": ["admin@example.com"],
                }
            }
        ),
        encoding="utf8",
    )

    with TestClient(app_main.app, base_url="http://testserver") as client:
        login_resp = client.get("/login", follow_redirects=False)
        assert login_resp.status_code == 302
        location = login_resp.headers["location"]
        assert location.startswith(f"{provider.issuer}/authorize?")
        assert len(login_resp.headers.get("set-cookie", "")) < 2048

        code, state = provider.issue_code_from_authorize_url(location)
        auth_resp = client.get(f"/auth?code={code}&state={state}", follow_redirects=False)
        assert auth_resp.status_code == 307
        assert auth_resp.headers["location"] == "/"

        assert len(provider.token_requests) == 1
        verifier = provider.token_requests[0]["body"]["code_verifier"]
        assert 43 <= len(verifier) <= 128

        # Embedded page user JSON only contains compacted fields (no IdP groups/roles leaked)
        index_resp = client.get("/")
        assert index_resp.status_code == 200
        match = re.search(r'<script type="application/json" id="app-config">(.*?)</script>', index_resp.text)
        assert match
        embedded_user = json.loads(match.group(1))["user"]
        assert embedded_user["sub"] == "user-123"
        assert embedded_user["email"] == "admin@example.com"
        assert "groups" not in embedded_user

        # /api/authz/forward-auth also sends PKCE S256
        fwd_resp = client.get("/api/authz/forward-auth", follow_redirects=False)
        assert fwd_resp.status_code == 302
        provider.issue_code_from_authorize_url(fwd_resp.headers["location"])

        # /settings renders checkbox checked
        settings_page = client.get("/settings")
        assert settings_page.status_code == 200
        assert re.search(r'id="auth-oidc-pkce"[^>]*checked', settings_page.text)

        # /logout uses RP-initiated logout with id_token_hint
        logout_resp = client.get("/logout", follow_redirects=False)
        assert logout_resp.status_code == 302
        assert logout_resp.headers["location"].startswith(f"{provider.issuer}/logout?")
        assert "id_token_hint=" in logout_resp.headers["location"]


def test_public_pkce_client_without_client_secret(load_app, patch_oidc_transport):
    app_main, settings_file = load_app(local_only=False)
    provider = MockOIDCProvider(require_pkce=True, public_client=True)
    patch_oidc_transport(provider)

    settings_file.write_text(
        yaml.safe_dump(
            {
                "auth_settings": {
                    "oidc_client_id": provider.client_id,
                    "oidc_client_secret": "",
                    "oidc_server_metadata_url": provider.metadata_url,
                    "oidc_pkce": True,
                }
            }
        ),
        encoding="utf8",
    )

    with TestClient(app_main.app, base_url="http://testserver") as client:
        login_resp = client.get("/login", follow_redirects=False)
        assert login_resp.status_code == 302
        code, state = provider.issue_code_from_authorize_url(login_resp.headers["location"])
        auth_resp = client.get(f"/auth?code={code}&state={state}", follow_redirects=False)
        assert auth_resp.status_code == 307
        assert provider.token_requests[0]["body"]["client_id"] == provider.client_id
        assert "authorization" not in {k.lower() for k in provider.token_requests[0]["headers"]}


def test_login_falls_back_to_id_token_when_userinfo_endpoint_omitted(load_app, patch_oidc_transport):
    app_main, settings_file = load_app(local_only=False)
    provider = MockOIDCProvider(require_pkce=True, expose_userinfo=False)
    patch_oidc_transport(provider)

    settings_file.write_text(
        yaml.safe_dump(
            {
                "auth_settings": {
                    "oidc_client_id": provider.client_id,
                    "oidc_client_secret": provider.client_secret,
                    "oidc_server_metadata_url": provider.metadata_url,
                }
            }
        ),
        encoding="utf8",
    )

    with TestClient(app_main.app, base_url="http://testserver") as client:
        login_resp = client.get("/login", follow_redirects=False)
        code, state = provider.issue_code_from_authorize_url(login_resp.headers["location"])
        auth_resp = client.get(f"/auth?code={code}&state={state}", follow_redirects=False)
        assert auth_resp.status_code == 307
        assert "Logged in as Test Admin" in client.get("/").text


def test_toggle_pkce_off_via_settings_and_hot_reload_back_on(load_app, patch_oidc_transport):
    app_main, settings_file = load_app(local_only=False)
    provider = MockOIDCProvider(require_pkce=True)
    patch_oidc_transport(provider)

    settings_file.write_text(
        yaml.safe_dump(
            {
                "auth_settings": {
                    "oidc_client_id": provider.client_id,
                    "oidc_client_secret": provider.client_secret,
                    "oidc_server_metadata_url": provider.metadata_url,
                    "admin_users": ["admin@example.com"],
                }
            }
        ),
        encoding="utf8",
    )

    with TestClient(app_main.app, base_url="http://testserver") as client:
        login_resp = client.get("/login", follow_redirects=False)
        code, state = provider.issue_code_from_authorize_url(login_resp.headers["location"])
        assert client.get(f"/auth?code={code}&state={state}", follow_redirects=False).status_code == 307

        # Disable PKCE via /settings/save (takes effect immediately without restart)
        save_resp = client.post(
            "/settings/save",
            json={"auth_settings": {"oidc_pkce": False}},
            headers={"Origin": "http://testserver"},
        )
        assert save_resp.status_code == 200
        assert not re.search(r'id="auth-oidc-pkce"[^>]*checked', client.get("/settings").text)

        provider.require_pkce = False
        provider.reject_pkce = True
        provider.token_requests.clear()

        login_resp2 = client.get("/login", follow_redirects=False)
        qs2 = parse_qs(urlparse(login_resp2.headers["location"]).query)
        assert "code_challenge" not in qs2
        assert "code_challenge_method" not in qs2

        code2, state2 = provider.issue_code_from_authorize_url(login_resp2.headers["location"])
        assert client.get(f"/auth?code={code2}&state={state2}", follow_redirects=False).status_code == 307
        assert "code_verifier" not in provider.token_requests[0]["body"]

        # Disk edit hot-reloads PKCE back on
        disk_cfg = yaml.safe_load(settings_file.read_text(encoding="utf8"))
        disk_cfg["auth_settings"]["oidc_pkce"] = True
        settings_file.write_text(yaml.safe_dump(disk_cfg), encoding="utf8")
        st = settings_file.stat()
        os.utime(settings_file, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))

        provider.require_pkce = True
        provider.reject_pkce = False
        login_resp3 = client.get("/login", follow_redirects=False)
        qs3 = parse_qs(urlparse(login_resp3.headers["location"]).query)
        assert qs3.get("code_challenge_method") == ["S256"]
        assert "code_challenge" in qs3


def test_app_public_url_used_for_oidc_redirect_and_logout_uris(load_app, patch_oidc_transport):
    app_main, settings_file = load_app(local_only=False)
    provider = MockOIDCProvider(require_pkce=True)
    patch_oidc_transport(provider)

    settings_file.write_text(
        yaml.safe_dump(
            {
                "app_settings": {"app_public_url": "https://files.example.org/wizard/"},
                "auth_settings": {
                    "oidc_client_id": provider.client_id,
                    "oidc_client_secret": provider.client_secret,
                    "oidc_server_metadata_url": provider.metadata_url,
                },
            }
        ),
        encoding="utf8",
    )

    with TestClient(app_main.app, base_url="http://internal:8000") as client:
        login_resp = client.get("/login", follow_redirects=False)
        qs = parse_qs(urlparse(login_resp.headers["location"]).query)
        assert qs["redirect_uri"] == ["https://files.example.org/wizard/auth"]

        code, state = provider.issue_code_from_authorize_url(login_resp.headers["location"])
        assert client.get(f"/auth?code={code}&state={state}", follow_redirects=False).status_code == 307

        logout_resp = client.get("/logout", follow_redirects=False)
        logout_qs = parse_qs(urlparse(logout_resp.headers["location"]).query)
        assert logout_qs["post_logout_redirect_uri"] == ["https://files.example.org/wizard/"]


def test_oversized_id_token_does_not_break_login_cookie(load_app, patch_oidc_transport):
    app_main, settings_file = load_app(local_only=False)
    # Huge IdP claim list pushing id_token past 3 KB
    provider = MockOIDCProvider(
        require_pkce=True,
        extra_id_token_claims={"groups": [f"enterprise-role-{i:04d}" for i in range(180)]},
    )
    patch_oidc_transport(provider)

    settings_file.write_text(
        yaml.safe_dump(
            {
                "auth_settings": {
                    "oidc_client_id": provider.client_id,
                    "oidc_client_secret": provider.client_secret,
                    "oidc_server_metadata_url": provider.metadata_url,
                }
            }
        ),
        encoding="utf8",
    )

    with TestClient(app_main.app, base_url="http://testserver") as client:
        login_resp = client.get("/login", follow_redirects=False)
        code, state = provider.issue_code_from_authorize_url(login_resp.headers["location"])
        auth_resp = client.get(f"/auth?code={code}&state={state}", follow_redirects=False)
        assert auth_resp.status_code == 307
        assert len(auth_resp.headers.get("set-cookie", "")) < 4096

        # Logout still works via client_id without id_token_hint
        logout_resp = client.get("/logout", follow_redirects=False)
        assert logout_resp.status_code == 302
        logout_qs = parse_qs(urlparse(logout_resp.headers["location"]).query)
        assert "id_token_hint" not in logout_qs
        assert logout_qs["client_id"] == [provider.client_id]


def test_pkce_mismatch_rejected_and_placeholder_urls_not_active(load_app, patch_oidc_transport):
    app_main, settings_file = load_app(local_only=False)
    provider = MockOIDCProvider(require_pkce=True)
    patch_oidc_transport(provider)

    # Old untouched settings.default.yml placeholder must remain disabled even with PKCE on
    settings_file.write_text(
        yaml.safe_dump(
            {
                "auth_settings": {
                    "oidc_client_id": "filewiz",
                    "oidc_client_secret": None,
                    "oidc_server_metadata_url": "https://accounts.example.com/oidc/.well-known/openid-configuration",
                }
            }
        ),
        encoding="utf8",
    )

    with TestClient(app_main.app, base_url="http://testserver") as client:
        resp = client.get("/login", follow_redirects=False)
        assert resp.headers["location"] == "/?error=auth_not_configured"

        # Configure real IdP on disk -> auto-recovers on next request
        settings_file.write_text(
            yaml.safe_dump(
                {
                    "auth_settings": {
                        "oidc_client_id": provider.client_id,
                        "oidc_client_secret": provider.client_secret,
                        "oidc_server_metadata_url": provider.metadata_url,
                    }
                }
            ),
            encoding="utf8",
        )
        st = settings_file.stat()
        os.utime(settings_file, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))

        login_resp = client.get("/login", follow_redirects=False)
        code, state = provider.issue_code_from_authorize_url(login_resp.headers["location"])
        provider.codes[code]["code_challenge"] = "A" * 43
        assert client.get(f"/auth?code={code}&state={state}", follow_redirects=False).status_code == 401


def test_bearer_token_routes_accept_oidc_bearer_tokens(load_app, patch_oidc_transport):
    app_main, settings_file = load_app(local_only=False)
    provider = MockOIDCProvider(require_pkce=True)
    patch_oidc_transport(provider)

    settings_file.write_text(
        yaml.safe_dump(
            {
                "auth_settings": {
                    "oidc_client_id": provider.client_id,
                    "oidc_client_secret": provider.client_secret,
                    "oidc_server_metadata_url": provider.metadata_url,
                }
            }
        ),
        encoding="utf8",
    )

    with TestClient(app_main.app, base_url="http://testserver") as client:
        # Unauthenticated calls fail with 401
        assert client.get("/api/v1/supported-formats/.pdf").status_code == 401
        assert client.get("/download/zip-batch/nonexistent").status_code == 401
        assert client.post("/download/batch", json={"job_ids": []}).status_code == 401

        # With valid OIDC bearer token, routes authenticate and reach business logic
        headers = {"Authorization": f"Bearer {provider.access_token}"}
        assert client.get("/api/v1/supported-formats/.pdf", headers=headers).status_code == 200
        assert client.get("/download/zip-batch/nonexistent", headers=headers).status_code == 404
        assert client.post("/download/batch", json={"job_ids": []}, headers=headers).status_code == 400


def test_auth_helper_unit_edge_cases():
    assert oidc_pkce_enabled({}) is True
    assert oidc_pkce_enabled({"oidc_pkce": None}) is True
    assert oidc_pkce_enabled({"oidc_pkce": False}) is False
    assert oidc_pkce_enabled({"oidc_pkce": "false"}) is False
    assert oidc_pkce_enabled({"oidc_pkce": "0"}) is False
    assert oidc_pkce_enabled({"oidc_pkce": "true"}) is True

    assert oidc_settings_complete({"oidc_client_id": "id", "oidc_server_metadata_url": "https://idp.test/.well-known/openid-configuration", "oidc_pkce": True}) is True
    assert oidc_settings_complete({"oidc_client_id": "id", "oidc_server_metadata_url": "https://idp.test/.well-known/openid-configuration", "oidc_pkce": False}) is False
    assert oidc_settings_complete({"oidc_client_id": "filewiz", "oidc_server_metadata_url": "https://accounts.example.com/oidc/.well-known/openid-configuration"}) is False

    # Unverified emails never pass allowlist or verified_email
    assert verified_email({"email": "a@example.com", "email_verified": False}) is None
    assert is_user_allowed({"sub": "1", "email": "a@example.com", "email_verified": False}, {"allowed_users": ["a@example.com"]}) is False
    assert is_user_allowed({"sub": "1", "email": "a@example.com", "email_verified": True}, {"allowed_domains": ["@example.com"]}) is True
    assert is_user_allowed({"sub": "1", "email": "a@other.com", "email_verified": True}, {"allowed_domains": ["example.com"]}) is False

    merged = merge_oidc_user_claims({"sub": "1", "email": "a@example.com", "email_verified": True}, {"email": "b@example.com"})
    assert "email_verified" not in merged
    compact = compact_session_user({"sub": "1", "preferred_username": "alice", "groups": ["a", "b"]})
    assert compact == {"sub": "1", "name": "alice"}
    assert fit_session_id_token(compact, "x" * 3000) is None
