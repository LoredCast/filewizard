"""Tests for settings loading, validation, secret redaction, and repository config files."""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from filewizard.settings import deep_merge, drop_blank_secrets, validate_settings_payload

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_deep_merge_preserves_nested_defaults():
    dest = {"auth_settings": {"oidc_pkce": True, "oidc_client_id": ""}}
    merged = deep_merge({"auth_settings": {"oidc_client_id": "my-app"}}, dest)
    assert merged == {"auth_settings": {"oidc_pkce": True, "oidc_client_id": "my-app"}}


def test_drop_blank_secrets_only_removes_empty_secrets():
    payload = {
        "auth_settings": {"oidc_client_secret": "   ", "oidc_client_id": "abc"},
        "webhook_settings": {"callback_bearer_token": "tok-123"},
    }
    drop_blank_secrets(payload)
    assert "oidc_client_secret" not in payload["auth_settings"]
    assert payload["webhook_settings"]["callback_bearer_token"] == "tok-123"


def test_validate_settings_payload_enforces_schema_types_and_command_lock():
    errs = validate_settings_payload(
        {
            "app_settings": {"max_file_size_mb": -5, "retention_days": -1},
            "auth_settings": {"oidc_pkce": "yes", "oidc_userinfo_endpoint": "https://evil.test"},
            "conversion_tools": {
                "new_tool": {"command_template": "echo {input}"},
            },
        },
        current_tools={},
        allow_command_edits=False,
        ocr_language_validator=lambda lang, _cfg: lang,
        command_validator=lambda tmpl, _m: [tmpl],
    )
    joined = "\n".join(errs)
    assert "max_file_size_mb" in joined
    assert "retention_days" in joined
    assert "oidc_pkce" in joined
    assert "oidc_userinfo_endpoint" in joined
    assert "Adding conversion tools" in joined


def test_local_only_settings_page_and_save_preserves_secret(load_app):
    app_main, settings_file = load_app(local_only=True)
    settings_file.write_text(
        yaml.safe_dump({"auth_settings": {"oidc_client_secret": "existing-secret"}}),
        encoding="utf8",
    )

    with TestClient(app_main.app, base_url="http://testserver") as client:
        resp = client.get("/settings")
        assert resp.status_code == 200
        assert "existing-secret" not in resp.text
        assert re.search(r'id="auth-oidc-pkce"[^>]*checked', resp.text)

        save_resp = client.post(
            "/settings/save",
            json={
                "app_settings": {"retention_days": 14},
                "auth_settings": {"oidc_client_secret": "", "oidc_pkce": False},
            },
            headers={"Origin": "http://testserver"},
        )
        assert save_resp.status_code == 200
        saved = yaml.safe_load(settings_file.read_text(encoding="utf8"))
        assert saved["auth_settings"]["oidc_client_secret"] == "existing-secret"
        assert saved["auth_settings"]["oidc_pkce"] is False
        assert saved["app_settings"]["retention_days"] == 14


def test_repo_default_settings_and_unraid_templates_are_valid():
    cfg = yaml.safe_load((REPO_ROOT / "settings.default.yml").read_text(encoding="utf8"))
    assert cfg["auth_settings"]["oidc_pkce"] is True
    assert cfg["auth_settings"]["oidc_client_id"] == ""
    assert cfg["auth_settings"]["oidc_server_metadata_url"] == ""
    assert isinstance(cfg["conversion_tools"], dict) and len(cfg["conversion_tools"]) > 5

    for xml_path in (REPO_ROOT / "unraid").glob("*.xml"):
        root = ET.fromstring(xml_path.read_text(encoding="utf8"))
        assert root.tag == "Container"
