"""Unit and integration tests for security, path, archive, command, and CSRF protections."""

import zipfile

import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import Headers

from filewizard.security import (
    ensure_path_is_safe,
    is_csrf_safe,
    resolve_whisper_model,
    safe_basename,
    safe_extract_zip,
    sanitize_filename,
    sanitize_output,
    validate_and_build_command,
    validate_file_type,
    validate_tts_model_name,
)


def test_ensure_path_is_safe_blocks_traversal(tmp_path):
    allowed = tmp_path / "uploads"
    allowed.mkdir()
    good = allowed / "file.pdf"
    assert ensure_path_is_safe(good, [allowed]) == good.resolve()

    with pytest.raises(ValueError):
        ensure_path_is_safe(allowed / ".." / "secret.txt", [allowed])


def test_safe_basename_preserves_extension_for_non_ascii_and_strips_dirs():
    assert safe_basename("文件.pdf") == "file.pdf"
    assert safe_basename("../../etc/passwd") == "passwd"
    assert safe_basename("my report (final).DOCX") == "my_report_final.docx"
    assert safe_basename("bad.ext_too_long_suffix") == "bad.ext_too_long_suffix"
    assert safe_basename("") == "file"


def test_sanitize_helpers():
    assert "&lt;script&gt;" in sanitize_output("<script>alert(1)</script>")
    assert len(sanitize_output("a" * 5000)) == 2000
    assert ".." not in sanitize_filename("../../evil.pdf")


def test_safe_extract_zip_filters_junk_symlinks_and_enforces_limits(tmp_path):
    zpath = tmp_path / "archive.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("doc.txt", "hello")
        zf.writestr("__MACOSX/._doc.txt", "junk")
        zf.writestr(".DS_Store", "junk")
        link_info = zipfile.ZipInfo("evil_link")
        link_info.external_attr = (0o120777 << 16)
        zf.writestr(link_info, "/etc/passwd")

    dest = tmp_path / "out"
    extracted = safe_extract_zip(zpath, dest, {"max_zip_members": 10, "max_zip_uncompressed_mb": 10})
    assert [p.name for p in extracted] == ["doc.txt"]

    # Uncompressed size limit (zip bomb protection)
    with pytest.raises(ValueError, match="max_zip_uncompressed_mb"):
        safe_extract_zip(zpath, tmp_path / "out2", {"max_zip_members": 10, "max_zip_uncompressed_mb": 0.000001})

    # Member count limit
    many = tmp_path / "many.zip"
    with zipfile.ZipFile(many, "w") as zf:
        for i in range(4):
            zf.writestr(f"f{i}.txt", "x")
    with pytest.raises(ValueError, match="limit is 2"):
        safe_extract_zip(many, tmp_path / "out3", {"max_zip_members": 2})


def test_model_and_file_type_validators():
    assert resolve_whisper_model("tiny", {"allowed_models": ["tiny", "base"]}) == "tiny"
    assert resolve_whisper_model(None, {"allowed_models": ["tiny", "base"]}) == "base"
    with pytest.raises(ValueError):
        resolve_whisper_model("/etc/passwd", {"allowed_models": ["tiny", "base"]})

    assert validate_tts_model_name("piper/en_US-lessac-medium") == "piper/en_US-lessac-medium"
    assert validate_tts_model_name("kokoro/en-us/af_bella") == "kokoro/en-us/af_bella"
    with pytest.raises(ValueError):
        validate_tts_model_name("piper/../../etc/passwd")
    with pytest.raises(ValueError):
        validate_tts_model_name("unknown/voice")

    assert validate_file_type("photo.JPG", {".jpg", ".png"}) is True
    assert validate_file_type("script.sh", {".jpg", ".png"}) is False
    assert validate_file_type("anything.xyz", set()) is True


def test_validate_and_build_command_prevents_argument_injection():
    argv = validate_and_build_command(
        "pandoc {input} -o {output}",
        {"input": "/tmp/file with spaces; rm -rf /.md", "output": "/tmp/out.pdf"},
    )
    assert argv == ["pandoc", "/tmp/file with spaces; rm -rf /.md", "-o", "/tmp/out.pdf"]

    with pytest.raises(ValueError, match="disallowed placeholders"):
        validate_and_build_command("pandoc {evil} {input}", {"input": "a.md"})


def test_csrf_helper_and_security_headers(load_app):
    assert is_csrf_safe("GET", Headers({"host": "app.local", "origin": "https://evil.test"}), None, []) is True
    assert is_csrf_safe("POST", Headers({"host": "app.local"}), None, []) is True
    assert is_csrf_safe("POST", Headers({"host": "app.local", "origin": "null"}), None, []) is False
    assert is_csrf_safe("POST", Headers({"host": "app.local", "origin": "https://evil.test"}), None, []) is False
    assert is_csrf_safe("POST", Headers({"host": "app.local:80", "origin": "http://app.local"}), None, []) is True
    assert is_csrf_safe(
        "POST",
        Headers({"host": "internal:8000", "origin": "https://files.example.com"}),
        "https://files.example.com",
        [],
    ) is True

    app_main, _ = load_app(local_only=True)
    with TestClient(app_main.app, base_url="http://testserver") as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.headers["x-content-type-options"] == "nosniff"
        assert resp.headers["x-frame-options"] == "SAMEORIGIN"
        assert "default-src 'self'" in resp.headers["content-security-policy"]

        blocked = client.post(
            "/settings/save",
            json={},
            headers={"Origin": "https://evil.example"},
        )
        assert blocked.status_code == 403
