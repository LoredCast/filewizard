"""Settings merging, preprocessing, and validation helpers."""

import re
from typing import Any, Callable, Dict, List

# Secrets are never sent to the browser; submitting an empty value keeps the stored one.
SECRET_SETTINGS = (("auth_settings", "oidc_client_secret"), ("webhook_settings", "callback_bearer_token"))


def _leaves(*names: str) -> dict:
    return {name: None for name in names}


# Everything the settings form may change. Other keys (TTS command templates, model paths, ...)
# can only be edited in settings.yml.
EDITABLE_SETTINGS = {
    "app_settings": _leaves(
        "app_public_url", "max_file_size_mb", "allowed_all_extensions",
        "model_concurrency", "model_inactivity_timeout", "cache_check_interval",
        "retention_days",
    ),
    "ocr_settings": {"ocrmypdf": _leaves("deskew", "clean", "force_ocr", "language")},
    "transcription_settings": {"whisper": _leaves("compute_type")},
    "auth_settings": _leaves(
        "oidc_client_id", "oidc_client_secret", "oidc_server_metadata_url",
        "oidc_pkce", "admin_users", "allowed_users", "allowed_domains",
    ),
    "webhook_settings": _leaves("enabled", "allow_chunked_api_uploads", "allowed_callback_urls", "callback_bearer_token"),
    "tts_settings": {"piper": {"use_cuda": None, "synthesis_config": _leaves("length_scale", "noise_scale", "noise_w")}},
}
EDITABLE_TOOL_KEYS = {"name", "command_template", "supported_input", "formats", "timeout"}
POSITIVE_NUMBER_SETTINGS = (
    ("app_settings", "max_file_size_mb"),
    ("app_settings", "model_concurrency"),
    ("app_settings", "model_inactivity_timeout"),
    ("app_settings", "cache_check_interval"),
)


def deep_merge(source: dict, dest: dict) -> dict:
    """Recursively merges source dict into dest dict. Modifies dest in place."""
    for key, value in source.items():
        if isinstance(value, dict) and key in dest and isinstance(dest[key], dict):
            deep_merge(value, dest[key])
        else:
            dest[key] = value
    return dest


def _collapse_whitespace(value: Any) -> str:
    return " ".join(str(value or "").split())


def drop_blank_secrets(payload: dict) -> None:
    for section, key in SECRET_SETTINGS:
        section_data = payload.get(section)
        if isinstance(section_data, dict) and key in section_data:
            value = section_data[key]
            if value is None or (isinstance(value, str) and not value.strip()):
                del section_data[key]


def preprocess_settings_for_saving(config: Dict) -> Dict:
    """
    Pre-processes the settings dictionary before saving it to YAML.
    Corrects data structures that may be misinterpreted when converted from JSON.
    """
    if "conversion_tools" in config and isinstance(config["conversion_tools"], dict):
        for _tool_name, tool_config in config["conversion_tools"].items():
            if isinstance(tool_config, dict):
                if "command_template" in tool_config:
                    ct = tool_config["command_template"]
                    if isinstance(ct, list) and len(ct) == 1 and isinstance(ct[0], str):
                        tool_config["command_template"] = ct[0]
                    elif isinstance(ct, str):
                        tool_config["command_template"] = ct.replace("\n", "\n")

                if "formats" in tool_config:
                    formats_data = tool_config["formats"]
                    if isinstance(formats_data, list):
                        new_formats = {}
                        for item in formats_data:
                            if isinstance(item, str):
                                for line in item.split("\n"):
                                    line = line.strip()
                                    if ":" in line:
                                        key, value = line.split(":", 1)
                                        new_formats[key.strip()] = value.strip()
                        tool_config["formats"] = new_formats
                    elif isinstance(formats_data, str):
                        new_formats = {}
                        for line in formats_data.split("\n"):
                            line = line.strip()
                            if ":" in line:
                                key, value = line.split(":", 1)
                                new_formats[key.strip()] = value.strip()
                        tool_config["formats"] = new_formats

    return config


def validate_settings_payload(
    payload: dict,
    current_tools: dict,
    allow_command_edits: bool,
    ocr_language_validator: Callable[[str, dict], str],
    command_validator: Callable[[str, dict], list],
) -> List[str]:
    """Returns a list of problems with a settings update coming from the web UI."""
    errors: List[str] = []

    def walk(data: dict, schema: dict, path: tuple):
        for key, value in data.items():
            key_path = ".".join(path + (key,))
            if key not in schema:
                errors.append(f"'{key_path}' cannot be changed from the web UI; edit settings.yml instead.")
                continue
            sub_schema = schema[key]
            if sub_schema is None:
                continue
            if not isinstance(value, dict):
                errors.append(f"'{key_path}' must be an object.")
                continue
            walk(value, sub_schema, path + (key,))

    walk({k: v for k, v in payload.items() if k != "conversion_tools"}, EDITABLE_SETTINGS, ())

    for section, key in POSITIVE_NUMBER_SETTINGS:
        section_data = payload.get(section)
        if isinstance(section_data, dict) and key in section_data:
            try:
                valid = float(section_data[key]) > 0
            except (TypeError, ValueError):
                valid = False
            if not valid:
                errors.append(f"'{section}.{key}' must be a positive number.")

    app_section = payload.get("app_settings")
    if isinstance(app_section, dict) and "retention_days" in app_section:
        try:
            valid = app_section["retention_days"] is None or float(app_section["retention_days"]) >= 0
        except (TypeError, ValueError):
            valid = False
        if not valid:
            errors.append("'app_settings.retention_days' must be 0 (keep forever) or a positive number.")

    auth_section = payload.get("auth_settings")
    if isinstance(auth_section, dict) and "oidc_pkce" in auth_section:
        if not isinstance(auth_section["oidc_pkce"], bool):
            errors.append("'auth_settings.oidc_pkce' must be true or false.")

    ocr_language = (payload.get("ocr_settings") or {}).get("ocrmypdf", {}) if isinstance(payload.get("ocr_settings"), dict) else {}
    if isinstance(ocr_language, dict) and "language" in ocr_language:
        try:
            ocr_language_validator(str(ocr_language["language"] or ""), {})
        except ValueError as e:
            errors.append(f"'ocr_settings.ocrmypdf.language': {e}")

    tools = payload.get("conversion_tools")
    if tools is None:
        return errors
    if not isinstance(tools, dict):
        errors.append("'conversion_tools' must be an object.")
        return errors
    for tool_id, tool in tools.items():
        if not isinstance(tool, dict):
            errors.append(f"'conversion_tools.{tool_id}' must be an object.")
            continue
        if not re.fullmatch(r"[A-Za-z0-9_]{1,64}", str(tool_id)):
            errors.append(f"Invalid tool id '{tool_id}' (letters, digits and underscores only).")
            continue
        for key in sorted(set(tool) - EDITABLE_TOOL_KEYS):
            errors.append(f"'conversion_tools.{tool_id}.{key}' cannot be changed from the web UI; edit settings.yml instead.")
        if "timeout" in tool:
            try:
                valid_timeout = 0 < float(tool["timeout"]) <= 7 * 24 * 3600
            except (TypeError, ValueError):
                valid_timeout = False
            if not valid_timeout:
                errors.append(f"'conversion_tools.{tool_id}.timeout' must be a positive number of seconds.")
        current = current_tools.get(tool_id)
        if current is None and not allow_command_edits:
            errors.append(
                f"Adding conversion tools ('{tool_id}') from the web UI is disabled. "
                "Edit config/settings.yml or set ALLOW_COMMAND_EDITS=true."
            )
            continue
        if "command_template" not in tool:
            if current is None:
                errors.append(f"New tool '{tool_id}' needs a command_template.")
            continue
        template = tool["command_template"]
        unchanged = current is not None and _collapse_whitespace(template) == _collapse_whitespace(current.get("command_template"))
        if not isinstance(template, str) or not template.strip():
            errors.append(f"'conversion_tools.{tool_id}.command_template' must be a non-empty string.")
        elif unchanged:
            continue
        elif not allow_command_edits:
            errors.append(
                f"Changing the command of '{tool_id}' from the web UI is disabled. "
                "Edit config/settings.yml or set ALLOW_COMMAND_EDITS=true."
            )
        else:
            try:
                command_validator(template, {})
            except ValueError as e:
                errors.append(f"Invalid command template for '{tool_id}': {e}")
    return errors
