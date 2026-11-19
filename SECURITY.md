# Security Policy

## Reporting a Vulnerability

Please do **not** open a public GitHub issue for suspected security vulnerabilities.

Instead, report vulnerabilities privately via GitHub's **Private Vulnerability Reporting** (Security Advisories):

- https://github.com/LoredCast/filewizard/security/advisories/new

Include as much detail as possible (affected version or image tag, reproduction steps, and impact). You can expect an acknowledgement and coordination on a fix before public disclosure.

## Deployment Security Notes

- In `LOCAL_ONLY=True` mode there is no authentication; keep the service on a trusted network or enable OIDC login (`LOCAL_ONLY=False`).
- FileWizard runs external document, image, audio, and video converters on uploaded files. Run the container as an unprivileged user (the default since `0.5.0`) and mount only the directories it needs.
- For additional hardening options (`SECRET_KEY`, `SESSION_COOKIE_SECURE`, `ALLOWED_HOSTS`, `CSRF_TRUSTED_ORIGINS`, `ALLOW_COMMAND_EDITS`, resource limits), see the [Security section in README.md](README.md#security).
