#!/usr/bin/env bash
# Updates FileWizard's dependencies on your machine. Nothing is committed or pushed: review
# `git diff`, build and smoke-test the images, then commit.
#
#   scripts/update-deps.sh audit              List known vulnerabilities in the Python lock files.
#   scripts/update-deps.sh security           Upgrade the Python packages that have advisories, to
#                                             the newest release of the same major version that
#                                             fixes them (a new major only when the fix needs it).
#   scripts/update-deps.sh python [PKG...]    Upgrade these Python packages (and what they need);
#                                             without names, every package.
#   scripts/update-deps.sh calibre [VERSION]  Use the official Calibre build (default: the latest
#                                             release): pins version and checksums in docker/versions.env.
#   scripts/update-deps.sh base [SNAPSHOT]    Pin the current Ubuntu/Alpine base images and an Ubuntu
#                                             archive snapshot (default: today, e.g. 20261106T000000Z).
#   scripts/update-deps.sh all                base + calibre + security.
#   scripts/update-deps.sh build [VARIANT...] Build the images locally (small full cuda) and run
#                                             docker/smoke-test.sh on each.
#
# Needs uv (pip install uv); docker for base and build; curl for calibre. BUILD_FLAGS adds flags
# to `docker buildx build`, e.g. "--no-cache". See docs/docker.md, "Updating dependencies".
set -euo pipefail
cd "$(dirname "$0")/.."

LOCKS="small full cuda"
CALIBRE_RELEASES="${CALIBRE_RELEASES:-https://github.com/kovidgoyal/calibre/releases}"
UBUNTU_SNAPSHOTS="${UBUNTU_SNAPSHOTS:-https://snapshot.ubuntu.com/ubuntu}"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

usage() {
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0" >&2
    exit 2
}

die() {
    echo "Error: $*" >&2
    exit 1
}

need() {
    command -v "$1" >/dev/null || die "$1 is required${2:+ ($2)}"
}

sha256() {
    if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

# set_line FILE REGEX LINE: replaces the line matching REGEX (sed differs between Linux and macOS).
set_line() {
    python3 - "$@" <<'PY'
import re, sys
path, pattern, line = sys.argv[1:]
text = open(path).read()
new, count = re.subn(pattern, lambda m: line, text, count=1, flags=re.M)
if count != 1:
    sys.exit(f"{path}: no line matches {pattern}")
open(path, "w").write(new)
PY
}

# Audits the lock files; writes "name fix-version" lines for vulnerable packages to $1.
audit() {
    need uv "pip install uv"
    out="$1"
    : > "${out}"
    for lock in ${LOCKS}; do
        # pip-audit exits 1 when it finds something; an unreadable report means it failed.
        uvx --quiet pip-audit --disable-pip --no-deps --require-hashes -r "locks/${lock}.txt" \
            --vulnerability-service pypi --progress-spinner off --format json \
            --output "${work}/audit-${lock}.json" 2>"${work}/audit-${lock}.log" || true
        [ -s "${work}/audit-${lock}.json" ] || { cat "${work}/audit-${lock}.log" >&2; die "pip-audit failed for locks/${lock}.txt"; }
    done
    python3 - "${work}" "${out}" ${LOCKS} <<'PY'
import json, sys
work, out, locks = sys.argv[1], sys.argv[2], sys.argv[3:]
def key(version):
    return [int(p) if p.isdigit() else -1 for p in version.replace("-", ".").split(".")]
found = {}
for lock in locks:
    for dep in json.load(open(f"{work}/audit-{lock}.json"))["dependencies"]:
        for vuln in dep.get("vulns", []):
            entry = found.setdefault(dep["name"], {"version": dep["version"], "ids": set(), "fix": "", "locks": set()})
            entry["ids"].add(vuln["id"])
            entry["locks"].add(lock)
            for fix in vuln["fix_versions"]:
                if not entry["fix"] or key(fix) > key(entry["fix"]):
                    entry["fix"] = fix
if not found:
    print("No known vulnerabilities in locks/" + ", locks/".join(f"{l}.txt" for l in locks) + ".")
else:
    print(f"{'package':22} {'pinned':12} {'fixed in':12} {'advisories':>10}  locks")
    for name, e in sorted(found.items()):
        print(f"{name:22} {e['version']:12} {e['fix'] or 'no fix yet':12} {len(e['ids']):10}  {' '.join(sorted(e['locks']))}")
with open(out, "w") as f:
    for name, e in sorted(found.items()):
        f.write(f"{name} {e['fix']}\n")
PY
}

snapshot_locks() {
    mkdir -p "${work}/before"
    cp locks/*.txt "${work}/before/"
}

show_lock_changes() {
    python3 - "${work}/before" ${LOCKS} <<'PY'
import re, sys
before, locks = sys.argv[1], sys.argv[2:]
def pins(path):
    return dict(re.findall(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?==([^ \\\n]+)", open(path).read(), re.M))
for lock in locks:
    old, new = pins(f"{before}/{lock}.txt"), pins(f"locks/{lock}.txt")
    changes = [f"  {n}: {old.get(n, '-')} -> {new.get(n, 'removed')}" for n in sorted(set(old) | set(new)) if old.get(n) != new.get(n)]
    print(f"locks/{lock}.txt: " + (f"{len(changes)} changed" if changes else "unchanged"))
    print("\n".join(changes)) if changes else None
PY
}

next_steps() {
    cat <<EOF

Next:
  git diff --stat                       # review the changes
  scripts/update-deps.sh build          # build small, full and cuda locally and smoke-test them
  TEST_TAG=nightly scripts/docker-release.sh test <version>   # optional: publish a nightly
EOF
}

cmd_audit() {
    audit "${work}/vulnerable.txt"
    [ ! -s "${work}/vulnerable.txt" ]
}

cmd_security() {
    echo "=== Auditing the Python lock files"
    audit "${work}/vulnerable.txt"
    if [ ! -s "${work}/vulnerable.txt" ]; then
        return 0
    fi
    # Upgrade each package to its fix, but not past the major version the fix is in:
    # a new major release can change APIs that main.py uses.
    args=()
    : > "${work}/constraints.txt"
    while read -r name fix; do
        args+=(--upgrade-package "${name}")
        if [ -n "${fix}" ]; then
            major="${fix%%.*}"
            if [ "${major}" -eq "${major}" ] 2>/dev/null; then
                echo "${name}>=${fix},<$((major + 1))" >> "${work}/constraints.txt"
            fi
        fi
    done < "${work}/vulnerable.txt"
    echo
    echo "=== Upgrading: $(cut -d' ' -f1 "${work}/vulnerable.txt" | tr '\n' ' ')"
    snapshot_locks
    scripts/lock-requirements.sh --constraint "${work}/constraints.txt" "${args[@]}"
    show_lock_changes
    echo
    echo "=== Auditing again"
    if audit "${work}/remaining.txt" && [ -s "${work}/remaining.txt" ]; then
        echo
        echo "Some advisories remain: another package holds these back, or no fixed release exists yet."
        echo "Upgrading the package together with what depends on it often helps:"
        echo "  scripts/update-deps.sh python $(cut -d' ' -f1 "${work}/remaining.txt" | tr '\n' ' ')<dependent packages>"
    fi
}

cmd_python() {
    snapshot_locks
    if [ $# -eq 0 ]; then
        echo "=== Upgrading every Python package"
        scripts/lock-requirements.sh --upgrade
    else
        args=()
        for name in "$@"; do args+=(--upgrade-package "${name}"); done
        echo "=== Upgrading: $*"
        scripts/lock-requirements.sh "${args[@]}"
    fi
    show_lock_changes
    echo
    echo "=== Auditing"
    audit "${work}/vulnerable.txt"
}

cmd_calibre() {
    need curl
    version="${1:-}"
    if [ -z "${version}" ]; then
        # The "latest" page redirects to .../releases/tag/vX.Y.Z.
        version="$(curl -fsSLI -o /dev/null -w '%{url_effective}' "${CALIBRE_RELEASES}/latest" | sed -n 's#.*/tag/v##p')"
    fi
    echo "${version}" | grep -Eq '^[0-9]+\.[0-9]+(\.[0-9]+)?$' || die "'${version}' is not a Calibre version like 9.18.0"
    echo "=== Calibre ${version}: downloading the official builds to record their checksums"
    for arch in x86_64 arm64; do
        curl -fL --retry 3 --progress-bar -o "${work}/calibre-${arch}.txz" \
            "${CALIBRE_RELEASES}/download/v${version}/calibre-${version}-${arch}.txz"
        eval "sha_${arch}=$(sha256 "${work}/calibre-${arch}.txz")"
        echo "  ${arch}: $(eval echo "\${sha_${arch}}")"
    done
    set_line docker/versions.env '^CALIBRE_VERSION=.*$' "CALIBRE_VERSION=${version}"
    set_line docker/versions.env '^CALIBRE_SHA256_X86_64=.*$' "CALIBRE_SHA256_X86_64=${sha_x86_64}"
    set_line docker/versions.env '^CALIBRE_SHA256_ARM64=.*$' "CALIBRE_SHA256_ARM64=${sha_arm64}"
    echo "docker/versions.env now pins Calibre ${version}."
}

cmd_base() {
    need docker
    need curl
    snapshot="${1:-$(date -u +%Y%m%dT000000Z)}"
    echo "${snapshot}" | grep -Eq '^[0-9]{8}T[0-9]{6}Z$' || die "'${snapshot}' is not a snapshot like 20261106T000000Z"
    curl -fsSI "${UBUNTU_SNAPSHOTS}/${snapshot}/dists/noble-security/InRelease" >/dev/null \
        || die "the Ubuntu archive has no snapshot ${snapshot} (${UBUNTU_SNAPSHOTS})"
    for arg in UBUNTU_IMAGE BOOTSTRAP_CA_IMAGE; do
        image="$(sed -n "s/^ARG ${arg}=\([^@]*\)@.*/\1/p" Dockerfile)"
        [ -n "${image}" ] || die "Dockerfile has no 'ARG ${arg}=<image>@<digest>' line"
        digest="$(docker buildx imagetools inspect "${image}" | awk '$1 == "Digest:" { print $2; exit }')"
        echo "${digest}" | grep -Eq '^sha256:[0-9a-f]{64}$' || die "could not read the digest of ${image}"
        set_line Dockerfile "^ARG ${arg}=.*$" "ARG ${arg}=${image}@${digest}"
        echo "${image}: ${digest}"
    done
    set_line Dockerfile '^ARG UBUNTU_SNAPSHOT=.*$' "ARG UBUNTU_SNAPSHOT=${snapshot}"
    echo "Ubuntu archive snapshot: ${snapshot}"
}

cmd_build() {
    need docker
    variants="${*:-small full cuda}"
    host_arch="$(docker version --format '{{.Server.Arch}}' 2>/dev/null || uname -m)"
    version="local-$(git rev-parse --short HEAD 2>/dev/null || echo dev)"
    for variant in ${variants}; do
        case "${variant}" in
            small|full) platform="" ;;
            cuda)
                if [ $# -eq 0 ] && [ "${host_arch}" != "amd64" ] && [ "${host_arch}" != "x86_64" ]; then
                    echo "Skipping cuda: it is amd64 only (name it explicitly to build it under emulation)."
                    continue
                fi
                platform="--platform linux/amd64" ;;
            *) die "unknown variant '${variant}' (small, full or cuda)" ;;
        esac
        tag="filewizard:local-${variant}"
        echo "=== Building ${tag}"
        # shellcheck disable=SC2086  # BUILD_FLAGS and platform hold several words
        docker buildx build ${platform} ${BUILD_FLAGS:-} --build-arg VARIANT="${variant}" \
            --build-arg VERSION="${version}" -t "${tag}" --load .
        echo "=== Smoke-testing ${tag}"
        docker/smoke-test.sh "${tag}"
    done
    echo
    echo "All built images passed the smoke test."
}

[ $# -ge 1 ] || usage
command="$1"
shift
case "${command}" in
    audit) cmd_audit ;;
    security) cmd_security; next_steps ;;
    python) cmd_python "$@"; next_steps ;;
    calibre) cmd_calibre "$@"; next_steps ;;
    base) cmd_base "$@"; next_steps ;;
    all)
        cmd_base
        echo
        cmd_calibre
        echo
        cmd_security
        next_steps ;;
    build) cmd_build "$@" ;;
    *) usage ;;
esac
