#!/bin/sh
# Starts a FileWizard image and checks that the web server, the background worker and a
# few converters work. Used by CI and handy after local builds:
#
#   docker/smoke-test.sh filewizard:test
set -eu

image="$1"
name="filewizard-smoke-$$"
work="$(mktemp -d)"

cleanup() {
    status=$?
    if [ "${status}" -ne 0 ]; then
        echo "--- container logs ---"
        docker logs "${name}" 2>&1 | tail -n 60 || true
    fi
    docker rm -f "${name}" >/dev/null 2>&1 || true
    rm -rf "${work}"
    exit "${status}"
}
trap cleanup EXIT

docker run -d --name "${name}" -p 127.0.0.1::8000 "${image}" >/dev/null
port="$(docker port "${name}" 8000/tcp | head -n 1 | sed 's/.*://')"
base="http://127.0.0.1:${port}"

printf 'Waiting for %s ' "${base}"
for _ in $(seq 1 90); do
    if curl -fsS "${base}/health" >/dev/null 2>&1; then break; fi
    printf '.'
    sleep 2
done
echo
curl -fsS "${base}/health"
echo

fail() { echo "FAIL: $*" >&2; exit 1; }

user="$(docker exec "${name}" stat -c '%U' /proc/1)"
[ "${user}" = "filewizard" ] || fail "processes run as '${user}', expected filewizard"
echo "ok: runs as ${user}"

curl -fsS "${base}/" | grep -q "File Wizard" || fail "index page"
echo "ok: web UI"

variant="$(curl -fsS "${base}/health" | python3 -c 'import json, sys; print(json.load(sys.stdin).get("variant", ""))')"
if [ "${variant}" != "small" ]; then
    docker exec "${name}" python -c 'import torch, docling; print("ok: torch", torch.__version__, "and docling import")' \
        || fail "torch/docling import"
fi
if [ "${variant}" = "cuda" ]; then
    docker exec "${name}" python -c 'import ctypes; ctypes.CDLL("libcublas.so.12"); print("ok: CUDA 12 cuBLAS loads")' \
        || fail "libcublas.so.12"
fi

curl -fsS "${base}/api/v1/ocr-languages" | grep -q '"eng"' || fail "tesseract languages"
echo "ok: OCR languages $(curl -fsS "${base}/api/v1/ocr-languages")"

# Uploads a file through the chunked upload API and waits for the job to finish.
convert() {
    file="$1"; name_in_ui="$2"; payload="$3"
    upload_id="smoke-$(date +%s)-$$-${name_in_ui%%.*}"
    curl -fsS -H "Origin: ${base}" -F "upload_id=${upload_id}" -F chunk_number=0 -F "chunk=@${file}" \
        "${base}/upload/chunk" >/dev/null
    job_id="$(curl -fsS -H "Origin: ${base}" -H 'Content-Type: application/json' \
        -d "{\"upload_id\": \"${upload_id}\", \"original_filename\": \"${name_in_ui}\", \"total_chunks\": 1, ${payload}}" \
        "${base}/upload/finalize" | python3 -c 'import json, sys; print(json.load(sys.stdin)["id"])')"
    for _ in $(seq 1 120); do
        status="$(curl -fsS "${base}/job/${job_id}" | python3 -c 'import json, sys; print(json.load(sys.stdin)["status"])')"
        case "${status}" in
            completed) echo "ok: ${name_in_ui} (${payload})"; return 0 ;;
            failed|cancelled) curl -fsS "${base}/job/${job_id}"; echo; fail "${name_in_ui} ${status}" ;;
        esac
        sleep 2
    done
    fail "${name_in_ui} timed out"
}

printf '# Smoke test\n\nHello *FileWizard*.\n' > "${work}/doc.md"
convert "${work}/doc.md" doc.md '"task_type": "conversion", "output_format": "pandoc_html"'
printf 'Hello from LibreOffice\n' > "${work}/note.txt"
convert "${work}/note.txt" note.txt '"task_type": "conversion", "output_format": "libreoffice_pdf"'

echo "Smoke test passed for ${image}"
