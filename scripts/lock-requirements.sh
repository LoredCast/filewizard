#!/usr/bin/env bash
# Regenerates the hash-pinned lock files in locks/ from the requirements*.txt files.
# Run it after changing requirements*.txt, and periodically to pick up updates:
#
#   pip install uv && scripts/lock-requirements.sh
#
# locks/full.txt and locks/small.txt cover linux/amd64 and linux/arm64, locks/cuda.txt
# linux/amd64. PyTorch is not part of the locks: the images install the CPU build of the
# version pinned in locks/torch-cpu.txt from the PyTorch index (PyPI's Linux wheels pull
# in several GB of CUDA libraries).
set -euo pipefail
cd "$(dirname "$0")/.."
repo="$PWD"
command -v uv >/dev/null || { echo "uv is required: pip install uv" >&2; exit 1; }

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
# Resolve against PyPI only; the PyTorch CPU index is used at install time.
for f in requirements.txt requirements_small.txt requirements_cuda.txt; do
    grep -v '^--extra-index-url' "$f" > "$tmp/$f"
done

compile() {
    (cd "$tmp" && uv pip compile "$@" --python-version 3.12 --generate-hashes --only-binary :all: \
        --custom-compile-command "scripts/lock-requirements.sh" --quiet)
}
mkdir -p locks
compile requirements_small.txt --universal -o "$repo/locks/small.txt"
compile requirements.txt --universal -o "$tmp/full-with-torch.txt"
compile requirements_cuda.txt --python-platform x86_64-manylinux_2_39 -o "$repo/locks/cuda.txt"

grep -E '^(torch|torchvision)==' "$tmp/full-with-torch.txt" | sed 's/ .*//' > locks/torch-cpu.txt
cp "$tmp/full-with-torch.txt" locks/full.txt
for lock in locks/full.txt locks/cuda.txt; do
    python3 scripts/prune_lock.py "$lock" --separate torch torchvision --drop 'nvidia-*' 'cuda-*' triton
done

# Every pinned package must be installable from wheels on each target platform.
check() {
    uv pip compile "$1" --python-platform "$2" --python-version 3.12 --only-binary :all: --no-deps -o "$tmp/check.txt" --quiet
}
for platform in x86_64-manylinux_2_39 aarch64-manylinux_2_39; do
    check locks/full.txt "$platform"
    check locks/small.txt "$platform"
done
check locks/cuda.txt x86_64-manylinux_2_39
echo "Locks updated: $(cat locks/torch-cpu.txt | tr '\n' ' ')"
