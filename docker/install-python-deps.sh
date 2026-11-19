#!/bin/sh
# Builds /opt/venv from hash-pinned lock files and strips build-only artifacts.
# Runs in the python-deps stage of Dockerfile (in parallel with system packages).
set -eux

variant="$1"

if [ -f /run/secrets/build_ca ]; then
    export PIP_CERT=/run/secrets/build_ca
fi

python3 -m venv /opt/venv
/opt/venv/bin/pip install --disable-pip-version-check --no-compile --no-deps --only-binary :all: \
    --require-hashes -r /tmp/locks/requirements.txt

if [ "${variant}" != "small" ]; then
    /opt/venv/bin/pip install --disable-pip-version-check --no-compile --no-deps --only-binary :all: \
        --index-url "${TORCH_INDEX_URL}" -r /tmp/locks/torch-cpu.txt
fi

/opt/venv/bin/pip check

# Remove pip and compile/cache artifacts not needed at runtime (PYTHONDONTWRITEBYTECODE=1 is set).
# Keep setuptools (torch imports it) and torch/bin (PyTorch requires torch_shm_manager on import).
/opt/venv/bin/pip uninstall -y pip
find /opt/venv -type d -name '__pycache__' -exec rm -rf {} +
find /opt/venv -type f \( -name '*.pyc' -o -name '*.pyo' -o -name '*.a' \) -delete
rm -rf /opt/venv/lib/python*/site-packages/torch/include \
       /opt/venv/lib/python*/site-packages/torch/share
