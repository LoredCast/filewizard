#!/bin/sh
# Starts FileWizard as an unprivileged user.
#
# When the container starts as root (the default), the app user's uid/gid are set to
# PUID/PGID (default 1000:1000, e.g. 99:100 on Unraid), the writable directories are
# handed to that user, and root privileges are dropped. When the container is started
# with --user, everything runs as that user and the mounted volumes must be writable.
set -eu

writable_dirs="/app/config /app/data /app/models /app/uploads /app/processed /home/filewizard"

if [ "$(id -u)" = "0" ]; then
    PUID="${PUID:-1000}"
    PGID="${PGID:-1000}"
    if [ "$(id -g filewizard)" != "${PGID}" ]; then
        groupmod --non-unique --gid "${PGID}" filewizard
    fi
    if [ "$(id -u filewizard)" != "${PUID}" ]; then
        usermod --non-unique --uid "${PUID}" --gid "${PGID}" filewizard
    fi
    for dir in ${writable_dirs}; do
        mkdir -p "${dir}"
        # Only walk the tree when the owner differs (e.g. volumes written by older, root-run images).
        if [ "$(stat -c '%u:%g' "${dir}")" != "${PUID}:${PGID}" ]; then
            echo "Setting owner of ${dir} to ${PUID}:${PGID}"
            chown -R "${PUID}:${PGID}" "${dir}"
        fi
    done
    # The container's stdout/stderr pipes belong to root; hand them over (as `docker run --user`
    # does) so the processes can reopen them for their logs.
    # ($$: this shell's descriptors; a redirect on chown itself would change what fd 2 points to.)
    chown "${PUID}:${PGID}" "/proc/$$/fd/1" "/proc/$$/fd/2" || true
    exec setpriv --reuid="${PUID}" --regid="${PGID}" --init-groups -- "$@"
fi

exec "$@"
