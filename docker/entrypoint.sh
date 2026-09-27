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
        # Only walk the tree when the folder or an entry directly in it has another owner
        # (e.g. volumes written by older, root-run images, or files copied in as root).
        if [ -n "$(find "${dir}" -maxdepth 1 \( ! -user "${PUID}" -o ! -group "${PGID}" \) -print -quit)" ]; then
            echo "Setting owner of ${dir} to ${PUID}:${PGID}"
            # Some storage (e.g. NFS/SMB shares) refuses ownership changes; that is fine as long
            # as the folder is writable anyway, which is checked below.
            chown -R "${PUID}:${PGID}" "${dir}" 2>/dev/null \
                || echo "Warning: could not change the owner of ${dir}."
        fi
        if ! setpriv --reuid="${PUID}" --regid="${PGID}" --init-groups -- test -w "${dir}"; then
            echo "Error: ${dir} is not writable for user ${PUID}:${PGID}. Set PUID/PGID to the owner of the mounted folder, or make it writable for that user." >&2
            exit 1
        fi
    done
    # The container's stdout/stderr pipes belong to root; hand them over (as `docker run --user`
    # does) so the processes can reopen them for their logs.
    # ($$: this shell's descriptors; a redirect on chown itself would change what fd 2 points to.)
    chown "${PUID}:${PGID}" "/proc/$$/fd/1" "/proc/$$/fd/2" || true
    exec setpriv --reuid="${PUID}" --regid="${PGID}" --init-groups -- "$@"
fi

exec "$@"
