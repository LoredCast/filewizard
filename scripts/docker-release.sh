#!/bin/sh
# Builds the FileWizard images on this machine and publishes them to Docker Hub in two steps:
#
#   scripts/docker-release.sh test 0.5.0
#       Builds every variant for every platform and pushes the images as test, test-small
#       and test-cuda. Try them before releasing.
#   scripts/docker-release.sh promote 0.5.0
#       Gives the tested images their release tags (0.5.0, 0.5, 0.5-latest, latest,
#       0.5.0-small, 0.5-small, small, 0.5.0-cuda, 0.5-cuda, cuda, latest-cuda).
#       Nothing is rebuilt: the release tags point to exactly the images you tested.
#   scripts/docker-release.sh tags 0.5.0 [full|small|cuda]
#       Prints the release tags without publishing anything.
#
# Versions look like 0.5.0, 0.5 or 0.5.0-rc1 (a pre-release only gets its own tag).
# Log in first with `docker login`. Optional environment variables:
#
#   IMAGE        repository to publish to (default: loredcast/filewizard)
#   VARIANTS     variants to build and promote (default: "full small cuda")
#   PLATFORMS    platforms of the full and small images (default: linux/amd64,linux/arm64);
#                the cuda image is linux/amd64 only
#   BUILDER      buildx builder to use (default: "filewizard", created if missing)
#   BUILD_FLAGS  extra flags for `docker buildx build`, e.g. "--no-cache" or
#                "--secret id=build_ca,src=/path/to/proxy-ca.crt"
#
# See docs/docker.md.
set -eu

IMAGE="${IMAGE:-loredcast/filewizard}"
VARIANTS="${VARIANTS:-full small cuda}"
PLATFORMS="${PLATFORMS:-linux/amd64,linux/arm64}"
BUILDER="${BUILDER:-filewizard}"
BUILD_FLAGS="${BUILD_FLAGS:-}"

repo_root="$(cd "$(dirname "$0")/.." && pwd)"

usage() {
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0" >&2
    exit 2
}

die() {
    echo "Error: $*" >&2
    exit 1
}

check_version() {
    echo "$1" | grep -Eq '^v?[0-9]+\.[0-9]+(\.[0-9]+)?(-[0-9A-Za-z.]+)?$' \
        || die "'$1' is not a version like 0.5.0, 0.5 or 0.5.0-rc1"
}

# Tag suffix of a variant.
suffix() {
    case "$1" in
        full) echo "" ;;
        small) echo "-small" ;;
        cuda) echo "-cuda" ;;
        *) die "unknown variant '$1' (full, small or cuda)" ;;
    esac
}

# Release tags of a variant, e.g. "0.5.0 0.5 0.5-latest latest" for 0.5.0 and full.
release_tags() {
    rt_version="${1#v}"
    rt_suffix="$(suffix "$2")"
    case "${rt_version}" in
        *-*) echo "${rt_version}${rt_suffix}"; return ;;
    esac
    rt_minor="$(echo "${rt_version}" | cut -d. -f1,2)"
    rt_tags="${rt_version}${rt_suffix}"
    if [ "${rt_minor}" != "${rt_version}" ]; then
        rt_tags="${rt_tags} ${rt_minor}${rt_suffix}"
    fi
    case "$2" in
        full) rt_tags="${rt_tags} ${rt_minor}-latest latest" ;;
        small) rt_tags="${rt_tags} small" ;;
        cuda) rt_tags="${rt_tags} cuda latest-cuda" ;;
    esac
    echo "${rt_tags}"
}

# Platforms a variant is built for (empty: none of PLATFORMS applies).
platforms_for() {
    if [ "$1" = "cuda" ]; then
        case ",${PLATFORMS}," in
            *,linux/amd64,*) echo "linux/amd64" ;;
            *) echo "" ;;
        esac
    else
        echo "${PLATFORMS}"
    fi
}

ensure_builder() {
    if ! docker buildx inspect "${BUILDER}" >/dev/null 2>&1; then
        echo "Creating buildx builder '${BUILDER}'"
        docker buildx create --name "${BUILDER}" --driver docker-container >/dev/null
    fi
    available="$(docker buildx inspect --bootstrap "${BUILDER}" | sed -n 's/^Platforms: *//p' | tr -d ' *')"
    needed=""
    for variant in ${VARIANTS}; do
        needed="${needed},$(platforms_for "${variant}")"
    done
    for platform in $(echo "${needed}" | tr ',' ' '); do
        case ",${available}," in
            *",${platform},"*) ;;
            *) die "the buildx builder '${BUILDER}' cannot build ${platform}.
On Linux, install QEMU emulation for it and let the script create the builder again:
    docker run --privileged --rm tonistiigi/binfmt --install ${platform#linux/}
    docker buildx rm ${BUILDER}
Or build only for your own platform, e.g. PLATFORMS=linux/amd64." ;;
        esac
    done
}

cmd_test() {
    version="${1#v}"
    ensure_builder
    if [ -n "$(git -C "${repo_root}" status --porcelain)" ]; then
        echo "Warning: the working tree has uncommitted changes; they end up in the images." >&2
    fi
    # File times inside the image come from the commit, so rebuilding a commit gives the same image.
    epoch="$(git -C "${repo_root}" log -1 --format=%ct)"
    revision="$(git -C "${repo_root}" rev-parse HEAD)"
    pushed=""
    for variant in ${VARIANTS}; do
        platforms="$(platforms_for "${variant}")"
        if [ -z "${platforms}" ]; then
            echo "Skipping ${variant}: it is not built for ${PLATFORMS}"
            continue
        fi
        tag="${IMAGE}:test$(suffix "${variant}")"
        echo "=== Building ${variant} for ${platforms} and pushing ${tag}"
        # shellcheck disable=SC2086  # BUILD_FLAGS holds several flags
        docker buildx build --builder "${BUILDER}" --platform "${platforms}" \
            --build-arg VARIANT="${variant}" --build-arg VERSION="${version}" \
            --build-arg SOURCE_DATE_EPOCH="${epoch}" \
            --label org.opencontainers.image.revision="${revision}" \
            --provenance=mode=max --sbom=true \
            --output "type=image,name=${tag},push=true,unpack=false,rewrite-timestamp=true" \
            ${BUILD_FLAGS} "${repo_root}"
        pushed="${pushed} ${tag}"
    done
    echo
    echo "Pushed the test images. Try them (pull first, so no older local copy is used):"
    for tag in ${pushed}; do
        echo "    docker pull ${tag} && docker/smoke-test.sh ${tag}"
    done
    echo "then publish them under the release tags:"
    echo "    $0 promote ${version}"
}

cmd_promote() {
    version="${1#v}"
    released=""
    for variant in ${VARIANTS}; do
        if [ -z "$(platforms_for "${variant}")" ]; then
            continue
        fi
        source="${IMAGE}:test$(suffix "${variant}")"
        # Refuse to release a test image that was built for another version.
        config="$(docker buildx imagetools inspect "${source}" --format '{{json .Image}}')" \
            || die "could not read ${source}; build it with: $0 test ${version}"
        built="$(echo "${config}" | grep -o '"org.opencontainers.image.version": *"[^"]*"' \
            | head -n 1 | sed 's/.*"\([^"]*\)"$/\1/')"
        if [ "${built}" != "${version}" ]; then
            die "${source} was built as version '${built}', not ${version}; run: $0 test ${version}"
        fi
        args=""
        for name in $(release_tags "${version}" "${variant}"); do
            args="${args} --tag ${IMAGE}:${name}"
        done
        echo "=== ${source} ->$(echo "${args}" | sed 's/ --tag / /g')"
        # shellcheck disable=SC2086  # args holds several flags
        docker buildx imagetools create ${args} "${source}"
        released="${released} ${IMAGE}:$(release_tags "${version}" "${variant}" | cut -d' ' -f1)"
    done
    echo
    echo "Released ${version}. Check the result with:"
    for tag in ${released}; do
        echo "    docker buildx imagetools inspect ${tag}"
    done
}

[ $# -ge 2 ] || usage
command="$1"
check_version "$2"
for variant in ${VARIANTS}; do
    suffix "${variant}" >/dev/null
done

case "${command}" in
    test) cmd_test "$2" ;;
    promote) cmd_promote "$2" ;;
    tags)
        if [ $# -ge 3 ]; then
            suffix "$3" >/dev/null
            release_tags "$2" "$3"
        else
            for variant in full small cuda; do
                echo "${variant}: $(release_tags "$2" "${variant}")"
            done
        fi
        ;;
    *) usage ;;
esac
