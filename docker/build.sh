#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: bash docker/build.sh /path/scip-6.0.1.tgz /path/soplex-4.0.1.tgz [image-tag]" >&2
  exit 2
fi
docker_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
scip_archive="$(realpath -- "$1")"
soplex_archive="$(realpath -- "$2")"
image_tag="${3:-llm2branch:scip6-cpu}"
[[ -f "$scip_archive" && -f "$soplex_archive" ]]
build_context="$(mktemp -d)"
trap 'rm -rf -- "$build_context"' EXIT
# An explicit allowlist prevents project data, API settings and logs entering Docker.
cp -- "$docker_dir/Dockerfile" "$docker_dir/requirements-runtime.txt" \
  "$docker_dir/verify_runtime.py" "$docker_dir/.dockerignore" "$build_context/"
cp -- "$docker_dir/../scip_patch/vanillafullstrong.patch" "$build_context/"
cp -- "$scip_archive" "$build_context/scip-6.0.1.tgz"
cp -- "$soplex_archive" "$build_context/soplex-4.0.1.tgz"
sha256sum "$build_context/scip-6.0.1.tgz" "$build_context/soplex-4.0.1.tgz"
docker build --platform linux/amd64 \
  --build-arg "JOBS=${JOBS:-4}" \
  --build-arg "PYTHON_IMAGE=${PYTHON_IMAGE:-python:3.8.20-slim-bookworm@sha256:1d52838af602b4b5a831beb13a0e4d073280665ea7be7f69ce2382f29c5a613f}" \
  --build-arg "DEBIAN_MIRROR=${DEBIAN_MIRROR:-https://deb.debian.org}" \
  --build-arg "PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.org/simple}" \
  -t "$image_tag" "$build_context"
docker run --rm --network none "$image_tag"
