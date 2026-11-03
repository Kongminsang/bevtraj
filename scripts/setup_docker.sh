#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
docker compose build
docker compose run --rm bevtraj python docker/verify_install.py --expect-gpus 1 "$@"
