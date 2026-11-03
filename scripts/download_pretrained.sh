#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
checkpoint="pretraining_ckpt/bevfusion-seg.pth"
url='https://www.dropbox.com/scl/fi/8lgd1hkod2a15mwry0fvd/bevfusion-seg.pth?rlkey=2tmgw7mcrlwy9qoqeui63tay9&dl=1'

if [[ -f "$checkpoint" ]]; then
    echo "$checkpoint already exists."
    exit 0
fi

mkdir -p pretraining_ckpt
curl --fail --location --output "$checkpoint.part" "$url"
mv "$checkpoint.part" "$checkpoint"
echo "Saved $checkpoint"
