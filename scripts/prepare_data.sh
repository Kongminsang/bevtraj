#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: bash scripts/prepare_data.sh {nuscenes|argo2} [all|infos|train|val]"
    echo "Set BEVTRAJ_WORKERS to override the dataset's default worker count."
}

case "${1:-}" in
    nuscenes)
        dataset=nuscenes
        data_root=data/nuscenes
        version=v1.0
        workers="${BEVTRAJ_WORKERS:-1}"
        ;;
    argo2)
        dataset=argo2
        data_root=data/av2_sensor
        version=trainval
        workers="${BEVTRAJ_WORKERS:-4}"
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

stage="${2:-all}"
case "$stage" in
    all|infos|train|val) ;;
    *)
        usage >&2
        exit 2
        ;;
esac

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ "$stage" == all || "$stage" == infos ]]; then
    echo "[$dataset] Generating sensor info files and token mappings."
    python -m mmdet3d_tools.create_data "$dataset" \
        --root-path "$data_root" --version "$version" --out-dir "$data_root" \
        --extra-tag BEVTraj --workers "$workers"
fi

for split in train val; do
    if [[ "$stage" != all && "$stage" != "$split" ]]; then
        continue
    fi
    echo "[$dataset] Converting $split trajectories."
    if [[ "$dataset" == nuscenes ]]; then
        python -m md_scenarionet.convert_nuscenes \
            -d "$data_root/prediction_converted_$split" --split "$split" \
            --dataroot "$data_root" --num_workers "$workers"
    else
        python -m md_scenarionet.convert_argoverse2 \
            -d "$data_root/${split}_converted" --raw_data_path "$data_root/$split" \
            --num_workers "$workers"
    fi
done
