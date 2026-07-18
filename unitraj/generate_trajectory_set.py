"""Create full-trajectory K-means centers from the training dataset."""

import argparse
from pathlib import Path
import pickle
import sys

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from datasets import build_dataset
from utils.utils import set_seed


NUM_FUTURE_STEPS = 60
OUTPUT_DIR = Path(__file__).resolve().parent / "models" / "bevtraj"
CLI_NUM_CLUSTERS = 32


def parse_cli_args(args):
    """Parse script options while preserving Hydra overrides."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--num-clusters", type=int, default=32)
    return parser.parse_known_args(args)


def collect_vehicle_trajectories(train_loader):
    """Collect complete 60-step vehicle futures in the target-agent frame."""
    vehicle_trajectory_list = []
    skipped_incomplete = 0

    for batch in train_loader:
        inputs = batch["input_dict"]
        future_xy = inputs["center_gt_trajs"][..., :2]
        future_mask = inputs["center_gt_trajs_mask"].bool()
        object_types = torch.as_tensor(inputs["center_objects_type"])

        if future_xy.ndim != 3 or future_xy.shape[1:] != (NUM_FUTURE_STEPS, 2):
            raise ValueError(
                "Expected center_gt_trajs[..., :2] to have shape "
                f"[B, {NUM_FUTURE_STEPS}, 2], got {tuple(future_xy.shape)}"
            )

        vehicle_mask = object_types == 1
        complete_mask = future_mask.all(dim=-1)
        selected_mask = vehicle_mask & complete_mask
        skipped_incomplete += int((vehicle_mask & ~complete_mask).sum())

        if selected_mask.any():
            vehicle_trajectory_list.append(
                future_xy[selected_mask].cpu().numpy().astype(np.float32)
            )

    if not vehicle_trajectory_list:
        raise RuntimeError("No complete vehicle trajectories were found in the train set")

    if skipped_incomplete:
        print(f"Skipped {skipped_incomplete:,} incomplete vehicle trajectories")
    return np.concatenate(vehicle_trajectory_list, axis=0)


def cluster_trajectories(trajectories, num_clusters, seed):
    """Cluster all 60 XY positions jointly and order centers by frequency."""
    flattened = trajectories.reshape(len(trajectories), -1)
    kmeans = KMeans(
        n_clusters=num_clusters,
        init="k-means++",
        n_init=20,
        max_iter=500,
        random_state=seed,
        algorithm="lloyd",
    )
    labels = kmeans.fit_predict(flattened)

    cluster_counts = np.bincount(labels, minlength=num_clusters)
    frequency_order = np.argsort(-cluster_counts, kind="stable")
    centers = kmeans.cluster_centers_.reshape(num_clusters, NUM_FUTURE_STEPS, 2)
    centers = centers[frequency_order].astype(np.float32)
    return centers, cluster_counts[frequency_order]


@hydra.main(version_base=None, config_path="configs", config_name="config")
def cluster(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, cfg.method)
    num_clusters = CLI_NUM_CLUSTERS
    if num_clusters <= 0:
        raise ValueError(f"num_clusters must be positive, got {num_clusters}")

    cfg.TRAIN_DATASET.TRAJ_DATASET.dataset_type = "BaseDataset"
    train_set = build_dataset(cfg.TRAIN_DATASET.TRAJ_DATASET)
    train_loader = DataLoader(
        train_set,
        batch_size=1024,
        num_workers=cfg.load_num_workers,
        drop_last=False,
        collate_fn=train_set.collate_fn,
    )

    vehicle_trajectories = collect_vehicle_trajectories(train_loader)
    print(
        f"Clustering {len(vehicle_trajectories):,} vehicle trajectories with shape "
        f"{vehicle_trajectories.shape[1:]}"
    )
    centers, cluster_counts = cluster_trajectories(
        vehicle_trajectories, num_clusters, cfg.seed
    )

    assert centers.shape == (num_clusters, NUM_FUTURE_STEPS, 2)
    output_path = OUTPUT_DIR / f"trajectory_set_{num_clusters}_60.pkl"
    with output_path.open("wb") as handle:
        pickle.dump(
            {"VEHICLE": centers}, handle, protocol=pickle.HIGHEST_PROTOCOL
        )

    print(f"Saved {output_path}: shape={centers.shape}, dtype={centers.dtype}")
    print(
        "Cluster sizes (largest/median/smallest): "
        f"{cluster_counts.max():,}/"
        f"{int(np.median(cluster_counts)):,}/"
        f"{cluster_counts.min():,}"
    )


if __name__ == "__main__":
    cli_args, hydra_args = parse_cli_args(sys.argv[1:])
    CLI_NUM_CLUSTERS = cli_args.num_clusters
    sys.argv = [sys.argv[0], *hydra_args]
    cluster()
