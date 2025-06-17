import pytorch_lightning as pl
import torch

torch.set_float32_matmul_precision('medium')
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from models import build_model
from datasets import build_dataset
from utils.utils import set_seed, find_latest_checkpoint
from utils.config import save_config_as_txt
from pytorch_lightning.callbacks import ModelCheckpoint
import hydra
from omegaconf import OmegaConf
import os


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)

    save_config_as_txt(cfg)

    cfg.MODEL.debug = cfg.debug
    model = build_model(cfg.MODEL)
    train_set = build_dataset(cfg.TRAIN_DATASET, val=False)
    val_set = build_dataset(cfg.VAL_DATASET, val=True)

    train_batch_size = max(cfg.method['train_batch_size'] // len(cfg.devices) // train_set.data_chunk_size, 1)
    eval_batch_size = max(cfg.method['eval_batch_size'] // len(cfg.devices) // val_set.data_chunk_size, 1)

    call_backs = []

    checkpoint_callback = ModelCheckpoint(
        dirpath='ckpt/' + cfg.exp_name,
        # monitor='val/minADE5',
        filename='{epoch}-{val/minADE5:.2f}',
        save_top_k=-1,
        mode='min',  # 'min' for loss/error, 'max' for accuracy
        every_n_epochs=3,
    )

    call_backs.append(checkpoint_callback)

    train_loader = DataLoader(
        train_set, batch_size=train_batch_size, num_workers=cfg.load_num_workers, drop_last=False,
        collate_fn=train_set.collate_fn)

    val_loader = DataLoader(
        val_set, batch_size=eval_batch_size, num_workers=cfg.load_num_workers, shuffle=False, drop_last=False,
        collate_fn=train_set.collate_fn)

    trainer = pl.Trainer(
        max_epochs=cfg.method.max_epochs,
        logger=None if cfg.debug else WandbLogger(project="unitraj", name=cfg.exp_name, id=cfg.exp_name),
        devices=1 if cfg.debug else cfg.devices,
        gradient_clip_val=cfg.method.grad_clip_norm,
        accelerator="cpu" if cfg.debug else "gpu",
        profiler="simple",
        strategy="auto" if cfg.debug else "ddp_find_unused_parameters_true",
        callbacks=call_backs,
        check_val_every_n_epoch=3,
    )

    # automatically resume training
    if cfg.ckpt_path is None and not cfg.debug:
        cfg.ckpt_path = find_latest_checkpoint(os.path.join('ckpt', cfg.exp_name))

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=cfg.ckpt_path)


if __name__ == '__main__':
    train()