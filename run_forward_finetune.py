"""Fine-tune the released 400-d DOS forward model (total DOS only) on a new dataset.

Unlike run_forward_finetune_800.py this does NO head surgery: the released
checkpoint (outputs/dos_forward_model) is already a 400-d monolithic-node_out
CSPProperty, and the total-only fine-tune keeps pred_dim=400, so every tensor
loads 1:1. The load is verified strict (missing == unexpected == []). The
released prop/lattice scalers are REUSED via scaler_path (not refit), keeping
the head on its pretraining calibration. Config: configs/dos_forward_dmx2_ft.yml.
"""
from typing import List

import os
from glob import glob
from pathlib import Path
from datetime import datetime
from omegaconf import DictConfig, OmegaConf

import torch

import pytorch_lightning as pl
from pytorch_lightning import Callback, seed_everything
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from dosmatgen.dataset.datamodule import CrystalDataModule
from dosmatgen.diffusion.property import CSPProperty
from dosmatgen.utils.utils import log_hyperparameters

# The released forward model's scalers were pickled under the upstream package name
# 'spectrodiff'; alias it to our 'dosmatgen' package so torch.load can resolve the class.
import sys, importlib
for _real, _alias in [("dosmatgen", "spectrodiff"),
                      ("dosmatgen.utils", "spectrodiff.utils"),
                      ("dosmatgen.utils.data", "spectrodiff.utils.data")]:
    sys.modules.setdefault(_alias, importlib.import_module(_real))


def build_callbacks(config: DictConfig, save_dir) -> List[Callback]:
    callbacks: List[Callback] = []
    if "lr_monitor" in config.logging:
        callbacks.append(LearningRateMonitor(
            logging_interval=config.logging.lr_monitor.logging_interval,
            log_momentum=config.logging.lr_monitor.log_momentum,
        ))
    if "early_stopping" in config.train:
        callbacks.append(EarlyStopping(
            monitor=config.optim.lr_scheduler.monitor_metric,
            mode=config.optim.lr_scheduler.monitor_metric_mode,
            patience=config.train.early_stopping.patience,
            verbose=config.train.early_stopping.verbose,
        ))
    if "model_checkpoints" in config.train:
        callbacks.append(ModelCheckpoint(
            dirpath=save_dir,
            monitor=config.optim.lr_scheduler.monitor_metric,
            mode=config.optim.lr_scheduler.monitor_metric_mode,
            save_top_k=config.train.model_checkpoints.save_top_k,
            verbose=config.train.model_checkpoints.verbose,
            save_last=config.train.model_checkpoints.save_last,
        ))
    return callbacks


def warm_start_forward(config: DictConfig, pretrain_dir: str):
    """Build the 400-d CSPProperty from the fine-tune config and load the released
    checkpoint strict. Returns (model, ckpt_path)."""
    model = CSPProperty(**config)
    assert not getattr(model.decoder, "node_out_split", False), \
        "total-only fine-tune must not construct a split head"

    ckpt_paths = glob(str(Path(pretrain_dir) / "*.ckpt"))
    if len(ckpt_paths) != 1:
        raise ValueError(f"expected 1 ckpt in {pretrain_dir}, found {ckpt_paths}")
    sd = torch.load(ckpt_paths[0], map_location="cpu", weights_only=False)["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("warm-start (forward total-400): missing =", list(missing),
          "| unexpected =", list(unexpected))
    assert not missing and not unexpected, \
        f"expected a clean 1:1 load, got missing={missing} unexpected={unexpected}"
    return model, ckpt_paths[0]


def run(config: DictConfig, resume_ckpt=None):
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    save_dir = Path(f"{config.save_dir}/{timestamp}_{config.run_name}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"[run] save_dir = {save_dir}", flush=True)

    if config.train.deterministic:
        seed_everything(config.train.random_seed)
    if config.train.trainer.precision == 32:
        torch.set_float32_matmul_precision("medium")

    # REUSE the released scalers (do NOT refit on the fine-tune set)
    data_module = CrystalDataModule(config, scaler_path=config.pretrain_dir)

    model, ckpt = warm_start_forward(config, config.pretrain_dir)
    print("warm-started from", ckpt, "| lr =", model.hparams.optim.params.lr)

    callbacks = build_callbacks(config, save_dir)

    model.lattice_scaler = data_module.lattice_scaler.copy()
    model.scaler = data_module.scaler.copy()
    torch.save(data_module.lattice_scaler, save_dir / "lattice_scaler.pt")
    torch.save(data_module.scaler, save_dir / "prop_scaler.pt")

    wandb_logger = None
    if "wandb" in config.logging:
        import wandb
        wandb_logger = WandbLogger(
            name=config.run_name, group=config.run_name,
            **config.logging.wandb,
            settings=wandb.Settings(start_method="fork"),
        )
        wandb_logger.watch(model, log=config.logging.wandb_watch.log,
                           log_freq=config.logging.wandb_watch.log_freq)

    (save_dir / "hparams.yaml").write_text(OmegaConf.to_yaml(cfg=config))

    csv_logger = CSVLogger(str(save_dir), name="csv")
    loggers = [csv_logger] + ([wandb_logger] if wandb_logger is not None else [])

    trainer = pl.Trainer(
        default_root_dir=save_dir, logger=loggers, callbacks=callbacks,
        deterministic=config.train.deterministic,
        check_val_every_n_epoch=config.logging.val_check_interval,
        **config.train.trainer,
    )
    log_hyperparameters(trainer=trainer, model=model, cfg=config)
    trainer.fit(model=model, datamodule=data_module, ckpt_path=resume_ckpt)
    if wandb_logger is not None:
        wandb_logger.experiment.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dos_forward_dmx2_ft.yml")
    parser.add_argument("--resume", default=None,
                        help="path to a last.ckpt to resume optimizer/epoch state from")
    args = parser.parse_args()
    conf = OmegaConf.load(args.config)
    print(OmegaConf.to_yaml(conf))
    run(conf, resume_ckpt=args.resume)
