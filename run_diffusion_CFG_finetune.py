"""Fine-tune the MP-DOS CFG diffusion model on the DMX 2D-magnet DOS data.

Ported from Spec2Struct/run_diffusion_CFG_finetune_yproj.py, with three changes
appropriate to our base checkpoint (outputs/260518_164718_dos_cfg, already
cfg=True / pred_dim=400):

  * NO xavier re-init of y_projection. Shuyi re-initialized it because his base
    was the perov-5 unconditional model (no DOS conditioning). Our base is the
    MP-DOS CFG model, so the DOS conditioning head transfers directly.
  * REUSE the pretrained MP-DOS scalers (scaler_path=pretrain_dir) instead of
    refitting on the fine-tune set, so y / lattice stay on the normalization the
    checkpoint expects.
  * Fine-tune at a reduced LR (configs/dos_cfg_dmx2_ft.yml: optim.params.lr=1e-4).

The model architecture is rebuilt from the checkpoint's own hparams.yaml (with
optim/cfg overridden), guaranteeing it matches the saved weights.
"""
from typing import List

import os
import wandb
from glob import glob
from pathlib import Path
from datetime import datetime
from omegaconf import DictConfig, OmegaConf

import torch

import pytorch_lightning as pl
from pytorch_lightning import Callback, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from dosmatgen.dataset.datamodule import CrystalDataModule
from dosmatgen.diffusion.diffusion_cfg import CSPDiffusion
from dosmatgen.utils.utils import log_hyperparameters


def build_callbacks(config: DictConfig, save_dir) -> List[Callback]:
    callbacks: List[Callback] = []

    if "lr_monitor" in config.logging:
        callbacks.append(
            LearningRateMonitor(
                logging_interval=config.logging.lr_monitor.logging_interval,
                log_momentum=config.logging.lr_monitor.log_momentum,
            )
        )

    if "early_stopping" in config.train:
        callbacks.append(
            EarlyStopping(
                monitor=config.optim.lr_scheduler.monitor_metric,
                mode=config.optim.lr_scheduler.monitor_metric_mode,
                patience=config.train.early_stopping.patience,
                verbose=config.train.early_stopping.verbose,
            )
        )

    if "model_checkpoints" in config.train:
        callbacks.append(
            ModelCheckpoint(
                dirpath=save_dir,
                monitor=config.optim.lr_scheduler.monitor_metric,
                mode=config.optim.lr_scheduler.monitor_metric_mode,
                save_top_k=config.train.model_checkpoints.save_top_k,
                verbose=config.train.model_checkpoints.verbose,
                save_last=config.train.model_checkpoints.save_last,
            )
        )

    return callbacks


def run(config: DictConfig, resume_ckpt=None):
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    save_dir = Path(f"{config.save_dir}/{timestamp}_{config.run_name}")
    os.makedirs(save_dir, exist_ok=True)

    if config.train.deterministic:
        seed_everything(config.train.random_seed)

    if config.train.trainer.precision == 32:
        torch.set_float32_matmul_precision('medium')

    # Data module: REUSE the pretrained MP-DOS scalers (do NOT refit on the
    # fine-tune set) so targets/lattices stay on the checkpoint's normalization.
    data_module = CrystalDataModule(config, scaler_path=config.pretrain_dir)

    # Load the pretrained MP-DOS CFG checkpoint. Rebuild the architecture from
    # the checkpoint's own hparams so it matches the saved weights, but override
    # optim (reduced LR) and keep CFG on.
    pretrained_model_path = Path(config.pretrain_dir)
    pretrained_config = OmegaConf.load(pretrained_model_path / "hparams.yaml")
    pretrained_config.diffusion.model.cfg = True
    pretrained_config.diffusion.model.cfg_prob = config.diffusion.model.cfg_prob
    pretrained_config.optim = config.optim  # fine-tune LR/schedule

    ckpt_paths = glob(str(pretrained_model_path / '*.ckpt'))
    if len(ckpt_paths) == 0:
        raise ValueError(f"No checkpoint found in {pretrained_model_path}")
    if len(ckpt_paths) > 1:
        raise ValueError(f"Multiple checkpoints found in {pretrained_model_path}: {ckpt_paths}")
    ckpt_path = ckpt_paths[0]

    # cfg=True, pred_dim=400 -> loads cleanly. strict=False is belt-and-suspenders.
    # NOTE: unlike the perov-5 base, we do NOT re-init y_projection here.
    # weights_only=False: our own trusted ckpt; torch>=2.6 defaults to True and
    # chokes on the omegaconf objects in hparams. No-op on torch 2.5.1 (L4).
    model = CSPDiffusion.load_from_checkpoint(ckpt_path, **pretrained_config, strict=False, weights_only=False)
    print("Loaded pretrained checkpoint:", ckpt_path)
    print("  pred_dim     =", model.hparams.diffusion.model.pred_dim)
    print("  fine-tune lr =", model.hparams.optim.params.lr)

    model.decoder.cfg = True
    model.decoder.cfg_prob = config.diffusion.model.cfg_prob

    callbacks: List[Callback] = build_callbacks(config, save_dir)

    # Persist the (reused) scalers into the run dir so generation is self-contained.
    if data_module.scaler is not None:
        model.lattice_scaler = data_module.lattice_scaler.copy()
        model.scaler = data_module.scaler.copy()
    torch.save(data_module.lattice_scaler, save_dir / 'lattice_scaler.pt')
    torch.save(data_module.scaler, save_dir / 'prop_scaler.pt')

    wandb_logger = None
    if "wandb" in config.logging:
        wandb_config = config.logging.wandb
        wandb_logger = WandbLogger(
            name=config.run_name,
            group=config.run_name,
            **wandb_config,
            settings=wandb.Settings(start_method="fork"),
        )
        wandb_logger.watch(
            model,
            log=config.logging.wandb_watch.log,
            log_freq=config.logging.wandb_watch.log_freq,
        )

    yaml_conf: str = OmegaConf.to_yaml(cfg=config)
    (save_dir / "hparams.yaml").write_text(yaml_conf)

    trainer = pl.Trainer(
        default_root_dir=save_dir,
        logger=wandb_logger,
        callbacks=callbacks,
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
    parser.add_argument("--config", default="configs/dos_cfg_dmx2_ft.yml")
    parser.add_argument("--resume", default=None,
                        help="path to a last.ckpt to resume optimizer/epoch state from")
    args = parser.parse_args()

    conf = OmegaConf.load(args.config)
    print(OmegaConf.to_yaml(conf))

    run(conf, resume_ckpt=args.resume)
