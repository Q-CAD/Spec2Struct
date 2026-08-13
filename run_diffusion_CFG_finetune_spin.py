"""Fine-tune the released 400-d CFG generator into an 800-d [total || m] model.

The generator conditions on a per-atom DOS vector. Widening that condition from
total-only (400) to [total(400) || m(400)] needs surgery on one layer, because the
released checkpoint's DOS projection expects a 400-d input:

  decoder.y_proj_total.{weight,bias} <- released decoder.y_projection.{weight,bias}
  decoder.y_proj_m.{weight,bias}     <- 0

The split projection is the `mag_zero_mixin` option of the model (see
dosmatgen/models/cspnet_cfg.py). Because the m pathway starts at exactly zero, the
model reproduces the released total-DOS generator at initialisation, while the m
pathway still receives gradient from the first step and can lift off on its own.

decoder.node_out is widened to 800 by the config but is used by neither the loss nor
the sampler for this model, so it is left at fresh initialisation rather than grafted.
Every other weight loads directly from the released checkpoint.

The 800-d prop_scaler is assembled as [released total stats || dataset m stats] so the
grafted total pathway keeps its pretraining normalisation; the released lattice_scaler
is reused unchanged.

  python run_diffusion_CFG_finetune_spin.py --config configs/dos_cfg_dmx2_spin_ft.yml
"""
from typing import List

import os
from glob import glob
from pathlib import Path
from datetime import datetime
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn.init as init

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
from dosmatgen.utils.data import StandardScalerTorch
from dosmatgen.utils.utils import log_hyperparameters


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


def warm_start_generator(config: DictConfig, pretrain_dir: str):
    """Build the split-projection generator and warm-start it from the released 400-d
    checkpoint. Returns (model, ckpt_path) and asserts the initialisation invariants:
    the total pathway is byte-identical to the released projection and the m pathway
    is exactly zero."""
    model = CSPDiffusion(**config)
    dec = model.decoder
    assert getattr(dec, "mag_zero_mixin", False), \
        "expected diffusion.model.mag_zero_mixin=True for the split-projection warm start"

    ckpt_paths = glob(str(Path(pretrain_dir) / "*.ckpt"))
    if len(ckpt_paths) != 1:
        raise ValueError(f"expected 1 ckpt in {pretrain_dir}, found {ckpt_paths}")
    sd = torch.load(ckpt_paths[0], map_location="cpu", weights_only=False)["state_dict"]

    # The released DOS projection is a single 400-d input layer; ours is split, so pop
    # it and copy it into the total pathway below. node_out has a different width here
    # (800 vs 400) and is unused by this model, so drop it and leave it fresh.
    ckpt_yp_w = sd.pop("decoder.y_projection.weight")  # [hidden, 400]
    ckpt_yp_b = sd.pop("decoder.y_projection.bias")    # [hidden]
    sd.pop("decoder.node_out.weight")                  # [400, hidden] - left fresh

    missing, unexpected = model.load_state_dict(sd, strict=False)

    with torch.no_grad():
        dec.y_proj_total.weight.copy_(ckpt_yp_w)
        dec.y_proj_total.bias.copy_(ckpt_yp_b)
        init.zeros_(dec.y_proj_m.weight)
        init.zeros_(dec.y_proj_m.bias)

    assert torch.equal(dec.y_proj_total.weight, ckpt_yp_w), "total pathway graft is not exact"
    assert torch.equal(dec.y_proj_total.bias, ckpt_yp_b), "total pathway bias graft is not exact"
    assert float(dec.y_proj_m.weight.abs().max()) == 0.0, "m pathway weight is not zero at init"
    assert float(dec.y_proj_m.bias.abs().max()) == 0.0, "m pathway bias is not zero at init"
    assert list(unexpected) == [], f"unexpected keys after backbone load: {unexpected}"

    print("warm start (split projection): missing =", list(missing),
          "| unexpected =", list(unexpected))
    print("  y_proj_total <- released y_projection", tuple(dec.y_proj_total.weight.shape),
          "| y_proj_m zero-init", tuple(dec.y_proj_m.weight.shape),
          "| max|W_m| =", float(dec.y_proj_m.weight.abs().max()))
    return model, ckpt_paths[0]


def build_split_scaler(data_module, pretrain_dir: str, half: int):
    """prop_scaler = [released total stats(half) || dataset m stats(half)]; the released
    lattice_scaler is reused. Keeping the released total stats means the grafted total
    pathway sees the normalisation it was trained with."""
    pre = Path(pretrain_dir)
    pre_prop = torch.load(pre / "prop_scaler.pt", map_location="cpu", weights_only=False)
    pre_lat = torch.load(pre / "lattice_scaler.pt", map_location="cpu", weights_only=False)
    fitted = data_module.scaler  # full-width, fit on the training split by the datamodule
    means = torch.cat([torch.as_tensor(pre_prop.means), torch.as_tensor(fitted.means)[half:]])
    stds = torch.cat([torch.as_tensor(pre_prop.stds), torch.as_tensor(fitted.stds)[half:]])
    return StandardScalerTorch(means=means, stds=stds), pre_lat


def run(config: DictConfig):
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    save_dir = Path(f"{config.save_dir}/{timestamp}_{config.run_name}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"[run] save_dir = {save_dir}", flush=True)

    if config.train.deterministic:
        seed_everything(config.train.random_seed)
    if config.train.trainer.precision == 32:
        torch.set_float32_matmul_precision("medium")

    pred_dim = config.diffusion.model.pred_dim
    if pred_dim % 2 != 0:
        raise ValueError(f"split conditioning needs an even pred_dim, got {pred_dim}")
    half = pred_dim // 2

    # The datamodule fits a full-width scaler on the training split; replace the total
    # half with the released statistics so the grafted pathway stays calibrated.
    data_module = CrystalDataModule(config)
    scaler, pre_lat = build_split_scaler(data_module, config.pretrain_dir, half)
    data_module.scaler = scaler
    data_module.lattice_scaler = pre_lat
    print(f"prop_scaler: width {scaler.means.shape[0]} "
          f"| total half from the released model, m half fit on this dataset")

    model, ckpt = warm_start_generator(config, config.pretrain_dir)
    print("warm-started from", ckpt, "| base lr =", model.hparams.optim.params.lr,
          "| new_pathway_lr =", model.hparams.optim.get("new_pathway_lr", None))
    model.decoder.cfg = True
    model.decoder.cfg_prob = config.diffusion.model.cfg_prob

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

    trainer = pl.Trainer(
        default_root_dir=save_dir, logger=wandb_logger, callbacks=callbacks,
        deterministic=config.train.deterministic,
        check_val_every_n_epoch=config.logging.val_check_interval,
        **config.train.trainer,
    )
    log_hyperparameters(trainer=trainer, model=model, cfg=config)
    trainer.fit(model=model, datamodule=data_module)
    if wandb_logger is not None:
        wandb_logger.experiment.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dos_cfg_dmx2_spin_ft.yml")
    args = parser.parse_args()
    conf = OmegaConf.load(args.config)
    print(OmegaConf.to_yaml(conf))
    run(conf)
