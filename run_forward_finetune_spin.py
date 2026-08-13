"""Fine-tune the released 400-d DOS forward model into an 800-d [total || m] predictor.

The forward model is a structure -> DOS regressor. Widening its per-atom output from
total-only (400) to [total(400) || m(400)] needs surgery on the output head, because
the released checkpoint has a single 400-d head:

  decoder.node_out_total.weight <- released decoder.node_out.weight
  decoder.node_out_m.weight     <- 0

The split head is the `node_out_split` option of the model (see
dosmatgen/models/cspnet.py). Because the m half starts at exactly zero, the predicted
m is zero at initialisation and the backbone is unchanged, while the m head still
receives gradient from the first step. Every other weight loads directly.

The 800-d prop_scaler is assembled as [released total stats || dataset m stats] so the
grafted total head keeps its pretraining calibration; the released lattice_scaler is
reused unchanged.

  python run_forward_finetune_spin.py --config configs/dos_forward_dmx2_spin_ft.yml
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
from dosmatgen.utils.data import StandardScalerTorch
from dosmatgen.utils.utils import log_hyperparameters

# The released forward model's scalers were pickled under the upstream package name
# 'spectrodiff'; alias it to this repo's 'dosmatgen' package so torch.load can resolve
# those classes.
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
    if "model_checkpoints_m" in config.train:
        # Optional second checkpoint tracking the m half alone. It is written to a
        # subdirectory so that a glob for "*.ckpt" in the run root still resolves to a
        # single file, which is what the inference and eval scripts expect.
        callbacks.append(ModelCheckpoint(
            dirpath=Path(save_dir) / config.train.model_checkpoints_m.get("subdir", "m_best"),
            monitor=config.train.model_checkpoints_m.monitor,
            mode="min",
            save_top_k=1,
            verbose=config.train.model_checkpoints_m.get("verbose", False),
            save_last=False,
        ))
    return callbacks


def warm_start_forward(config: DictConfig, pretrain_dir: str):
    """Build the split-head forward model and warm-start it from the released 400-d
    checkpoint. Returns (model, ckpt_path) and asserts the initialisation invariants:
    the total head is byte-identical to the released head and the m head is exactly
    zero."""
    model = CSPProperty(**config)
    dec = model.decoder
    assert getattr(dec, "node_out_split", False), \
        "expected diffusion.model.node_out_split=True for the split-head warm start"

    ckpt_paths = glob(str(Path(pretrain_dir) / "*.ckpt"))
    if len(ckpt_paths) != 1:
        raise ValueError(f"expected 1 ckpt in {pretrain_dir}, found {ckpt_paths}")
    sd = torch.load(ckpt_paths[0], map_location="cpu", weights_only=False)["state_dict"]

    # The released head is a single 400-d node_out; ours is split. Pop it, load the rest
    # of the backbone directly, then graft the total half and zero the m half.
    ckpt_node_w = sd.pop("decoder.node_out.weight")  # [400, hidden]
    missing, unexpected = model.load_state_dict(sd, strict=False)

    with torch.no_grad():
        dec.node_out_total.weight.copy_(ckpt_node_w)
        dec.node_out_m.weight.zero_()

    assert torch.equal(dec.node_out_total.weight, ckpt_node_w), "total head graft is not exact"
    assert float(dec.node_out_m.weight.abs().max()) == 0.0, "m head is not zero at init"
    assert set(missing) == {"decoder.node_out_total.weight", "decoder.node_out_m.weight"}, \
        f"unexpected missing keys after backbone load: {missing}"
    assert list(unexpected) == [], f"unexpected keys after backbone load: {unexpected}"

    print("warm start (split head): backbone loaded strict on everything but the head")
    print("  node_out_total <- released node_out", tuple(dec.node_out_total.weight.shape),
          "| node_out_m zero-init", tuple(dec.node_out_m.weight.shape),
          "| max|W_m| =", float(dec.node_out_m.weight.abs().max()))
    return model, ckpt_paths[0]


def build_split_scaler(data_module, pretrain_dir: str, half: int):
    """prop_scaler = [released total stats(half) || dataset m stats(half)]; the released
    lattice_scaler is reused. Keeping the released total stats means the grafted total
    head sees the calibration it was trained with."""
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
        raise ValueError(f"the split head needs an even pred_dim, got {pred_dim}")
    half = pred_dim // 2

    # The datamodule fits a full-width scaler on the training split; replace the total
    # half with the released statistics so the grafted head stays calibrated.
    data_module = CrystalDataModule(config)
    scaler, pre_lat = build_split_scaler(data_module, config.pretrain_dir, half)
    data_module.scaler = scaler
    data_module.lattice_scaler = pre_lat
    print(f"prop_scaler: width {scaler.means.shape[0]} "
          f"| total half from the released model, m half fit on this dataset")

    model, ckpt = warm_start_forward(config, config.pretrain_dir)
    print("warm-started from", ckpt, "| base lr =", model.hparams.optim.params.lr,
          "| new_pathway_lr =", model.hparams.optim.get("new_pathway_lr", None))

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

    # Per-epoch metrics CSV alongside the checkpoint, including the total and m halves.
    csv_logger = CSVLogger(str(save_dir), name="csv")
    loggers = [csv_logger] + ([wandb_logger] if wandb_logger is not None else [])

    trainer = pl.Trainer(
        default_root_dir=save_dir, logger=loggers, callbacks=callbacks,
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
    parser.add_argument("--config", default="configs/dos_forward_dmx2_spin_ft.yml")
    args = parser.parse_args()
    conf = OmegaConf.load(args.config)
    print(OmegaConf.to_yaml(conf))
    run(conf)
