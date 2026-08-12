"""Generate DOS-conditioned structures for reconstruction eval (A or C).

For every structure in the chosen split (test or val) we feed its DOS condition
(A: total-400 in batch.y; C: [total||m]-800 in batch.y) and generate k candidate
structures, preserving structure_id so each prediction is paired with its GT.

IMPORTANT — diff_ratio:
  cfg_sample(diff_ratio<1) seeds the reverse diffusion from the *ground-truth
  structure noised to t=diff_ratio*T (diffusion_cfg.py:351-367). That leaks the GT
  geometry into the sample, which would inflate a reconstruction metric. For a fair
  CSP-style RecEval we use diff_ratio=1.0 (pure noise): the only information given to
  the model is the DOS condition (batch.y) and the atom count (batch.num_atoms).

Output: a .pt blob consumed by dmx2_roundtrip.py:
  {'meta', 'structure_ids':[N], 'gt':[N], 'preds':[k][N]}  (crys_array_dicts)
"""
import argparse
from glob import glob
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

import sys
sys.path.append(".")
sys.path.append("eval")

from dosmatgen.diffusion.diffusion_cfg import CSPDiffusion
from dosmatgen.dataset.datamodule import CrystalDataModule
from dmx_eval_utils import lattices_to_params


def split_to_dicts(frac_coords, atom_types, lattices, num_atoms, structure_id):
    """Split a (possibly multi-structure) batch tensor set into per-structure dicts."""
    frac_coords = frac_coords.detach().cpu().numpy()
    atom_types = atom_types.detach().cpu().numpy()
    lattices = lattices.detach().cpu().numpy()      # [B,3,3]
    num_atoms = num_atoms.detach().cpu().numpy().tolist()
    out = []
    start = 0
    for i, na in enumerate(num_atoms):
        end = start + na
        lengths, angles = lattices_to_params(lattices[i])
        out.append({
            "structure_id": structure_id[i],
            "frac_coords": frac_coords[start:end] % 1.0,
            "atom_types": atom_types[start:end],     # Z ints
            "lengths": lengths,
            "angles": angles,
        })
        start = end
    return out


def gt_to_dicts(batch):
    """Ground-truth per-structure dicts from a batch (lengths/angles are physical)."""
    from dosmatgen.utils.data import lattice_params_to_matrix_torch
    lat = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
    return split_to_dicts(batch.frac_coords, batch.atom_types, lat,
                          batch.num_atoms, batch.structure_id)


def main(args):
    root = Path(args.root_path)
    config = OmegaConf.load(root / "hparams.yaml")

    if args.ckpt:
        ckpts = [args.ckpt]
    else:
        ckpts = glob(str(root / "*.ckpt"))
        assert len(ckpts) == 1, f"expected 1 ckpt in {root}, found {ckpts}; use --ckpt"
    name = torch.cuda.get_device_name(0)
    print(f"GPU: {name} | ckpt: {ckpts[0]}")

    model = CSPDiffusion.load_from_checkpoint(ckpts[0], config=config, weights_only=False)
    model.to("cuda").eval()

    # one batch covers the whole split unless overridden
    config.datamodule.batch_size.test = args.batch_size
    config.datamodule.batch_size.val = args.batch_size

    from torch_geometric.data import Batch

    dm = CrystalDataModule(config, scaler_path=str(root))
    if args.split == "test":
        dm.setup(stage="test")
        dataset = dm.test_dataset
    elif args.split == "val":
        dm.setup(stage="fit")          # builds val (and train); caches make it fast
        dataset = dm.val_dataset
    else:
        raise ValueError(args.split)

    # The whole split is one batch. Candidates are produced by TILING the split
    # k-fold into a single batched pass: each replica is an independent graph with
    # its own diffusion noise (keep_coords/keep_lattice are False), so identical
    # structures denoise to distinct candidates. B200 has ample memory; tile_chunk
    # caps replicas/pass because cfg_sample retains the full per-timestep trajectory.
    data_list = [dataset[i] for i in range(len(dataset))]
    N = len(data_list)

    gt_batch = Batch.from_data_list(data_list).to("cuda")
    gt = gt_to_dicts(gt_batch)
    structure_ids = [d["structure_id"] for d in gt]
    print(f"split={args.split}: N={N} structures (one batch); "
          f"pred_dim(y)={gt_batch.y.shape[-1]}; k={args.k}; tile_chunk={args.tile_chunk}")

    preds = [None] * args.k
    done = 0
    while done < args.k:
        tile = min(args.tile_chunk, args.k - done)
        big = Batch.from_data_list(data_list * tile).to("cuda")
        if args.unconditional:
            # CFG null branch only: decoder skips the DOS projection entirely
            # (cspnet_cfg.unconditional); batch.y is passed but unused.
            out, _ = model.sample(big, step_lr=args.step_lr,
                                  diff_ratio=args.diff_ratio,
                                  unconditional=True, conditional=False)
        else:
            out, _ = model.cfg_sample(big, step_lr=args.step_lr,
                                      diff_ratio=args.diff_ratio, w=args.w)
        pred_dicts = split_to_dicts(out["frac_coords"], out["atom_types"],
                                    out["lattices"], out["num_atoms"],
                                    big.structure_id)
        assert len(pred_dicts) == N * tile
        for c in range(tile):
            preds[done + c] = pred_dicts[c * N:(c + 1) * N]
        done += tile
        print(f"  candidates {done}/{args.k} done (tiled pass of {tile} x {N})")

    # sanity: alignment
    for j in range(args.k):
        assert len(preds[j]) == N
        for i in range(N):
            assert preds[j][i]["structure_id"] == structure_ids[i]

    blob = {
        "meta": {
            "model": args.model, "split": args.split, "k": args.k,
            "diff_ratio": args.diff_ratio, "step_lr": args.step_lr, "w": args.w,
            "unconditional": bool(args.unconditional),
            "root_path": str(root), "ckpt": ckpts[0],
            "pred_dim": int(config.diffusion.model.pred_dim),
            "device": name,
        },
        "structure_ids": structure_ids,
        "gt": gt,
        "preds": preds,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, args.out)
    print(f"Saved {N} structures x {args.k} candidates -> {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root_path", required=True, help="run dir w/ ckpt+hparams+scalers")
    p.add_argument("--ckpt", default=None,
                   help="explicit ckpt path (needed when the run dir also has last.ckpt)")
    p.add_argument("--split", choices=["test", "val"], required=True)
    p.add_argument("--k", type=int, required=True, help="candidates per spectrum")
    p.add_argument("--model", default="", help="label, e.g. A or C")
    p.add_argument("--out", required=True)
    p.add_argument("--diff_ratio", type=float, default=1.0,
                   help="1.0 = pure-noise generation (no GT leakage); <1 noises the GT")
    p.add_argument("--step_lr", type=float, default=1e-5)
    p.add_argument("--w", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--tile_chunk", type=int, default=10,
                   help="max candidate replicas per batched pass (memory cap)")
    p.add_argument("--unconditional", action="store_true",
                   help="sample the CFG null branch only (ignores the DOS condition); "
                        "w is irrelevant in this mode")
    args = p.parse_args()
    main(args)
