"""Forward-model test metrics (structure -> total DOS, 400-d).

Predicts per-atom total DOS for the test split with the best forward ckpt,
inverse-transformed to physical units (states/eV/atom), and reports MAE in three
energy windows at two levels, against two training-set baselines:

  windows: full [-10, 10] | all-valid [-10, +5.5] (covered by every structure
           per the audit; above +5.5 some targets are zero-padding artifacts)
           | near-E_F [-2, 2]
  levels : per-atom          mean |pred - gt| over atoms x window bins
           whole-cell /atom  MAE over window bins of the cell-summed DOS, / natoms
  baselines: (i) global train-mean per-atom curve
             (ii) per-element train-mean curve (fallback: global mean)

Outputs eval/results/dmx2_forward_eval/{metrics.csv, summary.md, overlays/*.png}
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# The released checkpoints and scalers were pickled under the upstream package
# name 'spectrodiff'; alias it to this repo's 'dosmatgen' package so torch.load
# can resolve those classes.
import importlib
for _r, _a in [("dosmatgen", "spectrodiff"), ("dosmatgen.utils", "spectrodiff.utils"),
               ("dosmatgen.utils.data", "spectrodiff.utils.data")]:
    sys.modules.setdefault(_a, importlib.import_module(_r))

from dosmatgen.diffusion.property import CSPProperty
from dosmatgen.dataset.datamodule import CrystalDataModule
from make_dmx2_split import coarse_class

EGRID = np.linspace(-10, 10, 400)
WINDOWS = {
    "full": EGRID <= np.inf,
    "allvalid": EGRID <= 5.5,
    "nearef": (EGRID >= -2) & (EGRID <= 2),
}


@torch.no_grad()
def predict_split(run_dir, ckpt, split="test"):
    """physical per-atom DOS predictions {sid: [n,400]} + scaled-y roundcheck."""
    cfg = OmegaConf.load(run_dir / "hparams.yaml")
    model = CSPProperty(**cfg)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=True)
    model.to("cuda").eval()
    scaler = torch.load(run_dir / "prop_scaler.pt", map_location="cpu", weights_only=False)

    dm = CrystalDataModule(cfg, scaler_path=str(run_dir))
    if split == "test":
        dm.setup(stage="test")
        loader = dm.test_dataloader()
    else:
        raise ValueError(split)

    preds = {}
    for batch in loader:
        batch = batch.to("cuda")
        pred, _ = model.infer(batch)
        pred = scaler.inverse_transform(pred.cpu()).numpy()
        off = 0
        for i, sid in enumerate(batch.structure_id):
            n = int(batch.num_atoms[i])
            preds[sid] = pred[off:off + n]
            off += n
    return preds


def window_maes(pred, gt, natoms):
    """dict of per-atom and cell-level MAE per window for one structure."""
    out = {}
    cell_p, cell_g = pred.sum(0), gt.sum(0)
    for wname, mask in WINDOWS.items():
        out[f"peratom_{wname}"] = float(np.abs(pred[:, mask] - gt[:, mask]).mean())
        out[f"cell_{wname}"] = float(np.abs(cell_p[mask] - cell_g[mask]).mean() / natoms)
    return out


def main(args):
    run_dir = Path(args.run_dir)
    out_dir = REPO / "eval/results/dmx2_forward_eval"
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)

    with open(REPO / "data/dmx2_dos/test.json") as f:
        test = {r["structure_id"]: r for r in json.load(f)}
    with open(REPO / "data/dmx2_dos/train.json") as f:
        train = json.load(f)

    # baselines from train
    all_atoms = np.concatenate([np.asarray(r["y"]) for r in train], axis=0)
    global_mean = all_atoms.mean(axis=0)                       # [400]
    by_elem_sum, by_elem_n = {}, {}
    for r in train:
        y = np.asarray(r["y"])
        for z, row in zip(r["atomic_numbers"], y):
            by_elem_sum[z] = by_elem_sum.get(z, 0) + row
            by_elem_n[z] = by_elem_n.get(z, 0) + 1
    elem_mean = {z: by_elem_sum[z] / by_elem_n[z] for z in by_elem_sum}
    print(f"baselines: global from {len(all_atoms)} train atoms; "
          f"{len(elem_mean)} elements (test elements missing from train get global)")

    preds = predict_split(run_dir, args.ckpt)
    assert set(preds) == set(test), "prediction/test id mismatch"

    audit = pd.read_csv(REPO / "data/dmx2_audit.csv").set_index("id")

    rows = []
    for sid, rec in test.items():
        gt = np.asarray(rec["y"])
        n = len(rec["atomic_numbers"])
        base_g = np.tile(global_mean, (n, 1))
        base_e = np.stack([elem_mean.get(z, global_mean) for z in rec["atomic_numbers"]])
        row = dict(structure_id=sid, natoms=n,
                   host=audit.loc[sid, "host"],
                   coarse_class=coarse_class(audit.loc[sid, "defect"]))
        for tag, p in [("model", preds[sid]), ("base_global", base_g), ("base_elem", base_e)]:
            for k, v in window_maes(p, gt, n).items():
                row[f"{tag}_{k}"] = v
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("structure_id")
    df.to_csv(out_dir / "metrics.csv", index=False)

    # aggregates (unweighted mean over structures)
    agg = df.drop(columns=["structure_id", "natoms", "host", "coarse_class"]).mean()
    by_class = df.groupby("coarse_class")[
        [c for c in df.columns if c.startswith(("model_", "base_"))]].mean()

    lines = ["# dmx2 forward-model test metrics (62 structures)\n",
             f"ckpt: `{args.ckpt}`\n",
             "MAE in states/eV/atom; mean over test structures.\n",
             "\n## Aggregate\n",
             "| metric | model | base_global | base_elem |",
             "|---|---|---|---|"]
    for lvl in ("peratom", "cell"):
        for w in WINDOWS:
            lines.append(f"| {lvl} {w} | {agg[f'model_{lvl}_{w}']:.4f} "
                         f"| {agg[f'base_global_{lvl}_{w}']:.4f} "
                         f"| {agg[f'base_elem_{lvl}_{w}']:.4f} |")
    lines += ["\n## By coarse defect class (model / base_elem, per-atom full window)\n",
              "| class | n | model | base_elem |", "|---|---|---|---|"]
    for cls, sub in df.groupby("coarse_class"):
        lines.append(f"| {cls} | {len(sub)} | {sub.model_peratom_full.mean():.4f} "
                     f"| {sub.base_elem_peratom_full.mean():.4f} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # overlays: median-model-MAE structure of each coarse class (up to 6)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    picked = []
    for cls, sub in df.groupby("coarse_class"):
        m = sub.sort_values("model_peratom_full")
        picked.append(m.iloc[len(m) // 2].structure_id)
    for sid in picked[:6]:
        rec = test[sid]
        gt = np.asarray(rec["y"])
        n = len(rec["atomic_numbers"])
        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.plot(EGRID, gt.sum(0) / n, "k-", lw=1.3, label="DFT target")
        ax.plot(EGRID, preds[sid].sum(0) / n, "r-", lw=1.0, label="forward model")
        ax.plot(EGRID, global_mean, "b:", lw=0.9, label="global train mean")
        ax.axvline(0, c="gray", lw=0.5, ls=":")
        ax.axvline(5.5, c="gray", lw=0.5, ls="--")
        r = df[df.structure_id == sid].iloc[0]
        ax.set_title(f"{sid} [{r.coarse_class}] cell-DOS/atom — "
                     f"model MAE {r.model_peratom_full:.3f} (per-atom, full)")
        ax.set_xlabel("E - E_F (eV)")
        ax.set_ylabel("DOS (states/eV/atom)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "overlays" / f"{sid}.png", dpi=140)
        plt.close(fig)
    print(f"overlays: {picked[:6]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", default="outputs/260714_121340_dmx2_forward_ft")
    p.add_argument("--ckpt", default="outputs/260714_121340_dmx2_forward_ft/epoch=89-step=1440.ckpt")
    args = p.parse_args()
    main(args)
