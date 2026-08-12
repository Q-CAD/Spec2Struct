"""Corrections to the template-vs-free comparison.

(i) Vacancy-class template runs have n_free = 0 (the defect is an absence, so
    every atom is pinned): the "generated" candidate is fully determined by the
    template construction — no generative content. Verify determinism from the
    blob (n_free) and the per-candidate round-trip CSV (within-target MAE
    spread across k must be 0), then recompute the template-vs-free table
    RESTRICTED to non-vacancy targets (the generative framing).

(ii) Framings, made explicit:
    A "construction+generation" (all 53 defective targets):
      credits the template arm for vacancy targets its builder fully determines.
    B "generation only" (non-vacancy targets): what the diffusion model itself
      contributes given host + defect-site count; species + geometry generated.

Output: eval/results/dmx2_generator_eval/template_vs_free_nonvacancy.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))
from dmx2_template_compare import arm_metrics, JUDGE_DIR, JUDGE_CKPT
from dosmatgen.diffusion.property import CSPProperty
from make_dmx2_split import coarse_class
from dmx2_defect_geometry import load_split_records
from omegaconf import OmegaConf

GEV = REPO / "eval/results/dmx2_generator_eval"


def main():
    audit = pd.read_csv(REPO / "data/dmx2_audit.csv").set_index("id")
    blob = torch.load(REPO / "eval/preds/dmx2_val_template_k5_w1.pt",
                      weights_only=False)
    sids = blob["structure_ids"]
    cls = {s: coarse_class(audit.loc[s, "defect"]) for s in sids}
    natoms = {g["structure_id"]: len(g["atom_types"]) for g in blob["gt"]}
    nfree = {s: natoms[s] - blob["n_pinned"][s] for s in sids}

    # (i) determinism check
    vac = sorted(s for s in sids if cls[s] == "vacancy")
    print(f"{len(sids)} template targets; vacancy class: {len(vac)}")
    print("n_free by class:")
    print(pd.Series(nfree).groupby(pd.Series(cls)).agg(["min", "max"]))
    pre = pd.read_csv(GEV / "val_template_k5_w1.csv")
    spread = pre.groupby("structure_id").mae_full.std()
    det = spread[spread.index.isin([s for s in sids if nfree[s] == 0])]
    nondet = spread[spread.index.isin([s for s in sids if nfree[s] > 0])]
    print(f"\nwithin-target MAE std across k, n_free=0 targets "
          f"({len(det)}): max {det.max():.2e}")
    print(f"within-target MAE std across k, n_free>0 targets "
          f"({len(nondet)}): median {nondet.median():.4f}")
    assert det.max() < 1e-6, "n_free=0 targets are NOT deterministic?!"
    # (observed max ~6e-9: pure float jitter from judge batching, not sampling)
    zero_free_nonvac = [s for s in sids if nfree[s] == 0 and cls[s] != "vacancy"]
    print(f"n_free=0 outside the vacancy class: {zero_free_nonvac}")

    # (ii) recompute both arms restricted to non-vacancy targets
    recs = load_split_records()
    with open(REPO / "data/dmx2_dos/val.json") as f:
        tgt = {r["structure_id"]: np.asarray(r["y"]) for r in json.load(f)}
    cfg = OmegaConf.load(JUDGE_DIR / "hparams.yaml")
    model = CSPProperty(**cfg)
    sd = torch.load(JUDGE_CKPT, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(sd, strict=True)
    model.to("cuda").eval()
    scaler = torch.load(JUDGE_DIR / "prop_scaler.pt", map_location="cpu",
                        weights_only=False)

    framings = {
        "nonvac": {s for s in sids if cls[s] != "vacancy"},
        "nfree_gt0": {s for s in sids if nfree[s] > 0},
    }
    cols = {}
    for tag, keep in framings.items():
        print(f"\n{tag} framing: {len(keep)} targets "
              f"({pd.Series([cls[s] for s in keep]).value_counts().to_dict()})")
        tm, _ = arm_metrics(REPO / "eval/preds/dmx2_val_template_k5_w1.pt",
                            "val_template", GEV / "val_template_k5_w1.csv",
                            recs, audit, tgt, model, scaler, align=False,
                            keep=keep)
        fm, _ = arm_metrics(REPO / "eval/preds/dmx2_val_k5_w1.pt",
                            "val_free", GEV / "sweep_w1.csv",
                            recs, audit, tgt, model, scaler, align=True,
                            keep=keep)
        cols[f"template_{tag}"] = tm
        cols[f"free_{tag}"] = fm
    df = pd.DataFrame(cols)
    df.to_csv(GEV / "template_vs_free_nonvacancy.csv")
    print("\n" + df.round(4).to_string())


if __name__ == "__main__":
    main()
