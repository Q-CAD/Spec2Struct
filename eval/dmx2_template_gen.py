"""Defect-template generation on val (DOS -> defect in a known host).

Template construction per defective val target (needs a pristine host ref):
  - GT defect signature vs the pristine reference (shared frame, no alignment).
  - PINNED atoms = pristine host sites matched to a same-species GT atom;
    they carry HOST truth: pristine fractional coords + species, held fixed at
    every denoising step; lattice fixed to the host cell.
  - FREE atoms = the remaining GT atom count (additions + substitution sites);
    they start from pure noise. NO diff_ratio warm start anywhere.
  - Per-atom DOS conditioning y (target's own, scaled with the generator's
    prop_scaler) on ALL atoms: pinned atoms get their matched GT atom's y row,
    free slots get the defect-involved GT atoms' y rows.

Outputs a generate_for_eval-format blob (preds[k][N]) so dmx2_roundtrip.py and
the D1/D2 machinery apply unchanged. --smoke runs 1 vacancy + 1 substitution
target at k=1 with hard assertions on the pinning invariants.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from glob import glob
from omegaconf import OmegaConf
from torch_geometric.data import Data, Batch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))
import pandas as pd
from make_dmx2_split import coarse_class
from dmx2_defect_geometry import (load_split_records, frac_coords,
                                  defect_signature, match_sites)
from dmx_eval_utils import lattices_to_params

from dosmatgen.diffusion.diffusion_cfg import CSPDiffusion

GEN_DIR = REPO / "outputs/260714_125010_dmx2_cfg_ft"
GEN_CKPT = GEN_DIR / "epoch=789-step=12640.ckpt"


def build_template(target_rec, pris_rec, scaler):
    """-> (Data with pin info, meta dict) or None if target has no clean map."""
    tf, tcell = frac_coords(target_rec)
    tz = np.asarray(target_rec["atomic_numbers"])
    pf, pcell = frac_coords(pris_rec)
    pz = np.asarray(pris_rec["atomic_numbers"])
    s2a, a2s, _ = match_sites(pf, pz, pcell, tf, tz)

    pinned_sites = [si for si in range(len(pz))
                    if s2a[si] != -1 and pz[si] == tz[s2a[si]]]
    defect_atoms = [ai for ai in range(len(tz))
                    if a2s[ai] == -1 or pz[a2s[ai]] != tz[ai]]
    n = len(pinned_sites) + len(defect_atoms)
    if n != len(tz):
        return None  # inconsistent mapping; skip

    y = np.asarray(target_rec["y"])
    coords = np.concatenate([pf[pinned_sites],
                             np.zeros((len(defect_atoms), 3))])
    types = np.concatenate([pz[pinned_sites], tz[defect_atoms]])
    # NOTE free-slot types are placeholders for batch construction only — the
    # sampler re-noises them; but pinned rows' types ARE the pinned truth.
    yrows = np.concatenate([y[[s2a[si] for si in pinned_sites]],
                            y[defect_atoms]])
    lengths, angles = lattices_to_params(pcell)
    d = Data(
        structure_id=target_rec["structure_id"],
        frac_coords=torch.tensor(coords, dtype=torch.float),
        atom_types=torch.tensor(types, dtype=torch.long),
        lengths=torch.tensor(lengths, dtype=torch.float).view(1, -1),
        angles=torch.tensor(angles, dtype=torch.float).view(1, -1),
        y=scaler.transform(torch.tensor(yrows, dtype=torch.float)),
        pin=torch.tensor([True] * len(pinned_sites) + [False] * len(defect_atoms)),
        num_atoms=n, num_nodes=n)
    meta = dict(n_pinned=len(pinned_sites), n_free=len(defect_atoms))
    return d, meta


def out_to_dicts(out, sids):
    fc = out["frac_coords"].cpu().numpy()
    at = out["atom_types"].cpu().numpy()
    lat = out["lattices"].cpu().numpy()
    na = out["num_atoms"].cpu().numpy().tolist()
    res, start = [], 0
    for i, n in enumerate(na):
        lengths, angles = lattices_to_params(lat[i])
        res.append(dict(structure_id=sids[i], frac_coords=fc[start:start + n] % 1.0,
                        atom_types=at[start:start + n], lengths=lengths,
                        angles=angles))
        start += n
    return res


def main(args):
    recs = load_split_records()
    audit = pd.read_csv(REPO / "data/dmx2_audit.csv").set_index("id")
    with open(REPO / "data/dmx2_dos/val.json") as f:
        val = json.load(f)

    cfg = OmegaConf.load(GEN_DIR / "hparams.yaml")
    model = CSPDiffusion.load_from_checkpoint(
        str(GEN_CKPT), **cfg, strict=False, weights_only=False)
    model.to("cuda").eval()
    scaler = torch.load(GEN_DIR / "prop_scaler.pt", map_location="cpu",
                        weights_only=False)

    targets = []
    for r in val:
        sid = r["structure_id"]
        host = sid.split("_")[0]
        if audit.loc[sid, "defect"] == "Defect-Free":
            continue
        if f"{host}_Defect-Free" not in recs:
            continue
        if args.classes and coarse_class(audit.loc[sid, "defect"]) not in \
                args.classes.split(","):
            continue
        targets.append(r)
    if args.smoke:
        vac = next(r for r in targets
                   if audit.loc[r["structure_id"], "defect"].startswith("Vacancy"))
        sub = next(r for r in targets
                   if coarse_class(audit.loc[r["structure_id"], "defect"]) == "substitution")
        targets, k = [vac, sub], 1
    else:
        k = args.k
    print(f"{len(targets)} template targets, k={k}")

    data_list, metas, skipped = [], [], []
    for r in targets:
        host = r["structure_id"].split("_")[0]
        built = build_template(r, recs[f"{host}_Defect-Free"], scaler)
        if built is None:
            skipped.append(r["structure_id"])
            continue
        data_list.append(built[0])
        metas.append(built[1])
    print(f"built {len(data_list)} templates (skipped: {skipped}); "
          f"free atoms per template: "
          f"{sorted(set(m['n_free'] for m in metas))}")

    sids = [d.structure_id for d in data_list]
    gt = []
    for sid in sids:
        r = next(x for x in val if x["structure_id"] == sid)
        f, cell = frac_coords(r)
        lengths, angles = lattices_to_params(cell)
        gt.append(dict(structure_id=sid, frac_coords=f % 1.0,
                       atom_types=np.asarray(r["atomic_numbers"]),
                       lengths=lengths, angles=angles))

    preds = []
    for c in range(k):
        big = Batch.from_data_list(data_list).to("cuda")
        out, _ = model.template_cfg_sample(big, big.pin, step_lr=args.step_lr,
                                           w=args.w)
        dicts = out_to_dicts(out, [d.structure_id for d in data_list])

        if args.smoke:
            for d0, dd, m in zip(data_list, dicts, metas):
                npin = m["n_pinned"]
                tpl_f = d0.frac_coords.numpy()
                assert np.abs((dd["frac_coords"][:npin] - tpl_f[:npin] + 0.5) % 1.0
                              - 0.5).max() < 1e-5, "pinned coords moved"
                assert (dd["atom_types"][:npin]
                        == d0.atom_types.numpy()[:npin]).all(), "pinned types changed"
                assert np.allclose(dd["lengths"], d0.lengths.numpy().ravel(),
                                   atol=1e-4), "lattice lengths changed"
                assert np.allclose(dd["angles"], d0.angles.numpy().ravel(),
                                   atol=1e-3), "lattice angles changed"
                free_moved = (np.abs((dd["frac_coords"][npin:]
                                      - tpl_f[npin:] + 0.5) % 1.0 - 0.5).max()
                              if m["n_free"] else 0.0)
                assert m["n_free"] == 0 or free_moved > 1e-3, "free atoms did not move"
                print(f"  SMOKE OK {dd['structure_id']}: pinned {npin} fixed, "
                      f"{m['n_free']} free moved (max disp {free_moved:.3f} frac)")
        preds.append(dicts)
        print(f"  candidate {c + 1}/{k} done")

    blob = {
        "meta": {"model": "dmx2_template", "split": "val", "k": k,
                 "diff_ratio": 1.0, "step_lr": args.step_lr, "w": args.w,
                 "unconditional": False, "template": True,
                 "root_path": str(GEN_DIR), "ckpt": str(GEN_CKPT),
                 "pred_dim": 400},
        "structure_ids": sids,
        "gt": gt,
        "preds": preds,
        "n_pinned": {s: m["n_pinned"] for s, m in zip(sids, metas)},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, args.out)
    print(f"saved {len(sids)} x {k} -> {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--classes", default="",
                   help="comma-separated coarse classes to keep (default all)")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--w", type=float, default=1.0)
    p.add_argument("--step_lr", type=float, default=1e-5)
    p.add_argument("--out", default="eval/preds/dmx2_val_template_k5_w1.pt")
    args = p.parse_args()
    main(args)
