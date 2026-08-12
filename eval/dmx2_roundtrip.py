"""Round-trip scorer for dmx2 total-400 generation (DOS -> structure -> DOS).

Scores a predictions blob from generate_for_eval.py with the dmx2 forward model
as judge: for every candidate structure, predict its total DOS (physical units)
and MAE it against the conditioning target. 400-d only — no total/m split, no
spin-flip (that logic lives in the 800-era roundtrip_val.py, not here).

Also per candidate: structural validity (min pairwise dist > 0.5 A), smact
compositional validity, and composition-vs-target stats (exact reduced-formula
match, element-set match). Fingerprint/coverage metrics are not computed here.

Output: per-candidate CSV (one row per (candidate k, structure)) + printed and
JSON aggregate. Aggregates report mean-over-all-k and best-of-k round-trip MAE.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from glob import glob
from omegaconf import OmegaConf
from torch_geometric.data import Data, Batch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))

# The released checkpoints and scalers were pickled under the upstream package
# name 'spectrodiff'; alias it to this repo's 'dosmatgen' package so torch.load
# can resolve those classes.
import importlib
for _r, _a in [("dosmatgen", "spectrodiff"), ("dosmatgen.utils", "spectrodiff.utils"),
               ("dosmatgen.utils.data", "spectrodiff.utils.data")]:
    sys.modules.setdefault(_a, importlib.import_module(_r))

from dosmatgen.diffusion.property import CSPProperty
from dmx_eval_utils import smact_validity, structure_validity

EGRID = np.linspace(-10, 10, 400)
NEAREF = (EGRID >= -2) & (EGRID <= 2)


def to_data(d):
    n = len(d["atom_types"])
    return Data(
        structure_id=d["structure_id"],
        frac_coords=torch.tensor(np.asarray(d["frac_coords"]), dtype=torch.float),
        atom_types=torch.tensor(np.asarray(d["atom_types"]), dtype=torch.long),
        lengths=torch.tensor(np.asarray(d["lengths"]), dtype=torch.float).view(1, -1),
        angles=torch.tensor(np.asarray(d["angles"]), dtype=torch.float).view(1, -1),
        num_atoms=n, num_nodes=n)


@torch.no_grad()
def judge_dos(dicts, model, scaler, bs=32):
    """physical [n,400] DOS per structure dict, keyed by list index."""
    out = [None] * len(dicts)
    for i in range(0, len(dicts), bs):
        chunk = dicts[i:i + bs]
        b = Batch.from_data_list([to_data(d) for d in chunk]).to("cuda")
        pred, _ = model.infer(b)
        pred = scaler.inverse_transform(pred.cpu()).numpy()
        off = 0
        for j, d in enumerate(chunk):
            n = len(d["atom_types"])
            out[i + j] = pred[off:off + n]
            off += n
    return out


def comp_key(atom_types):
    c = Counter(int(z) for z in atom_types)
    counts = np.array([c[z] for z in sorted(c)])
    counts = counts // np.gcd.reduce(counts)
    return tuple(zip(sorted(c), counts.tolist()))


def candidate_row(d, gt_d, tgt_y, judge_y):
    from pymatgen.core import Structure, Lattice
    row = {}
    row["mae_full"] = float(np.abs(judge_y - tgt_y).mean())
    row["mae_nearef"] = float(np.abs(judge_y[:, NEAREF] - tgt_y[:, NEAREF]).mean())
    try:
        s = Structure(Lattice.from_parameters(
            *(np.asarray(d["lengths"], float).tolist()
              + np.asarray(d["angles"], float).tolist())),
            d["atom_types"], d["frac_coords"], coords_are_cartesian=False)
        row["struct_valid"] = bool(structure_validity(s))
    except Exception:
        row["struct_valid"] = False
    try:
        c = Counter(int(z) for z in d["atom_types"])
        elems = tuple(sorted(c))
        counts = np.array([c[z] for z in elems])
        counts = counts // np.gcd.reduce(counts)
        row["comp_valid"] = bool(smact_validity(elems, tuple(counts.tolist())))
    except Exception:
        row["comp_valid"] = False
    row["valid"] = row["struct_valid"] and row["comp_valid"]
    row["comp_exact_match"] = comp_key(d["atom_types"]) == comp_key(gt_d["atom_types"])
    row["elemset_match"] = (set(int(z) for z in d["atom_types"])
                            == set(int(z) for z in gt_d["atom_types"]))
    return row


def main(args):
    blob = torch.load(args.pred_blob, weights_only=False)
    sids = blob["structure_ids"]
    preds = blob["preds"]                # [K][N]
    gt = blob["gt"]
    K, N = len(preds), len(sids)
    print(f"[rt] {args.pred_blob}: N={N} K={K} w={blob['meta'].get('w')} "
          f"uncond={blob['meta'].get('unconditional', False)}")
    assert blob["meta"]["diff_ratio"] == 1.0, "GT-leaking diff_ratio<1 blob refused"

    run_dir = Path(args.judge_dir)
    cfg = OmegaConf.load(run_dir / "hparams.yaml")
    model = CSPProperty(**cfg)
    sd = torch.load(args.judge_ckpt, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(sd, strict=True)
    model.to("cuda").eval()
    scaler = torch.load(run_dir / "prop_scaler.pt", map_location="cpu", weights_only=False)

    with open(args.target_json) as f:
        tgt = {r["structure_id"]: np.asarray(r["y"]) for r in json.load(f)}

    rows = []
    for k in range(K):
        dos = judge_dos(preds[k], model, scaler)
        for i, sid in enumerate(sids):
            r = dict(structure_id=sid, k=k,
                     **candidate_row(preds[k][i], gt[i], tgt[sid], dos[i]))
            rows.append(r)
        print(f"  candidate set {k + 1}/{K} judged")
    df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    best = df.groupby("structure_id")[["mae_full", "mae_nearef"]].min()
    agg = {
        "pred_blob": str(args.pred_blob), "N": N, "K": K,
        "w": blob["meta"].get("w"),
        "unconditional": bool(blob["meta"].get("unconditional", False)),
        "mae_full_meank": float(df.mae_full.mean()),
        "mae_full_bestofk": float(best.mae_full.mean()),
        "mae_nearef_meank": float(df.mae_nearef.mean()),
        "mae_nearef_bestofk": float(best.mae_nearef.mean()),
        "valid_rate": float(df.valid.mean()),
        "struct_valid_rate": float(df.struct_valid.mean()),
        "comp_valid_rate": float(df.comp_valid.mean()),
        "comp_exact_match_rate": float(df.comp_exact_match.mean()),
        "elemset_match_rate": float(df.elemset_match.mean()),
    }
    print(json.dumps(agg, indent=2))
    Path(args.out_csv).with_suffix(".agg.json").write_text(json.dumps(agg, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pred_blob", required=True)
    p.add_argument("--judge_dir", default="outputs/260714_121340_dmx2_forward_ft")
    p.add_argument("--judge_ckpt",
                   default="outputs/260714_121340_dmx2_forward_ft/epoch=89-step=1440.ckpt")
    p.add_argument("--target_json", required=True)
    p.add_argument("--out_csv", required=True)
    args = p.parse_args()
    main(args)
