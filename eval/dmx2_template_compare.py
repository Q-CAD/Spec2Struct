"""Scoring: template-constrained vs free generation on defective val.

Both arms: k=5, w=1, same 53 targets (defective val with pristine reference).
  pre-relax round-trip  : from the per-candidate roundtrip CSVs
  post-relax round-trip : judge on the MACE-relaxed geometries of both arms
  D1 stoichiometry      : composition delta vs pristine == GT delta
  geometric recovery    : defect_signature status per candidate; template arm
                          maps in the pristine frame directly (align=False —
                          pinned atoms sit on pristine sites by construction),
                          free arm needs the origin/lattice-op alignment search.

Output: eval/results/dmx2_generator_eval/template_vs_free.csv + printed table.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
from make_dmx2_split import coarse_class
from dmx2_defect_geometry import (load_split_records, frac_coords,
                                  defect_signature, sig_key, addition_placement)
from dmx2_defect_stoich import formula_to_counter, znums_to_counter, delta

JUDGE_DIR = REPO / "outputs/260714_121340_dmx2_forward_ft"
JUDGE_CKPT = JUDGE_DIR / "epoch=89-step=1440.ckpt"
RELAXED = REPO / "eval/results/dmx2_relax_screen/structures_relaxed"
EGRID = np.linspace(-10, 10, 400)
NEAREF = (EGRID >= -2) & (EGRID <= 2)


@torch.no_grad()
def judge_cifs(cif_dir, tgt, model, scaler):
    from ase.io import read as ase_read
    rows = []
    cifs = sorted(Path(cif_dir).glob("*.cif"))
    for i in range(0, len(cifs), 32):
        chunk = cifs[i:i + 32]
        dl = []
        for c in chunk:
            a = ase_read(str(c))
            cp = a.cell.cellpar()
            dl.append(Data(
                frac_coords=torch.tensor(a.get_scaled_positions(), dtype=torch.float),
                atom_types=torch.tensor(a.get_atomic_numbers(), dtype=torch.long),
                lengths=torch.tensor(cp[:3], dtype=torch.float).view(1, -1),
                angles=torch.tensor(cp[3:], dtype=torch.float).view(1, -1),
                num_atoms=len(a), num_nodes=len(a)))
        b = Batch.from_data_list(dl).to("cuda")
        pred, _ = model.infer(b)
        pred = scaler.inverse_transform(pred.cpu()).numpy()
        off = 0
        for c, d in zip(chunk, dl):
            sid, k = c.stem.rsplit("__k", 1)
            n = int(d.num_atoms)
            p = pred[off:off + n]
            off += n
            t = tgt.get(sid)
            if t is None or len(t) != n:
                continue
            rows.append(dict(structure_id=sid, k=int(k),
                             post_mae_full=float(np.abs(p - t).mean()),
                             post_mae_nearef=float(np.abs(p[:, NEAREF]
                                                          - t[:, NEAREF]).mean())))
    return pd.DataFrame(rows)


def arm_metrics(blob_path, relax_tag, pre_csv, recs, audit, tgt, model, scaler,
                align, keep=None):
    blob = torch.load(blob_path, weights_only=False)
    if keep is None:
        keep = set(blob["structure_ids"])
    pre = pd.read_csv(pre_csv)
    pre = pre[pre.structure_id.isin(keep)]
    post = judge_cifs(RELAXED / relax_tag, tgt, model, scaler)
    post = post[post.structure_id.isin(keep)]
    relax = pd.read_csv(REPO / f"eval/results/dmx2_relax_screen/relax_{relax_tag}.csv")
    relax = relax[relax.sid.str.rsplit("__k", n=1).str[0].isin(keep)]

    # D1 + geometry per candidate
    pris_formula = {r.host: formula_to_counter(r.formula)
                    for r in audit.reset_index().itertuples()
                    if r.defect == "Defect-Free"}
    gt_formula = audit.formula.to_dict()
    rows = []
    for k, cands in enumerate(blob["preds"]):
        for i, c in enumerate(cands):
            sid = c["structure_id"]
            if sid not in keep:
                continue
            host = sid.split("_")[0]
            d_gen = delta(znums_to_counter(c["atom_types"]), pris_formula[host])
            d_gt = delta(formula_to_counter(gt_formula[sid]), pris_formula[host])
            pris = recs[f"{host}_Defect-Free"]
            gt_rec = recs[sid]
            gt_sig = defect_signature(pris, *frac_coords(gt_rec)[:1],
                                      np.asarray(gt_rec["atomic_numbers"]),
                                      np.asarray(gt_rec["cell"]))
            from ase.geometry import cellpar_to_cell
            cell = cellpar_to_cell(np.concatenate([np.asarray(c["lengths"], float),
                                                   np.asarray(c["angles"], float)]))
            gen_sig = defect_signature(pris, np.asarray(c["frac_coords"]),
                                       np.asarray(c["atom_types"]), cell,
                                       align=align)
            if gen_sig["host_matched_frac"] < 0.95:
                geo = "host_broken_or_polymorph"
            elif sig_key(gen_sig) != sig_key(gt_sig):
                geo = "wrong_defect_signature"
            elif (addition_placement(gen_sig, pris)
                  != addition_placement(gt_sig, pris)):
                geo = "defect_wrong_placement"
            else:
                geo = "recovered"
            rows.append(dict(structure_id=sid, k=k, stoich_ok=(d_gen == d_gt),
                             geo_status=geo))
    dg = pd.DataFrame(rows)

    m = dict(n_targets=len(keep), n_cand=len(dg))
    m["pre_mae_meank"] = pre.mae_full.mean()
    m["pre_mae_bestofk"] = pre.groupby("structure_id").mae_full.min().mean()
    m["post_mae_meank"] = post.post_mae_full.mean()
    m["post_mae_bestofk"] = post.groupby("structure_id").post_mae_full.min().mean()
    m["post_nearef_bestofk"] = post.groupby("structure_id").post_mae_nearef.min().mean()
    ok = relax[relax.error.isna() | (relax.error == "")]
    m["relax_converged"] = ok.converged.mean()
    m["relax_rmsd_median"] = ok.rmsd.median()
    m["in_band_rate"] = (ok.converged & (ok.rmsd <= 0.3617)).mean()
    m["stoich_rate"] = dg.stoich_ok.mean()
    m["stoich_bestofk"] = dg.groupby("structure_id").stoich_ok.max().mean()
    m["geo_recovered_rate"] = (dg.geo_status == "recovered").mean()
    m["geo_recovered_bestofk"] = (dg.assign(r=dg.geo_status == "recovered")
                                    .groupby("structure_id").r.max().mean())
    return m, dg


def main():
    recs = load_split_records()
    audit = pd.read_csv(REPO / "data/dmx2_audit.csv").set_index("id")
    with open(REPO / "data/dmx2_dos/val.json") as f:
        tgt = {r["structure_id"]: np.asarray(r["y"]) for r in json.load(f)}

    cfg = OmegaConf.load(JUDGE_DIR / "hparams.yaml")
    model = CSPProperty(**cfg)
    sd = torch.load(JUDGE_CKPT, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(sd, strict=True)
    model.to("cuda").eval()
    scaler = torch.load(JUDGE_DIR / "prop_scaler.pt", map_location="cpu",
                        weights_only=False)

    tm, tdg = arm_metrics(REPO / "eval/preds/dmx2_val_template_k5_w1.pt",
                          "val_template",
                          REPO / "eval/results/dmx2_generator_eval/val_template_k5_w1.csv",
                          recs, audit, tgt, model, scaler, align=False)
    tmpl_blob = torch.load(REPO / "eval/preds/dmx2_val_template_k5_w1.pt",
                           weights_only=False)
    fm, fdg = arm_metrics(REPO / "eval/preds/dmx2_val_k5_w1.pt",
                          "val_free",
                          REPO / "eval/results/dmx2_generator_eval/sweep_w1.csv",
                          recs, audit, tgt, model, scaler, align=True,
                          keep=set(tmpl_blob["structure_ids"]))

    df = pd.DataFrame({"template": tm, "free": fm})
    df.to_csv(REPO / "eval/results/dmx2_generator_eval/template_vs_free.csv")
    print(df.round(4).to_string())
    tdg.to_csv(REPO / "eval/results/dmx2_generator_eval/template_arm_geo.csv",
               index=False)
    fdg.to_csv(REPO / "eval/results/dmx2_generator_eval/free_arm_geo.csv",
               index=False)
    print("\ntemplate geo status:", tdg.geo_status.value_counts().to_dict())
    print("free geo status:", fdg.geo_status.value_counts().to_dict())
    cls = {s: coarse_class(audit.loc[s, "defect"]) for s in tdg.structure_id.unique()}
    tdg["cls"] = tdg.structure_id.map(cls)
    fdg["cls"] = fdg.structure_id.map(cls)
    print("\nrecovered rate by class (template / free):")
    tr = tdg.groupby("cls").geo_status.apply(lambda s: (s == "recovered").mean())
    fr = fdg.groupby("cls").geo_status.apply(lambda s: (s == "recovered").mean())
    print(pd.DataFrame({"template": tr, "free": fr}).round(3).to_string())


if __name__ == "__main__":
    main()
