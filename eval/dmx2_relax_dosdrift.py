"""Judge DOS drift across MACE relaxation for the generated test set.

Runs the dmx2 forward judge on the RELAXED geometries (structures_relaxed/gen)
and merges with the pre-relax round-trip scores (test_k20_w1.csv). Reports
round-trip MAE before/after relaxation, mean-of-k and best-of-k, plus the
correlation of drift with relaxation RMSD.

Output: eval/results/dmx2_relax_screen/dosdrift.csv (+ printed aggregate).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data, Batch

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

JUDGE_DIR = REPO / "outputs/260714_121340_dmx2_forward_ft"
JUDGE_CKPT = JUDGE_DIR / "epoch=89-step=1440.ckpt"
RELAXED = REPO / "eval/results/dmx2_relax_screen/structures_relaxed/gen"
EGRID = np.linspace(-10, 10, 400)
NEAREF = (EGRID >= -2) & (EGRID <= 2)


@torch.no_grad()
def main():
    from ase.io import read as ase_read

    cfg = OmegaConf.load(JUDGE_DIR / "hparams.yaml")
    model = CSPProperty(**cfg)
    sd = torch.load(JUDGE_CKPT, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(sd, strict=True)
    model.to("cuda").eval()
    scaler = torch.load(JUDGE_DIR / "prop_scaler.pt", map_location="cpu",
                        weights_only=False)

    with open(REPO / "data/dmx2_dos/test.json") as f:
        tgt = {r["structure_id"]: np.asarray(r["y"]) for r in json.load(f)}

    pre = pd.read_csv(REPO / "eval/results/dmx2_generator_eval/test_k20_w1.csv")
    relax = pd.read_csv(REPO / "eval/results/dmx2_relax_screen/generated.csv")

    cifs = sorted(RELAXED.glob("*.cif"))
    if not cifs:
        raise SystemExit(
            f"no relaxed structures in {RELAXED}\n"
            "Run  python eval/dmx2_relax_screen.py --set gen  first.")
    print(f"{len(cifs)} relaxed CIFs")
    rows, batch_items = [], []

    def flush():
        if not batch_items:
            return
        b = Batch.from_data_list([d for _, _, d in batch_items]).to("cuda")
        pred, _ = model.infer(b)
        pred = scaler.inverse_transform(pred.cpu()).numpy()
        off = 0
        for sid, k, d in batch_items:
            n = int(d.num_atoms)
            p = pred[off:off + n]
            off += n
            t = tgt[sid]
            if len(t) != n:
                rows.append(dict(structure_id=sid, k=k, post_mae_full=np.nan,
                                 post_mae_nearef=np.nan))
                continue
            rows.append(dict(structure_id=sid, k=k,
                             post_mae_full=float(np.abs(p - t).mean()),
                             post_mae_nearef=float(np.abs(p[:, NEAREF] - t[:, NEAREF]).mean())))
        batch_items.clear()

    for cif in cifs:
        sid, k = cif.stem.rsplit("__k", 1)
        a = ase_read(str(cif))
        cellpar = a.cell.cellpar()
        d = Data(structure_id=sid,
                 frac_coords=torch.tensor(a.get_scaled_positions(), dtype=torch.float),
                 atom_types=torch.tensor(a.get_atomic_numbers(), dtype=torch.long),
                 lengths=torch.tensor(cellpar[:3], dtype=torch.float).view(1, -1),
                 angles=torch.tensor(cellpar[3:], dtype=torch.float).view(1, -1),
                 num_atoms=len(a), num_nodes=len(a))
        batch_items.append((sid, int(k), d))
        if len(batch_items) == 32:
            flush()
    flush()

    post = pd.DataFrame(rows)
    df = pre.merge(post, on=["structure_id", "k"], how="left")
    relax["structure_id"] = relax.sid.str.rsplit("__k", n=1).str[0]
    relax["k"] = relax.sid.str.rsplit("__k", n=1).str[1].astype(int)
    df = df.merge(relax[["structure_id", "k", "converged", "steps", "rmsd",
                         "edrop_per_atom"]], on=["structure_id", "k"], how="left")
    out = REPO / "eval/results/dmx2_relax_screen/dosdrift.csv"
    df.to_csv(out, index=False)

    ok = df[df.post_mae_full.notna()]
    best_pre = ok.groupby("structure_id").mae_full.min()
    best_post = ok.groupby("structure_id").post_mae_full.min()
    print(f"scored {len(ok)}/{len(df)} candidates (relaxed geometry judged)")
    print(f"round-trip MAE full  : pre {ok.mae_full.mean():.4f} -> post "
          f"{ok.post_mae_full.mean():.4f} (mean-of-k) | pre {best_pre.mean():.4f} "
          f"-> post {best_post.mean():.4f} (best-of-k)")
    bn_pre = ok.groupby("structure_id").mae_nearef.min()
    bn_post = ok.groupby("structure_id").post_mae_nearef.min()
    print(f"round-trip MAE nearEF: pre {ok.mae_nearef.mean():.4f} -> post "
          f"{ok.post_mae_nearef.mean():.4f} (mean-of-k) | pre {bn_pre.mean():.4f} "
          f"-> post {bn_post.mean():.4f} (best-of-k)")
    print(f"drift vs relax RMSD corr: "
          f"{ok[['rmsd']].join((ok.post_mae_full - ok.mae_full).rename('d')).corr().iloc[0, 1]:.3f}")


if __name__ == "__main__":
    main()
