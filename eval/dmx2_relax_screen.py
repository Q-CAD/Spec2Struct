"""MACE pre-relaxation screen for generated structures.

Requires a local MACE-MPA-0 medium checkpoint: pass --mace_model or set
MACE_MODEL_PATH. Runs in a separate environment with mace-torch installed.

GEOMETRY SCREEN ONLY: the potential is the non-magnetic MACE-MPA-0 medium
foundation model — energies/forces ignore spin, so this certifies proximity to
a (non-magnetic) local minimum, NOT energetic or magnetic truth.

Protocol (identical for calibration + generated): positions only (cell fixed,
keeps the 2D vacuum), ASE FIRE, fmax 0.05 eV/A, <=200 steps.

  --set gt   : the 62 ground-truth test structures (data/dmx2_dos/test.json).
               Their RMSD/energy-drop/steps distribution = the reference band
               for "already-relaxed DFT structure under this potential".
  --set gen  : all 1240 conditional test candidates (eval/preds blob, w=1).

Per structure: converged, steps, E_initial, E_final (eV), energy drop per atom,
RMSD (A, min-image, positions only). Relaxed geometries -> CIFs under
eval/results/dmx2_relax_screen/structures_relaxed/<set>/.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "eval/results/dmx2_relax_screen"
DEFAULT_MACE_MODEL = os.environ.get("MACE_MODEL_PATH", "MACE-MPA-0-medium.model")
FMAX, MAX_STEPS = 0.05, 200


def gt_atoms_list():
    from ase import Atoms
    with open(REPO / "data/dmx2_dos/test.json") as f:
        recs = json.load(f)
    out = []
    for r in recs:
        a = Atoms(numbers=r["atomic_numbers"], positions=r["positions"],
                  cell=r["cell"], pbc=True)
        out.append((r["structure_id"], a))
    return out


def blob_atoms_list(blob_path):
    from ase import Atoms
    from ase.geometry import cellpar_to_cell
    blob = torch.load(blob_path, weights_only=False)
    out = []
    for k, cands in enumerate(blob["preds"]):
        for c in cands:
            cell = cellpar_to_cell(np.concatenate([np.asarray(c["lengths"], float),
                                                   np.asarray(c["angles"], float)]))
            a = Atoms(numbers=np.asarray(c["atom_types"]),
                      scaled_positions=np.asarray(c["frac_coords"]) % 1.0,
                      cell=cell, pbc=True)
            out.append((f"{c['structure_id']}__k{k}", a))
    return out


def min_image_rmsd(a0, a1):
    """RMSD of positions under fixed cell, min-image (fractional wrap)."""
    cell = a0.cell.array
    df = a1.get_scaled_positions(wrap=False) - a0.get_scaled_positions(wrap=False)
    df -= np.round(df)
    d = df @ cell
    return float(np.sqrt((d ** 2).sum(axis=1).mean()))


def main(args):
    from mace.calculators import MACECalculator
    from ase.optimize import FIRE

    tag = args.tag or args.set
    (OUT / "structures_relaxed" / tag).mkdir(parents=True, exist_ok=True)
    if args.set == "gt":
        items = gt_atoms_list()
    elif args.set == "blob":
        items = blob_atoms_list(args.blob)
    else:
        items = blob_atoms_list(REPO / "eval/preds/dmx2_test_k20_w1.pt")
    if args.limit:
        items = items[:args.limit]
    print(f"[relax:{tag}] {len(items)} structures | model={args.mace_model} "
          f"| fmax={FMAX} steps<={MAX_STEPS} | positions only, cell fixed", flush=True)

    calc = MACECalculator(model_paths=args.mace_model, device="cuda",
                          default_dtype="float64")

    rows = []
    t0 = time.time()
    for i, (sid, atoms) in enumerate(items):
        a0 = atoms.copy()
        atoms.calc = calc
        try:
            e0 = float(atoms.get_potential_energy())
            opt = FIRE(atoms, logfile=None)
            opt.run(fmax=FMAX, steps=MAX_STEPS)
            e1 = float(atoms.get_potential_energy())
            fmax_final = float(np.abs(atoms.get_forces()).max())
            rows.append(dict(
                sid=sid, natoms=len(atoms), converged=bool(fmax_final <= FMAX),
                steps=int(opt.nsteps), e_initial=e0, e_final=e1,
                edrop_per_atom=(e0 - e1) / len(atoms),
                rmsd=min_image_rmsd(a0, atoms), fmax_final=fmax_final, error=""))
            atoms.calc = None
            from ase.io import write as ase_write
            ase_write(str(OUT / "structures_relaxed" / tag / f"{sid}.cif"), atoms)
        except Exception as ex:
            rows.append(dict(sid=sid, natoms=len(a0), converged=False, steps=-1,
                             e_initial=np.nan, e_final=np.nan, edrop_per_atom=np.nan,
                             rmsd=np.nan, fmax_final=np.nan, error=repr(ex)[:120]))
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  {i + 1}/{len(items)} done ({el / (i + 1):.1f} s/structure, "
                  f"eta {(len(items) - i - 1) * el / (i + 1) / 60:.0f} min)", flush=True)

    df = pd.DataFrame(rows)
    out_csv = OUT / {"gt": "calibration.csv", "gen": "generated.csv",
                     "blob": f"relax_{tag}.csv"}[args.set]
    df.to_csv(out_csv, index=False)
    ok = df[df.error == ""]
    print(f"[relax:{tag}] wrote {out_csv}")
    print(f"  converged: {ok.converged.mean():.3f} | steps median {ok.steps.median():.0f} "
          f"p90 {ok.steps.quantile(.9):.0f} | RMSD median {ok.rmsd.median():.4f} "
          f"p90 {ok.rmsd.quantile(.9):.4f} A | edrop/atom median "
          f"{ok.edrop_per_atom.median():.4f} p90 {ok.edrop_per_atom.quantile(.9):.4f} eV")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--set", choices=["gt", "gen", "blob"], required=True)
    p.add_argument("--blob", help="pred blob path for --set blob")
    p.add_argument("--tag", help="output subdir/csv tag for --set blob")
    p.add_argument("--mace_model", default=DEFAULT_MACE_MODEL,
                   help="path to the MACE-MPA-0 medium checkpoint "
                        "(default: $MACE_MODEL_PATH)")
    p.add_argument("--limit", type=int, default=0, help="debug: first N only")
    args = p.parse_args()
    main(args)
