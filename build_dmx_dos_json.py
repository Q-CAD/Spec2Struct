"""Build the DMX total-DOS fine-tune JSON from vaspruns, reproducing Shuyi's
recipe (Spec2Struct/structure_id.ipynb) exactly:

  per-atom y = sum over ALL orbitals AND BOTH spins of vasp.pdos[i][orb][spin]
  x = tdos.energies - tdos.efermi
  interp1d(linear, bounds_error=False, fill_value=0) onto linspace(-10, 10, 400)
  positions/cell/atomic_numbers from ASE atoms of structures[-1] (Cartesian)
  structure_id = parent folder name

Two target layouts, selected with --spin_split:

  default        y = up + down, shape [N, 400]
  --spin_split   y = [total(400) || m(400)], shape [N, 800], m = up - down

Both halves use the same energy grid, Fermi reference and interpolation, so the
first 400 columns of a spin-split build are identical to a default build of the
same structures.

Output goes to --out_dir (default data/dmx2_dos, or data/dmx2_dos_spin with
--spin_split). The split is read from --split_file (default splits/dmx2_v1.json,
frozen; see make_dmx2_split.py) and exclusions from --exclude_file (default
data/dmx2_exclude.json). Never re-randomizes: the build fails loudly if the parsed
folder set does not match the split id set exactly.

  python build_dmx_dos_json.py --dmx_dir /path/to/DMX_DOS_new
  python build_dmx_dos_json.py --dmx_dir /path/to/DMX_DOS_new --spin_split
"""
import os
import json
import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy import interpolate

warnings.filterwarnings("ignore")

DOS_LENGTH, EMIN, EMAX = 400, -10, 10
DEFAULT_OUT_DIR = "data/dmx2_dos"            # repo-relative, as the configs resolve it
DEFAULT_OUT_DIR_SPIN = "data/dmx2_dos_spin"  # default for --spin_split builds
DEFAULT_SPLIT_FILE = "splits/dmx2_v1.json"
DEFAULT_EXCLUDE_FILE = "data/dmx2_exclude.json"


def build_one(folder, dmx_dir, spin_split=False):
    from pymatgen.io.vasp import Vasprun
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.electronic_structure.core import Spin

    xml = os.path.join(dmx_dir, folder, "vasprun.xml")
    try:
        vr = Vasprun(filename=xml, parse_dos=True,
                     parse_eigen=False, parse_projected_eigen=False)
        n = len(vr.pdos)
        nE = len(vr.tdos.energies)

        # per-atom up / down summed over all orbitals
        up = np.zeros((n, nE))
        dn = np.zeros((n, nE))
        for i in range(n):
            for orb in vr.pdos[i]:
                d = vr.pdos[i][orb]
                if Spin.up in d:
                    up[i, :] += d[Spin.up]
                if Spin.down in d:
                    dn[i, :] += d[Spin.down]
        total = up + dn          # = baseline recipe (sum over orbitals + both spins)

        xfit = vr.tdos.energies - vr.tdos.efermi
        xnew = np.linspace(EMIN, EMAX, DOS_LENGTH)

        def interp_rows(arr):
            out = np.zeros((n, DOS_LENGTH))
            for i in range(n):
                f = interpolate.interp1d(xfit, arr[i, :], kind="linear",
                                         bounds_error=False, fill_value=0)
                out[i, :] = f(xnew)
            return out

        if spin_split:
            # [total || m]: the same per-atom curves, with the spin asymmetry
            # m = up - down appended on the identical energy grid.
            y = np.concatenate([interp_rows(total), interp_rows(up - dn)], axis=1)
        else:
            y = interp_rows(total)                        # [N, 400]

        atoms = AseAtomsAdaptor.get_atoms(vr.structures[-1])
        rec = {
            "structure_id": folder,
            "positions": atoms.get_positions().tolist(),
            "cell": np.asarray(atoms.get_cell()).tolist(),
            "atomic_numbers": atoms.get_atomic_numbers().tolist(),
            "y": y.tolist(),
        }
        meta = {
            "structure_id": folder, "N": int(n), "ydim": int(y.shape[1]),
            "natoms": len(atoms),
            "converged_e": bool(vr.converged_electronic),
            "ymin": float(y.min()), "ymax": float(y.max()),
            "finite": bool(np.isfinite(y).all()),
            "emin_data": float(xfit.min()), "emax_data": float(xfit.max()),
        }
        return rec, meta, None
    except Exception as e:
        return None, None, f"{folder}: {repr(e)[:160]}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dmx_dir", required=True,
                    help="raw data root: one <host>_<defect>/vasprun.xml per structure")
    ap.add_argument("--spin_split", action="store_true",
                    help="build [total(400) || m(400)] targets (m = up - down) "
                         "instead of total-only 400-d targets")
    ap.add_argument("--out_dir", default=None,
                    help=f"where train/val/test.json + build_meta.json are written "
                         f"(default {DEFAULT_OUT_DIR}, or {DEFAULT_OUT_DIR_SPIN} "
                         f"with --spin_split)")
    ap.add_argument("--split_file", default=DEFAULT_SPLIT_FILE,
                    help="frozen split id lists (splits/dmx2_v1.json or dmx2_v1_1.json)")
    ap.add_argument("--exclude_file", default=DEFAULT_EXCLUDE_FILE,
                    help="JSON with an 'exclude_all' list of structure ids to skip")
    args = ap.parse_args()
    out_dir = args.out_dir or (DEFAULT_OUT_DIR_SPIN if args.spin_split else DEFAULT_OUT_DIR)
    width = 2 * DOS_LENGTH if args.spin_split else DOS_LENGTH
    layout = "[total || m]" if args.spin_split else "total"
    print(f"{args.dmx_dir} -> {out_dir} (split {args.split_file}) "
          f"| target {layout}, width {width}")

    with open(args.exclude_file) as f:
        excluded = set(json.load(f)["exclude_all"])
    folders = sorted(
        d for d in os.listdir(args.dmx_dir)
        if os.path.isfile(os.path.join(args.dmx_dir, d, "vasprun.xml"))
        and d not in excluded
    )
    print(f"folders with vasprun.xml: {len(folders)} "
          f"(after skipping {len(excluded)} excluded)")

    nproc = min(24, os.cpu_count() or 8)
    print(f"parsing with {nproc} workers ...")
    recs, metas, errs = [], [], []
    with ProcessPoolExecutor(max_workers=nproc) as ex:
        futs = {ex.submit(build_one, f, args.dmx_dir, args.spin_split): f
                for f in folders}
        for k, fut in enumerate(as_completed(futs)):
            rec, meta, err = fut.result()
            if err:
                errs.append(err)
            else:
                recs.append(rec)
                metas.append(meta)
            if (k + 1) % 50 == 0:
                print(f"  {k + 1}/{len(folders)} done")

    print(f"\nbuilt: {len(recs)} | failed: {len(errs)}")
    for e in errs:
        print("  ERR", e)
    bad = [m["structure_id"] for m in metas if not m["finite"]]
    print("non-finite y:", bad if bad else "none")
    ydims = sorted({m["ydim"] for m in metas})
    print("y dims:", ydims)
    unconv = sorted(m["structure_id"] for m in metas if not m["converged_e"])
    print(f"electronically unconverged: {len(unconv)}")

    # frozen split from --split_file — never re-randomize
    with open(args.split_file) as f:
        sp = json.load(f)
    splits = {name: set(sp[name]) for name in ("train", "val", "test")}
    split_ids = splits["train"] | splits["val"] | splits["test"]
    built_ids = {r["structure_id"] for r in recs}
    if split_ids != built_ids:
        missing = sorted(split_ids - built_ids)
        extra = sorted(built_ids - split_ids)
        raise SystemExit(
            f"split/build id mismatch ({args.split_file} vs parsed folders):\n"
            f"  in split but not built ({len(missing)}): {missing}\n"
            f"  built but not in split ({len(extra)}): {extra}"
        )

    os.makedirs(out_dir, exist_ok=True)
    by = {r["structure_id"]: r for r in recs}
    for name, idset in splits.items():
        rows = [by[i] for i in sorted(idset)]
        path = os.path.join(out_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(rows, f)
        atoms = sum(len(s["atomic_numbers"]) for s in rows)
        print(f"{name}: {len(rows)} structures, {atoms} atoms -> {path}")

    with open(os.path.join(out_dir, "build_meta.json"), "w") as f:
        json.dump(sorted(metas, key=lambda m: m["structure_id"]), f, indent=0)


if __name__ == "__main__":
    main()
