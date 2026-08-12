"""Read-only audit of the raw DOS dataset. One row per structure folder.

Per structure:
  vasprun.xml -> natoms, formula, ISPIN, NEDOS, NSW, n_ionic_steps,
                 converged_electronic, converged_ionic (vasprun's own check),
                 efermi, energy window min/max(E - E_F) vs [-10, 10],
                 y stats using Shuyi's recipe (per-atom PDOS summed over
                 orbitals+spins, no interp) -> ymin/ymax/finite
  OUTCAR      -> present?, "reached required accuracy" line, per-atom magmoms
                 (tot) -> full lists to a side JSON for the magnetic arm

Writes: <out-dir>/dmx2_audit.csv, dmx2_magmoms.json, dmx2_errors.txt
Overlap/dedupe columns are merged in from md5 lists passed via --md5-new/--md5-old.

  python audit_dmx2.py --dmx_dir /path/to/DMX_DOS_new \
      --md5-new new.md5 --md5-old old.md5
"""
import os
import csv
import json
import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

warnings.filterwarnings("ignore")

EMIN_T, EMAX_T = -10.0, 10.0


def audit_one(folder, dmx_dir):
    from pymatgen.io.vasp import Vasprun, Outcar
    from pymatgen.electronic_structure.core import Spin

    row = {"id": folder, "host": folder.split("_")[0],
           "defect": folder.split("_", 1)[1] if "_" in folder else ""}
    err = None
    xml = os.path.join(dmx_dir, folder, "vasprun.xml")
    try:
        vr = Vasprun(filename=xml, parse_dos=True,
                     parse_eigen=False, parse_projected_eigen=False)
        st = vr.final_structure
        row.update(
            natoms=len(st),
            formula=st.composition.formula.replace(" ", ""),
            ispin=int(vr.parameters.get("ISPIN", -1)),
            nedos=int(vr.parameters.get("NEDOS", -1)),
            nsw=int(vr.parameters.get("NSW", -1)),
            n_ionic_steps=len(vr.structures),
            converged_electronic=bool(vr.converged_electronic),
            converged_ionic=bool(vr.converged_ionic),
            efermi=float(vr.efermi),
        )
        e = vr.tdos.energies - vr.tdos.efermi
        row.update(emin_data=float(e.min()), emax_data=float(e.max()),
                   window_covers=bool(e.min() <= EMIN_T and e.max() >= EMAX_T))

        # y per Shuyi's recipe (sum orbitals + both spins), pre-interp
        n = len(vr.pdos)
        nE = len(vr.tdos.energies)
        y = np.zeros((n, nE))
        for i in range(n):
            for orb in vr.pdos[i]:
                d = vr.pdos[i][orb]
                if Spin.up in d:
                    y[i, :] += d[Spin.up]
                if Spin.down in d:
                    y[i, :] += d[Spin.down]
        row.update(n_pdos=int(n), ymin=float(y.min()), ymax=float(y.max()),
                   y_finite=bool(np.isfinite(y).all()))
    except Exception as ex:
        err = f"{folder}: vasprun: {repr(ex)[:200]}"

    out = os.path.join(dmx_dir, folder, "OUTCAR")
    magmoms = None
    row["has_outcar"] = os.path.isfile(out)
    if row["has_outcar"]:
        try:
            with open(out, "rb") as f:
                row["reached_accuracy"] = b"reached required accuracy" in f.read()
            oc = Outcar(out)
            if oc.magnetization:
                magmoms = [float(m["tot"]) for m in oc.magnetization]
                row["mag_abs_sum"] = float(np.abs(magmoms).sum()) if magmoms else 0.0
                row["mag_max_abs"] = float(np.abs(magmoms).max()) if magmoms else 0.0
        except Exception as ex:
            err = (err + " | " if err else f"{folder}: ") + f"OUTCAR: {repr(ex)[:200]}"
    return row, magmoms, err


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dmx_dir", required=True,
                    help="raw data root: one <host>_<defect>/ per structure")
    ap.add_argument("--md5-new", required=True)
    ap.add_argument("--md5-old", required=True)
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    def read_md5(path):
        d = {}
        with open(path) as f:
            for line in f:
                h, p = line.split()
                d[os.path.basename(os.path.dirname(p))] = h
        return d

    md5_new, md5_old = read_md5(args.md5_new), read_md5(args.md5_old)

    folders = sorted(d for d in os.listdir(args.dmx_dir)
                     if os.path.isdir(os.path.join(args.dmx_dir, d)))
    print(f"folders: {len(folders)}")

    nproc = min(8, os.cpu_count() or 4)
    rows, mags, errs = [], {}, []
    with ProcessPoolExecutor(max_workers=nproc) as ex:
        futs = {ex.submit(audit_one, f, args.dmx_dir): f for f in folders}
        for k, fut in enumerate(as_completed(futs)):
            row, mm, err = fut.result()
            rows.append(row)
            if mm is not None:
                mags[row["id"]] = mm
            if err:
                errs.append(err)
            if (k + 1) % 50 == 0:
                print(f"  {k + 1}/{len(folders)}")

    # overlap / dedupe
    from collections import Counter
    dup_groups = Counter(md5_new.values())
    for r in rows:
        h = md5_new.get(r["id"])
        r["md5"] = h
        r["in_old"] = r["id"] in md5_old
        r["identical_to_old"] = (h == md5_old[r["id"]]) if r["in_old"] else ""
        r["dup_within_new"] = dup_groups.get(h, 0) > 1 if h else ""

    rows.sort(key=lambda r: r["id"])
    os.makedirs(args.out_dir, exist_ok=True)
    cols = ["id", "host", "defect", "natoms", "formula", "ispin", "nedos", "nsw",
            "n_ionic_steps", "converged_electronic", "converged_ionic", "efermi",
            "emin_data", "emax_data", "window_covers", "n_pdos", "ymin", "ymax",
            "y_finite", "has_outcar", "reached_accuracy", "mag_abs_sum",
            "mag_max_abs", "in_old", "identical_to_old", "dup_within_new", "md5"]
    with open(os.path.join(args.out_dir, "dmx2_audit.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out_dir, "dmx2_magmoms.json"), "w") as f:
        json.dump(mags, f)
    with open(os.path.join(args.out_dir, "dmx2_errors.txt"), "w") as f:
        f.write("\n".join(errs))
    print(f"wrote {len(rows)} rows, {len(mags)} magmom sets, {len(errs)} errors")


if __name__ == "__main__":
    main()
