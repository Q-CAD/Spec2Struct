"""D2/D3: geometric defect verification against pristine host references.

Mapping: pristine host supercell sites <-> structure atoms by nearest-neighbor
matching in fractional space (min-image, distances measured with the pristine
cell). Defect signature of a structure = {vacancies (unmatched pristine sites),
additions (unmatched atoms), substitutions (matched pairs with different
species)}. MATCH_TOL = 1.2 A absorbs DFT/MACE relaxation displacements; host
integrity = fraction of pristine sites matched and their mean displacement.

Modes:
  gt_locality  (D3): map every GT test structure vs its pristine reference,
      split atoms into defect-adjacent (< 4 A of a defect center) vs host-bulk,
      report judge per-atom MAE for the two groups (full + near-E_F windows).
  gen_verify   (D2): for given relaxed generated CIFs (best-of-k subset), map
      vs pristine and vs the GT defect signature: right defect type/species/
      sublattice, defect localized, host lattice intact. Reports failure modes.
"""
import argparse
import re
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RELAXED_GEN = REPO / "eval/results/dmx2_relax_screen/structures_relaxed/gen"
sys.path.insert(0, str(REPO))
from make_dmx2_split import coarse_class

MATCH_TOL = 1.2   # A: pristine site <-> atom pairing tolerance
ADJ_R = 4.0       # A: defect-adjacent shell for D3
LATT_TOL = 0.08   # relative in-plane lattice mismatch that counts as "broken cell"


def load_split_records():
    recs = {}
    for split in ("train", "val", "test"):
        with open(REPO / f"data/dmx2_dos/{split}.json") as f:
            for r in json.load(f):
                recs[r["structure_id"]] = r
    return recs


def frac_coords(rec):
    cell = np.asarray(rec["cell"])
    return np.asarray(rec["positions"]) @ np.linalg.inv(cell), cell


def match_sites(pris_frac, pris_z, cell, at_frac, at_z):
    """Greedy nearest matching pristine sites <-> atoms (min-image, pristine cell).
    Returns (site->atom map or -1, atom->site map or -1, displacements)."""
    dvec = pris_frac[:, None, :] - at_frac[None, :, :]
    dvec -= np.round(dvec)
    dist = np.linalg.norm(dvec @ cell, axis=2)          # [n_sites, n_atoms]
    site_to_atom = -np.ones(len(pris_frac), dtype=int)
    atom_to_site = -np.ones(len(at_frac), dtype=int)
    order = np.dstack(np.unravel_index(np.argsort(dist, axis=None), dist.shape))[0]
    for si, ai in order:
        if dist[si, ai] > MATCH_TOL:
            break
        if site_to_atom[si] == -1 and atom_to_site[ai] == -1:
            site_to_atom[si] = ai
            atom_to_site[ai] = si
    disp = np.array([dist[si, site_to_atom[si]]
                     for si in range(len(pris_frac)) if site_to_atom[si] != -1])
    return site_to_atom, atom_to_site, disp


# in-plane lattice ops for alignment (generated structures carry an arbitrary
# origin and possibly a lattice-symmetry setting relative to the reference)
_ALIGN_OPS = [np.diag(d) for d in ([1, 1, 1], [-1, -1, 1], [1, -1, 1], [-1, 1, 1])]
_ALIGN_OPS += [np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]) @ op for op in _ALIGN_OPS[:4]]


def align_to_reference(pris_frac, pris_z, pris_cell, at_frac, at_z):
    """Search op x translation maximizing site matches; returns aligned at_frac."""
    best = (-1.0, at_frac)
    anchors = [i for i in range(len(at_z)) if (pris_z == at_z[i]).any()][:2]
    for op in _ALIGN_OPS:
        af_op = at_frac @ op.T
        for ai in anchors:
            for si in np.where(pris_z == at_z[ai])[0]:
                cand = (af_op + (pris_frac[si] - af_op[ai])) % 1.0
                s2a, _, _ = match_sites(pris_frac, pris_z, pris_cell, cand, at_z)
                frac = (s2a != -1).mean()
                if frac > best[0]:
                    best = (frac, cand)
                    if frac == 1.0:
                        return cand
    return best[1]


def defect_signature(pris_rec, at_frac, at_z, at_cell, align=False):
    """Map atoms onto the pristine reference; return signature + host integrity."""
    pris_frac, pris_cell = frac_coords(pris_rec)
    pris_z = np.asarray(pris_rec["atomic_numbers"])
    # in-plane lattice mismatch (a, b rows)
    pl = np.linalg.norm(pris_cell[:2], axis=1)
    al = np.linalg.norm(np.asarray(at_cell)[:2], axis=1)
    latt_mismatch = float(np.abs(al - pl).max() / pl.max())

    if align:
        at_frac = align_to_reference(pris_frac, pris_z, pris_cell,
                                     np.asarray(at_frac), np.asarray(at_z))
    s2a, a2s, disp = match_sites(pris_frac, pris_z, pris_cell, at_frac, at_z)
    vac = [(int(pris_z[si]), pris_frac[si]) for si in np.where(s2a == -1)[0]]
    add = [(int(at_z[ai]), at_frac[ai]) for ai in np.where(a2s == -1)[0]]
    sub = [(int(pris_z[si]), int(at_z[s2a[si]]), pris_frac[si])
           for si in range(len(pris_z))
           if s2a[si] != -1 and pris_z[si] != at_z[s2a[si]]]
    centers = [c for _, c in vac] + [c for _, c in add] + [c for _, _, c in sub]
    return dict(
        vac=Counter(z for z, _ in vac), add=Counter(z for z, _ in add),
        sub=Counter((a, b) for a, b, _ in sub),
        n_vac=len(vac), n_add=len(add), n_sub=len(sub),
        centers_frac=np.array(centers) if centers else np.zeros((0, 3)),
        host_matched_frac=float((s2a != -1).mean()),
        host_mean_disp=float(disp.mean()) if len(disp) else np.nan,
        latt_mismatch=latt_mismatch, pris_cell=pris_cell)


def sig_key(sig):
    return (tuple(sorted(sig["vac"].items())), tuple(sorted(sig["add"].items())),
            tuple(sorted(sig["sub"].items())))


def addition_placement(sig, pris_rec):
    """Sorted (species, above_layer?) labels for each ADDED atom in a signature.
    Layer z-band = pristine cart-z range + 1.0 A margin."""
    if sig["n_add"] == 0:
        return []
    pris_frac, pris_cell = frac_coords(pris_rec)
    zlo = (pris_frac @ pris_cell)[:, 2].min() - 1.0
    zhi = (pris_frac @ pris_cell)[:, 2].max() + 1.0
    out = []
    # additions occupy the middle block of centers_frac (vac | add | sub ordering)
    n_vac = sig["n_vac"]
    adds = sig["centers_frac"][n_vac:n_vac + sig["n_add"]]
    for f in adds:
        z = float((f @ pris_cell)[2])
        out.append(bool(z < zlo or z > zhi))
    return sorted(out)


def adjacency_mask(sig, at_frac):
    """True for atoms within ADJ_R (A) of any defect center (min-image)."""
    if len(sig["centers_frac"]) == 0:
        return np.zeros(len(at_frac), dtype=bool)
    d = at_frac[:, None, :] - sig["centers_frac"][None, :, :]
    d -= np.round(d)
    return (np.linalg.norm(d @ sig["pris_cell"], axis=2) < ADJ_R).any(axis=1)


def mode_gt_locality():
    """D3: judge per-atom MAE, defect-adjacent vs host-bulk atoms, GT test set."""
    sys.path.insert(0, str(REPO / "eval"))
    from dmx2_forward_eval import predict_split, EGRID
    NEAREF = (EGRID >= -2) & (EGRID <= 2)

    recs = load_split_records()
    audit = pd.read_csv(REPO / "data/dmx2_audit.csv").set_index("id")
    preds = predict_split(Path("outputs/260714_121340_dmx2_forward_ft"),
                          "outputs/260714_121340_dmx2_forward_ft/epoch=89-step=1440.ckpt")

    with open(REPO / "data/dmx2_dos/test.json") as f:
        test = json.load(f)

    rows = []
    for r in test:
        sid = r["structure_id"]
        host = sid.split("_")[0]
        pris_id = f"{host}_Defect-Free"
        if pris_id not in recs or audit.loc[sid, "defect"] == "Defect-Free":
            continue
        at_frac, cell = frac_coords(r)
        sig = defect_signature(recs[pris_id], at_frac,
                               np.asarray(r["atomic_numbers"]), cell)
        adj = adjacency_mask(sig, at_frac)
        if adj.sum() == 0 or (~adj).sum() == 0:
            continue
        err = np.abs(preds[sid] - np.asarray(r["y"]))
        rows.append(dict(
            structure_id=sid, coarse_class=coarse_class(audit.loc[sid, "defect"]),
            n_adj=int(adj.sum()), n_bulk=int((~adj).sum()),
            mae_adj_full=float(err[adj].mean()), mae_bulk_full=float(err[~adj].mean()),
            mae_adj_nearef=float(err[adj][:, NEAREF].mean()),
            mae_bulk_nearef=float(err[~adj][:, NEAREF].mean())))
    df = pd.DataFrame(rows)
    out = REPO / "eval/results/dmx2_relax_screen/judge_locality.csv"
    df.to_csv(out, index=False)
    print(f"D3 judge locality, {len(df)} defective GT test structures "
          f"(adjacent = within {ADJ_R} A of defect):")
    print(df[["mae_adj_full", "mae_bulk_full", "mae_adj_nearef",
              "mae_bulk_nearef"]].mean().round(4).to_string())
    print("\nby class:")
    print(df.groupby("coarse_class")[["mae_adj_full", "mae_bulk_full",
        "mae_adj_nearef", "mae_bulk_nearef"]].mean().round(4).to_string())


def mode_gen_verify(subset_csv):
    """D2: verify relaxed best-of-k generated structures vs GT defect signature."""
    from ase.io import read as ase_read
    recs = load_split_records()
    audit = pd.read_csv(REPO / "data/dmx2_audit.csv").set_index("id")
    subset = pd.read_csv(subset_csv)   # columns: sid (target), k
    rows = []
    for _, s in subset.iterrows():
        sid, k = s.sid, int(s.k)
        host = sid.split("_")[0]
        pris_id = f"{host}_Defect-Free"
        cif = RELAXED_GEN / f"{sid}__k{k}.cif"
        if pris_id not in recs or not cif.exists():
            rows.append(dict(sid=sid, k=k, status="no_reference_or_cif"))
            continue
        a = ase_read(str(cif))
        gt = recs[sid]
        gt_sig = defect_signature(recs[pris_id], *frac_coords(gt)[:1],
                                  np.asarray(gt["atomic_numbers"]),
                                  np.asarray(gt["cell"]))
        gen_sig = defect_signature(recs[pris_id], a.get_scaled_positions(),
                                   a.get_atomic_numbers(), a.cell.array, align=True)
        # failure-mode decision tree (first failing condition labels the row).
        # NOTE on "right site": absolute positions are only defined up to the
        # host supercell's translational symmetry, so site correctness =
        # signature equality (removed/substituted species encode the sublattice)
        # + layer placement of any ADDED atoms (above layer = adatom vs
        # in-layer = interstitial), compared against the GT placement.
        if gen_sig["latt_mismatch"] > LATT_TOL:
            status = "cell_mismatch"
        elif gen_sig["host_matched_frac"] < 0.95:
            # distinguish "right chemistry, wrong stacking": does a sibling
            # polymorph reference (same base host, different -A/-B suffix)
            # match this structure's host sites instead?
            status = "host_lattice_broken"
            base = re.sub(r"-[A-Za-z]+$", "", host)
            for ref_id in recs:
                if not ref_id.endswith("_Defect-Free"):
                    continue
                rh = ref_id.rsplit("_", 1)[0]
                if rh == host or re.sub(r"-[A-Za-z]+$", "", rh) != base:
                    continue
                sib = defect_signature(recs[ref_id], a.get_scaled_positions(),
                                       a.get_atomic_numbers(), a.cell.array,
                                       align=True)
                if sib["host_matched_frac"] >= 0.95:
                    status = "wrong_polymorph"
                    break
        elif sig_key(gen_sig) != sig_key(gt_sig):
            status = "wrong_defect_signature"
        elif (addition_placement(gen_sig, recs[pris_id])
              != addition_placement(gt_sig, recs[pris_id])):
            status = "defect_wrong_placement"
        else:
            status = "recovered"
        rows.append(dict(sid=sid, k=k, status=status,
                         coarse_class=coarse_class(audit.loc[sid, "defect"]),
                         latt_mismatch=gen_sig["latt_mismatch"],
                         host_matched_frac=gen_sig["host_matched_frac"],
                         host_mean_disp=gen_sig["host_mean_disp"],
                         n_vac=gen_sig["n_vac"], n_add=gen_sig["n_add"],
                         n_sub=gen_sig["n_sub"]))
    df = pd.DataFrame(rows)
    if df.empty or (df.status != "no_reference_or_cif").sum() == 0:
        raise SystemExit(
            "no relaxed generated structures matched the subset CSV.\n"
            f"  looked for <sid>__k<k>.cif in {RELAXED_GEN}\n"
            f"  for the {len(subset)} (sid, k) pairs in {subset_csv}\n"
            "Run  python eval/dmx2_relax_screen.py --set gen  first, and make sure\n"
            "the subset CSV refers to candidates present in that relaxation run.")
    out = REPO / "eval/results/dmx2_relax_screen/defect_geometry.csv"
    df.to_csv(out, index=False)
    ok = df[df.status != "no_reference_or_cif"]
    print(f"D2 geometric verification on {len(ok)} relaxed best-of-k structures:")
    print(ok.status.value_counts().to_string())
    print(f"\nrecovery rate: {(ok.status == 'recovered').mean():.4f}")
    print("\nby class:")
    print(ok.groupby("coarse_class").status
            .apply(lambda s: (s == "recovered").mean()).round(4).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["gt_locality", "gen_verify"], required=True)
    p.add_argument("--subset_csv", help="gen_verify: CSV with sid,k of best-of-k picks")
    args = p.parse_args()
    if args.mode == "gt_locality":
        mode_gt_locality()
    else:
        mode_gen_verify(args.subset_csv)
