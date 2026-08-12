"""dmx2_v1 -> dmx2_v1_1: duplicate-structure leak repair, with DFT-energy
adjudication for the 6 twin pairs whose members converged to different
magnetic states.

Rule: for each identical-geometry cross-split or within-train pair, exclude one member.
  - mislabeled member out first (mos2-A_Doped-Na = added-Na not substitution;
    tise2_Anti-Anion = K adatom not antisite);
  - state-mismatch pairs: KEEP the lower final e_0_energy member (last SCF step,
    identical geometry+composition; both members verified converged_electronic) —
    "every kept structure carries its DFT ground-state magnetic solution";
    tie |dE| < 1 meV/atom -> default (train member out for cross-split, ws2-c member
    out within train);
  - identical-state pairs: default rule.

No id ever moves between splits; v1 is untouched. Writes splits/dmx2_v1_1.json
and data/dmx2_exclude_v1_1.json (the v1 exclusions plus the 12 leak exclusions).
data/dmx2_exclude.json is read but never modified, so a v1 build still works
after running this script. Pass the v1_1 exclusion file to the build explicitly:
  python build_dmx_dos_json.py --dmx_dir ... \
      --split_file splits/dmx2_v1_1.json --exclude_file data/dmx2_exclude_v1_1.json \
      --out_dir data/dmx2_dos_v1_1
Energies were read from vasprun.xml ionic_steps[-1].electronic_steps[-1].e_0_energy
and are recorded below verbatim (eV).
"""
import json

V1 = "splits/dmx2_v1.json"
V11 = "splits/dmx2_v1_1.json"
EXCL = "data/dmx2_exclude.json"          # read-only input (v1 exclusions)
EXCL_V11 = "data/dmx2_exclude_v1_1.json"  # written here; v1's file is never modified

# (excluded member, kept member, reason) — energies in the report addendum
LEAK_EXCLUSIONS = [
    ("mos2-A_Doped-Na", "mos2-A_Adatom-Na", "mislabeled (Na1Mo16S32 = added atom)"),
    ("tise2_Anti-Anion", "tise2_Adatom-K", "mislabeled (K1Ti16Se32 = K adatom)"),
    ("ws2-B_Doped-O", "ws2-A_Doped-O", "default: train member out (identical state)"),
    ("ws2-c_Anti-Anion", "ws2-B_Anti-Anion", "default: ws2-c out (identical state)"),
    ("ws2-c_O_doped", "ws2-B_O_doped", "default: ws2-c out (identical state)"),
    ("ws2-c_Vacancy-Metal", "ws2-B_Vacancy-Metal", "default: ws2-c out (identical state)"),
    ("ws2-B_Anti-Metal", "ws2-c_Anti-Metal",
     "energy: -445.34512 vs -445.43282 eV, dE=1.83 meV/at -> keep ws2-c (val, FM-like)"),
    ("ws2-B_V48_V44", "ws2-c_V48_V44",
     "energy: -427.50072 vs -427.58687 eV, dE=1.87 meV/at -> keep ws2-c (test, FM-like)"),
    ("ws2-c_Re_doped", "ws2-B_Re_doped",
     "tie (dE=0.42 meV/at < 1) -> default train member out; keeps test (nonmagnetic)"),
    ("ws2-c_Cr_doped", "ws2-B_Cr_doped",
     "tie (dE=0.92 meV/at < 1) -> default ws2-c out; agrees with energy (B lower)"),
    ("ws2-c_Mo_doped", "ws2-B_Mo_doped",
     "tie (dE=0.36 meV/at < 1) -> default ws2-c out (energy alone would keep c)"),
    ("ws2-c_Vacancy-Anion", "ws2-B_Vacancy-Anion",
     "tie (dE=0.46 meV/at < 1) -> default ws2-c out; agrees with energy (B lower)"),
]


def main():
    excluded = [t[0] for t in LEAK_EXCLUSIONS]
    assert len(set(excluded)) == 12

    v1 = json.load(open(V1))
    kept = {t[1] for t in LEAK_EXCLUSIONS}
    all_v1 = set(v1["train"]) | set(v1["val"]) | set(v1["test"])
    assert set(excluded) <= all_v1 and kept <= all_v1

    out = {
        "version": "dmx2_v1_1",
        "derived_from": "dmx2_v1 (seed/source unchanged; ids only removed, never moved)",
        "provenance": "duplicate-structure leak repair with DFT-energy "
                      "adjudication for state-mismatch twins",
        "seed": v1["seed"],
        "source_audit": v1["source_audit"],
        "excluded": sorted(set(v1["excluded"]) | set(excluded)),
        "leak_excluded": sorted(excluded),
    }
    for s in ("train", "val", "test"):
        out[s] = [i for i in v1[s] if i not in set(excluded)]
    out["sizes"] = {s: len(out[s]) for s in ("train", "val", "test")}
    print("v1 sizes:", v1["sizes"], "-> v1_1 sizes:", out["sizes"])
    for s in ("train", "val", "test"):
        removed = sorted(set(v1[s]) - set(out[s]))
        print(f"  removed from {s}: {removed if removed else 'none'}")
    with open(V11, "w") as f:
        json.dump(out, f, indent=1)

    # Derive the v1_1 exclusion artifact WITHOUT touching the v1 file, so a v1
    # build still reproduces after this script has been run.
    base = json.load(open(EXCL))
    ex = dict(base)
    ex["derived_from"] = EXCL
    ex["leak_exclusions"] = sorted(excluded)
    ex["leak_exclusion_reasons"] = {t[0]: f"kept {t[1]}; {t[2]}" for t in LEAK_EXCLUSIONS}
    ex["exclude_all"] = sorted(set(base["exclude_all"]) | set(excluded))
    ex["comment"] = base["comment"] + (
        " | +12 leak_exclusions (identical-geometry twin pairs, energy-adjudicated;"
        " see the table in make_dmx2_split_v1_1.py)")
    with open(EXCL_V11, "w") as f:
        json.dump(ex, f, indent=1)
    assert json.load(open(EXCL))["exclude_all"] == base["exclude_all"], \
        f"{EXCL} must not be modified by this script"
    print(f"wrote {V11}")
    print(f"wrote {EXCL_V11} (exclude_all {len(ex['exclude_all'])}; "
          f"{EXCL} left unchanged at {len(base['exclude_all'])})")


if __name__ == "__main__":
    main()
