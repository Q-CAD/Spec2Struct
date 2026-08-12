# Dataset audit summary

Read-only audit of the raw DOS dataset. Script: `audit_dmx2.py`; per-structure
table: `data/dmx2_audit.csv`; per-atom magnetic moments:
`data/dmx2_magmoms.json`; recomputed-pair listing:
`data/dmx2_changed_pairs.csv`; exclusions: `data/dmx2_exclude.json`.

## Inventory

- **624 folders**, all with `vasprun.xml`. **226 have no OUTCAR** (7 hosts have
  none at all: irte2, mnte2, pbte2, rete2, sns2, snse2, snte2; 40 of 59 hosts
  affected). Per-atom magnetic moments therefore exist for only **398/624**
  structures. This does not constrain the total-DOS pipeline, which reads only
  `vasprun.xml`.
- Identifier naming `host_defect`: `id.split("_")[0]` parses sensibly for
  **all 624** ids (hosts such as `mos2-A` keep their polymorph suffix; there
  are no underscore-free ids).
- All 624 parse with pymatgen without error, and `n_pdos == natoms` everywhere.

## Calculation character

All 624 runs are **static DOS calculations**: `NSW=0`, exactly one ionic step,
`ISPIN=2`, `NEDOS=3000`. No OUTCAR contains "reached required accuracy"
(0 of 398), because ionic convergence does not apply, and `vr.structures[-1]`
is therefore just the input structure. OUTCAR convergence checking is not part
of the build.

## Convergence and validity

- **4 electronically unconverged** runs, where the SCF hit `NELM` exactly
  (60/60 or 40/40), all in the cose2 host: `cose2_Fe_adatom`,
  `cose2_Fe_doped`, `cose2_Ni_doped`, `cose2_Vacancy-Anion-09B`. These form the
  exclusion list.
- Non-finite `y` values: **none**. `ymin ≥ 0` everywhere.
- Atom counts 44–49, so no conflict with the `max_atoms: 100` model setting.

## Energy window against the [-10, 10] eV target

- Lower edge: always covered — the dataset-wide maximum of `min(E−E_F)` is
  −12.5 eV.
- Upper edge: **503 of 624 (81%) end below +10 eV**. Quantiles of `max(E−E_F)`:
  min +5.51, p10 +7.04, median +8.65, p90 +10.85. The build's zero-padding
  therefore fabricates zeros across roughly the top 1–4.5 eV of the window for
  most structures. The window is kept at [-10, 10] for checkpoint
  compatibility, and evaluation is weighted toward the near-E_F window; the
  [-2, 2] eV defect-state metric is unaffected, being always covered.

## Duplicate and overlap checks

- **456 of 525** structures shared with an earlier version of this dataset are
  byte-identical. The other **69 have identical geometry but were recomputed**:
  all 69 are electronically converged here, against 31 of them unconverged
  previously (worst case `cro2_Int-Anion`, ΔE = 1450 eV). Energies and DOS for
  those 69 differ, so metrics computed on the earlier data are not comparable
  to metrics computed here.
- Exact duplicates **within** this dataset: none by md5.

Note that md5 equality does not catch structures that are physically identical
but were computed separately. A later check found that the hosts `ws2-B` and
`ws2-c` are the same host under two labels; see the split discussion in the
top-level README and `make_dmx2_split_v1_1.py`.

## Marginals relevant to the split design

- **59 hosts**, with 2–18 structures each. Distribution
  (n_structures → n_hosts): 2→1, 3→12, 4→2, 5→2, 6→3, 7→2, 8→3, 9→3, 10→1,
  11→6, 12→1, 14→1, 15→1, 16→1, 17→8, 18→12. Only `rete2`, with 2 structures,
  cannot satisfy the "every host with ≥3 structures appears in all three
  splits" rule.
- **48 defect types**. The most common are Vacancy-Anion-03, Vacancy-Metal and
  Defect-Free (58 each), then Anti-Metal and Anti-Anion (33 each), then
  Vacancy-Anion-06A/B (28 each). There is a long tail of 8 types with ≤2
  samples, including the labels `V48_V44` (2), `V48_V38` (1) and an index-free
  `Vacancy-Anion` (2), whose exact meaning is not documented in the source
  data.

## Exclusion list

`data/dmx2_exclude.json` holds the 4 unconverged cose2 runs. Nothing else
qualifies: no non-finite `y`, no duplicates. Resulting dataset size:
**620 structures**.
