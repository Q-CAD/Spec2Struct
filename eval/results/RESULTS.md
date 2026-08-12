# Results

Total-DOS ↔ structure on the 2D-defect dataset. All MAEs are in states/eV/atom
on the 400-point grid `linspace(-10, 10, 400)` in `E − E_F`. Three energy
windows are reported throughout:

- **full** `[-10, 10]` eV
- **all-valid** `[-10, +5.5]` eV — covered by real data in every structure
- **near-E_F** `[-2, 2]` eV — always real data

Two levels: **per-atom** (mean over atoms × window bins) and **cell**
(atom-summed DOS divided by the atom count; needs no atom correspondence).

## Setup

- 624 static HSE06 DOS calculations (`NSW=0`, `ISPIN=2`, `NEDOS=3000`; 44–49
  atoms per cell: pristine 2D hosts plus point defects). Four runs saturated
  `NELM` and are excluded, leaving **620**.
- Per-atom total DOS, summed over orbitals and both spins. Frozen
  host-stratified split **496/62/62** (`splits/dmx2_v1.json`), used unchanged
  for every number on this page.
- Caveat carried by every table: 81% of the source runs end below +10 eV, so
  the top of the window is zero-padded for most structures. Near-E_F is always
  covered by real data.

| model | checkpoint | val loss |
|---|---|---|
| forward (structure → DOS), also the judge | `outputs/260714_121340_dmx2_forward_ft/epoch=89-step=1440.ckpt` | 0.2370 |
| generator (DOS → structure, CFG) | `outputs/260714_125010_dmx2_cfg_ft/epoch=789-step=12640.ckpt` | 0.6042 |

The forward model doubles as the judge that scores generated structures, so
every round-trip number below inherits its blind spots: they measure agreement
with the forward model, not with DFT.

---

## 1. Forward model — structure → DOS

62 test structures. Script: `eval/dmx2_forward_eval.py`; data:
`dmx2_forward_eval/metrics.csv`.

| metric | model | global train-mean baseline | per-element train-mean baseline |
|---|---|---|---|
| per-atom full | **0.1311** | 0.2355 | 0.1681 |
| per-atom all-valid | **0.1524** | 0.2781 | 0.1955 |
| per-atom near-E_F | **0.1676** | 0.2978 | 0.2211 |
| cell full | **0.1068** | 0.1700 | 0.1395 |
| cell all-valid | **0.1229** | 0.1963 | 0.1601 |
| cell near-E_F | **0.1344** | 0.2041 | 0.1811 |

The model beats the per-element baseline by ~22% (full) and ~24% (near-E_F).

By coarse defect class (per-atom, full window):

| class | n | model | per-element baseline |
|---|---|---|---|
| adatom | 7 | 0.1016 | 0.1812 |
| antisite | 5 | 0.1052 | 0.1650 |
| interstitial | 2 | 0.1536 | 0.1628 |
| other | 1 | 0.1215 | 0.1467 |
| pristine | 8 | 0.1688 | 0.1843 |
| substitution | 13 | 0.1039 | 0.1547 |
| vacancy | 26 | 0.1447 | 0.1680 |

## 2. Conditional generation — DOS → structure

Generation with `eval/generate_for_eval.py` at `diff_ratio=1.0` (pure noise,
so no ground-truth geometry leaks into the sample), atom count fixed to the
target's, per-atom 400-point DOS conditioning. Scored with
`eval/dmx2_roundtrip.py`. Data: `dmx2_generator_eval/`.

### Guidance-scale sweep (val, k=5 per target)

| w | MAE mean-of-k | MAE best-of-5 | struct_valid | composition exact match |
|---|---|---|---|---|
| 0 | 0.1496 | 0.1360 | 0.76 | 0.66 |
| **1** | **0.1488** | **0.1357** | 0.77 | **0.68** |
| 2 | 0.1519 | 0.1373 | 0.82 | 0.68 |
| 3 | 0.1615 | 0.1424 | 0.84 | 0.60 |
| 5 | 0.1896 | 0.1483 | 0.85 | 0.46 |

**w = 1 is chosen**: best round-trip MAE at both mean-of-k and best-of-k, and
best composition match. Higher guidance trades DOS fidelity (+27% MAE at w=5)
for modest structural-validity gains.

A note on validity metrics: smact composition validity is uninformative on
this dataset — the ground-truth val compositions themselves pass it only 23%
of the time, because defect supercells rarely charge-balance, and every
guidance scale sits at that floor. `struct_valid` (minimum pairwise distance
> 0.5 Å), which is ≥ 0.76 everywhere, is the meaningful structural check.

### Test at w = 1, k = 20

| | conditional w=1 | unconditional (CFG null branch) |
|---|---|---|
| round-trip MAE, mean-of-20 | **0.1576** | 0.2691 |
| round-trip MAE, best-of-20 | **0.1392** | 0.2271 |
| near-E_F, mean-of-20 | **0.2111** | 0.3214 |
| near-E_F, best-of-20 | **0.1842** | 0.2550 |
| struct_valid | 0.84 | 0.87 |
| composition exact match | **0.70** | 0.00 |
| element-set match | 0.73 | 0.002 |

DOS conditioning cuts round-trip MAE by **41%** (mean-of-k) and **39%**
(best-of-k) against the unconditional control at identical k, and recovers the
target composition exactly in 70% of samples versus 0% unconditionally.

The unconditional round-trip (0.269) lands at the global-train-mean forward
baseline (0.236 on the same judge scale) — unconditioned samples carry no
target-specific DOS information, as expected. For context on the other side,
the forward model on ground-truth test structures scores 0.131, so the
conditional round-trip (0.158) is within 20% of that judge floor and
best-of-20 (0.139) nearly reaches it.

## 3. Relaxation screen and defect verification

Potential: **MACE-MPA-0 medium**, run in a separate `mmace` environment
(mace 0.3.12). **Non-magnetic — this is a geometry screen, not energetics or
magnetism truth.** Protocol, identical for calibration and generated sets:
positions-only FIRE relaxation, cell fixed, `fmax` 0.05 eV/Å, ≤ 200 steps.
Script: `eval/dmx2_relax_screen.py`; data: `dmx2_relax_screen/`.

### 3.1 Calibration on 62 ground-truth test structures

100% converged | steps median 21, p95 44 | RMSD median 0.059 Å, **p95 0.362 Å**
| energy drop median 0.005 eV/atom, p95 0.097.

This band defines "an already-relaxed DFT structure under this potential". It
is nonzero because MACE-MPA-0 is not the functional the dataset was computed
with: its minima sit a few tenths of an Å from the dataset geometries.

### 3.2 Generated — 1240 conditional test candidates

1216 relaxed cleanly (24 raised errors) | 93.4% converged | steps median 68 |
RMSD median 0.380 Å (q75 0.729, q90 1.18) | energy drop median 0.30 eV/atom,
with the q90 tail blowing up on unphysical starts. **47% of candidates land
inside the ground-truth band** (converged and RMSD ≤ 0.362 Å); 25% are far
outside it (RMSD > 2× the ground-truth p95, or step-capped). **85.5% of test
targets have at least one in-band candidate.**

### 3.3 DOS drift through relaxation

Judge round-trip MAE over the 1216 relaxed candidates:

| | pre-relax | post-relax |
|---|---|---|
| full window, mean-of-k | 0.1529 | 0.1532 |
| full window, best-of-k | 0.1392 | **0.1296** |
| near-E_F, mean-of-k | 0.2073 | 0.2059 |
| near-E_F, best-of-k | 0.1842 | **0.1655** |

Best-of-k **improves** through relaxation and lands below the judge's own
error on ground-truth structures (0.1311): after the geometry screen, the
judge cannot distinguish the best generated structures from real ones. This is
the strongest realism evidence available here. Drift is weakly
anticorrelated with relaxation RMSD (r = −0.12).

### 3.4 Defect stoichiometry recovery (D1)

Delta of the generated composition against the host's pristine supercell
formula, compared to the ground-truth delta (exact species + count match,
stricter than plain composition matching). 1220/1240 candidates have a
pristine reference — cose2 has no defect-free entry.

**Per-candidate 0.698 | best-of-20 0.918.** By class: vacancy 0.81, other
0.90, antisite 0.75, pristine 0.68, interstitial 0.63, substitution 0.62,
adatom 0.43.

### 3.5 Geometric defect verification (D2)

61 relaxed best-of-k picks. Mapping requires an origin/lattice-operation
alignment search, because the diffusion output carries an arbitrary origin —
without alignment every structure looks broken.

After alignment: recovered 14/61 (**23%**) | wrong_defect_signature 22 |
host_lattice_broken 24 | cell_mismatch 1. Recovery by class: vacancy 0.32,
substitution 0.31, adatom 0.14, pristine 0.13, antisite/interstitial/other 0.

- "Recovered" means host ≥ 95% site-matched, exact defect signature (species
  and sublattice, via removed/substituted species), and correct above-layer or
  in-layer placement of added atoms. Site identity is only defined up to host
  translational symmetry.
- The wrong_defect_signature cases have **intact** hosts (match ≈ 1.0) but a
  different defect delta: the best-by-MAE candidate is often not the
  stoichiometry-correct one. Since D1 best-of-20 is 0.92, selecting by MAE and
  selecting by stoichiometry disagree — which motivates the combined selection
  rule in §4.
- The host_lattice_broken group (median site-match 0.70, lattice parameters
  fine) was checked against sibling polymorph references: **5 of the 9
  checkable cases match a sibling polymorph at 0.92–0.98** — right chemistry,
  wrong stacking. The true "lattice actually broken" rate is therefore well
  below 24/61.
- Alignment searches 8 in-plane lattice operations × anchor translations with
  a 1.2 Å site tolerance; registry shifts beyond that read as broken or
  wrong-polymorph.

### 3.6 Judge defect-locality (D3)

50 defective ground-truth test structures, judge per-atom MAE for
defect-adjacent atoms (within 4 Å of the defect) versus host-bulk atoms:
full window **0.136 vs 0.122** (+11%); near-E_F **0.181 vs 0.158** (+14%).

The judge resolves defect-local DOS at near-bulk accuracy, so round-trip MAE
is a meaningful — if slightly optimistic at defect sites — certificate for
defect physics. Weakest class is interstitial (near-E_F 0.312 adjacent vs
0.172 bulk).

## 4. Combined selection rule

Per-target pick = argmin post-relaxation round-trip MAE, subject to (i) an
in-band relaxation (converged, RMSD ≤ the ground-truth p95 of 0.362 Å) and
(ii) correct D1 defect stoichiometry. 45 of 62 targets satisfy both, 8 drop
(ii), 9 drop both; 8 picks change relative to selecting on MAE alone.

- D2 geometric recovery: **23.0% (MAE only) → 31.1%**, at an MAE cost of
  0.1296 → 0.1348.
- wrong_defect_signature falls 22 → 16. A `wrong_polymorph` D2 category
  (sibling-polymorph host match ≥ 0.95) accounts for 2 cases; sibling matches
  at 0.92–0.94 remain inside host_lattice_broken, so the true broken-lattice
  rate is lower still.

Residual failure modes are the wrong polymorph/stacking of the host, and
right-host-wrong-defect signatures.

## 5. Template mode — defect completion in a known host

`CSPDiffusion.template_cfg_sample` pins per-node coordinates and types at
every denoising step, fixes the lattice to the host cell, starts free atoms
from pure noise, and applies full per-atom CFG conditioning. Pinned atoms are
the pristine host sites matched to same-species ground-truth atoms; free atoms
are the defect-involved atom count. Scripts: `eval/dmx2_template_gen.py`,
`eval/dmx2_template_compare.py`, `eval/dmx2_template_reframe.py`.

**What this task actually is.** The pinned/free split is derived from the
ground-truth defect signature, so the model receives the host cell, the
pristine coordinates and species of all host-matched atoms, the *count* of
defect-involved atoms, and per-atom target DOS rows (including the defect
atoms' rows). It generates the species and positions of the free atoms only.
This is defect completion in a known host — **not** blind DOS → structure
generation. The free arm (§2) is the blind task.

**Vacancy targets are construction, not generation.** For a pure vacancy the
defect is an absence, so **26 of 30 vacancy targets have `n_free = 0`**: every
coordinate, every species and the lattice are fixed by the template and the
sampler generates nothing. Measured within-target round-trip MAE spread across
k=5 is ≤ 6.5e-9 (float jitter) for all 26, against a median of 1.3e-3 for
`n_free > 0` targets. The remaining 4 vacancy targets have 1–4 free atoms
(host atoms displaced or species-mismatched in the ground-truth map). Vacancy
rows therefore measure template construction plus MACE-relaxation stability,
and are excluded from generative claims.

Val experiment, 53 defective targets, k=5, w=1, under all three framings:

| metric (template / free) | all 53 (construction + generation) | non-vacancy (23) | n_free > 0 (27) |
|---|---|---|---|
| geometric recovery, per-candidate | 0.536 / 0.075 | **0.235** / 0.104 | **0.200** / 0.089 |
| geometric recovery, best-of-5 | 0.623 / 0.245 | **0.435** / 0.304 | **0.370** / 0.259 |
| post-relax MAE, best-of-5 | 0.1333 / 0.1350 | 0.1291 / 0.1291 | 0.1305 / 0.1317 |
| in-band relax rate | 0.80 / 0.46 | **0.69** / 0.46 | **0.65** / 0.47 |
| stoichiometry, per-candidate | 0.85 / 0.65 | 0.65 / 0.62 | 0.70 / 0.63 |

Supporting detail on the all-53 framing: relaxation converged 0.96 vs 0.89;
pre-relax MAE best-of-k is *worse* for the template arm (0.1586 vs 0.1341)
because pinned atoms sit at idealized pristine positions — one MACE relaxation
later it wins on every axis. Per-class recovery (per-candidate, template/free):
vacancy 0.77/0.05, adatom 0.40/0.08, antisite 0.27/0.07, substitution
0.18/0.16, interstitial 0/0.

**On targets where the model generates anything**, templating roughly doubles
per-candidate defect recovery (0.20–0.24 vs 0.09–0.10) and raises best-of-5
recovery ~1.4× (0.37–0.43 vs 0.26–0.30), with markedly more realistic
geometries (in-band 0.65–0.69 vs 0.46). Round-trip DOS match is at parity with
free generation, not better.

### More sampling on the weak classes (11 adatom/antisite targets)

| | template k=20 | template k=5 | free k=5 |
|---|---|---|---|
| geometric recovery, best-of-k | **0.636** | 0.545 | 0.182 |
| post-relax MAE, best-of-k | **0.1319** | 0.1378 | 0.1360 |
| post-relax near-E_F, best-of-k | **0.1726** | 0.1859 | 0.1825 |

Sampling monotonically helps the template arm while per-candidate rates stay
flat, as expected. These targets all have `n_free = 1`, so this table is
unaffected by the vacancy caveat above.

The dominant residual failure is the **type** of the generated free atom —
wrong_defect_signature accounts for 79 of 265 template-arm failures. A
type-prior or species-constrained variant is the clearest next lever.

## 6. Per-family benchmark (host anion O / S / Se / Te)

Script: `eval/dmx2_family_benchmark.py`; data: `dmx2_family_benchmark/`.
Bootstrap 95% CIs, 10k resamples, structure-level. With only 8 oxide test
structures, the CIs decide what is claimable.

### 6.1 Oxides are 6.5% of the data

| family | train | val | test | total | share | pristine/defective |
|---|---|---|---|---|---|---|
| O | 24 | 8 | 8 | 40 | **6.5%** | 8 / 32 |
| S | 169 | 19 | 19 | 207 | 33.4% | 17 / 190 |
| Se | 158 | 19 | 19 | 196 | 31.6% | 16 / 180 |
| Te | 145 | 16 | 16 | 177 | 28.6% | 17 / 160 |

### 6.2 Forward model: oxides ~1.8× worse

Per-atom, full window:

| family | model [95% CI] | global base | per-element base | model/per-element [95% CI] |
|---|---|---|---|---|
| O | **0.211** [0.154, 0.277] | 0.293 | 0.244 | **0.86** [0.66, 1.12] |
| S | 0.123 [0.110, 0.137] | 0.242 | 0.170 | 0.72 [0.63, 0.82] |
| Se | 0.124 [0.111, 0.138] | 0.226 | 0.157 | 0.79 [0.72, 0.86] |
| Te | 0.109 [0.098, 0.121] | 0.211 | 0.142 | 0.77 [0.70, 0.84] |

- "Oxides are harder in absolute terms" is **significant**: the oxide model
  CI [0.154, 0.277] is disjoint from every other family's.
- The baselines are also ~1.5× worse on oxides (0.244 vs 0.142–0.170
  per-element), so part of the gap is intrinsic target difficulty rather than
  model failure.
- The model/baseline ratio is worst on O (0.86 vs 0.72–0.79) — the model
  learns least on top of the baseline exactly where data is scarcest — but at
  n=8 the ratio CI [0.66, 1.12] **crosses 1.0**, so it cannot be claimed at
  95% that the model beats the per-element baseline on oxides at all.

### 6.3 Generator: same ordering

Test k=20, w=1:

| family | round-trip best-of-20 [CI] | uncond best-of-20 | uncond/cond | selected post-relax |
|---|---|---|---|---|
| O | **0.197** [0.171, 0.223] | 0.276 | **1.40** | **0.203** |
| S | 0.145 [0.126, 0.168] | 0.236 | 1.63 | 0.125 |
| Se | 0.134 [0.120, 0.149] | 0.225 | 1.68 | 0.133 |
| Te | 0.110 [0.100, 0.120] | 0.195 | 1.77 | 0.115 |

The oxide best-of-20 CI is disjoint from Se and Te, with marginal overlap with
S. The DOS-conditioning benefit (uncond/cond ratio) shrinks monotonically
toward O.

### 6.4 Pristine versus defective within family

| family | defective fwd/gen(bo20) | pristine fwd/gen(bo20) | n def/pris |
|---|---|---|---|
| O | 0.196 / 0.190 | 0.317 / 0.242 | 7 / 1 |
| S | 0.120 / 0.146 | 0.145 / 0.137 | 17 / 2 |
| Se | 0.117 / 0.127 | 0.181 / 0.191 | 17 / 2 |
| Te | 0.105 / 0.106 | 0.127 / 0.127 | 13 / 3 |

The single pristine oxide is mno2, the worst structure in the entire test set
(forward 0.317) and a long-standing outlier. Oxide *defective* structures
alone still sit at 0.196, about 1.7× the other families, so mno2 is not the
whole oxide story. Pristine cells number 1–3 per family, so this table is
directional only.

### 6.5 Defect recovery: oxide DOS is hard, oxide geometry is not

D1 per-candidate over k=20; D2 on the best picks:

| family | D1 stoich per-candidate | D1 best-of-20 | D2 recovered (n) |
|---|---|---|---|
| O | **0.58** | 0.88 | **0.50** (8) |
| S | 0.69 | 0.89 | 0.32 (19) |
| Se | 0.69 | 0.94 | 0.28 (18) |
| Te | 0.78 | 0.94 | 0.25 (16) |

Two-sided result: oxides are the *worst* at per-candidate stoichiometry but
the *best* at strict geometric defect recovery of the selected pick. The DOS
mismatch and the geometry channel dissociate, consistent with the difficulty
living in the DOS targets (sharp O-2p features) rather than in the structures.
Counts include each family's pristine targets, and the small-n caveat applies.

### 6.6 Why oxides are harder — training-set statistics

From the 496 training structures:

| family | n | pairwise cell-DOS MAE [CI] | mean dDOS/dE | per-atom DOS max | emax < +10 eV |
|---|---|---|---|---|---|
| O | 24 | **0.289** [0.279, 0.300] | **1.23** | **14.5** | 8% |
| S | 169 | 0.215 [0.214, 0.216] | 1.11 | 6.9 | 72% |
| Se | 158 | 0.208 [0.207, 0.209] | 1.02 | 6.7 | 89% |
| Te | 145 | 0.181 [0.180, 0.182] | 0.93 | 7.0 | 99% |

Oxide targets are (a) the most diverse family — pairwise MAE 0.29 against
0.18–0.22, CIs disjoint — while having the fewest training examples, (b) the
sharpest, with the steepest DOS gradients, and (c) carry 2× larger per-atom
DOS peaks from localized O-2p states, which an MAE metric penalizes heavily
when a sharp peak is misplaced.

Energy-window coverage acts as a **negative control**: oxides have the *best*
coverage (only 8% of runs end below +10 eV, against 72–99% elsewhere), so the
zero-padding artifact cannot explain the oxide gap.

In short, oxides combine the least data with the most diverse and sharpest
targets. Both "intrinsically harder" and "under-trained" are true; §7.2's
ratio row says the under-trained part is real but unproven at n=8.

## 7. Oxide ablation — oversampling and leave-one-out

Observational tables cannot separate "intrinsically harder" from
"under-represented"; the causal test is retraining. Two arms, run with the
same frozen split (`splits/dmx2_v1.json`, untouched), the same seed (2024),
the same warm starts and configs except for the training set, and the same
eval pipeline (original judge, original per-structure baselines, structure-level
bootstrap CIs, 10k resamples). Data: `dmx2_family_benchmark/ablation/`.

Datasets, with caches derived from the committed `train_ori.pt` so nothing is
re-preprocessed:

- `data/dmx2_dos_ox4`: train = 496 + 3× the 24 oxides = 568 (16.9% oxide);
  val and test byte-identical to `data/dmx2_dos`.
- `data/dmx2_dos_loo`: train = 472 (0% oxide); val and test identical.

Runs, 1000 epochs each, best-val checkpoint:

- forward ox4: `outputs/260721_162622_dmx2_forward_ox4`, best val 0.2395 @ ep 64
  (baseline model: 0.2370 @ ep 89)
- generator ox4: `outputs/260721_171048_dmx2_cfg_ox4`, best checkpoint ep 894
- forward LOO: `outputs/260721_174632_dmx2_forward_loo`, best val 0.2502 @ ep 74

### 7.1 Arm 1 (oxide oversampling) is a clean null

The success criterion was fixed before the runs: *oxide per-atom full MAE mean
drops below 0.154 (outside the current CI) while S/Se/Te stay inside their
current CIs; also check that the model/per-element ratio CI moves off 1.0.*

| family | baseline model [95% CI] | ox4 model [CI] | criterion check |
|---|---|---|---|
| O | 0.2109 [0.154, 0.277] | **0.2118** [0.161, 0.269] | needs < 0.154 → **fails, unchanged** |
| S | 0.1229 [0.110, 0.137] | 0.1235 | inside baseline CI ✓ |
| Se | 0.1240 [0.111, 0.138] | 0.1261 | inside baseline CI ✓ |
| Te | 0.1094 [0.098, 0.121] | 0.1099 | inside baseline CI ✓ |

Oxide MAE moved 0.2109 → 0.2118 (+0.4%): quadrupling the oxide sampling weight
produced no change at all, while leaving the other families untouched — a
clean null rather than a trade-off. The ratio check agrees: ox4 model/per-element
on O is 0.87, CI [0.69, 1.08], still crossing 1.0 and unchanged from the
baseline [0.66, 1.12]. The generator behaves the same way: oxide round-trip
best-of-20 **0.1963 against a baseline 0.1966**, with all families inside
baseline CIs.

### 7.2 Arm 2 (leave-oxide-out) shows the model does use oxide data

| family | baseline model | LOO model [CI] | vs per-element baseline |
|---|---|---|---|
| O | 0.2109 | **0.2522** [0.211, 0.300] | ratio 1.03 [0.88, 1.23] |
| S | 0.1229 | 0.1284 (inside CI) | 0.76 |
| Se | 0.1240 | 0.1250 (inside CI) | 0.80 |
| Te | 0.1094 | 0.1107 (inside CI) | 0.78 |

Removing all 24 oxide training structures degrades oxide MAE by 20%
(0.211 → 0.252) and erases the model's entire edge over the per-element
baseline on oxides (ratio → 1.03). Oxide skill is therefore genuinely learned
from the oxide data, not free-ridden from the chalcogenides.

### 7.3 Combined interpretation

The two arms bracket the mechanism cleanly. Leave-one-out shows the 24 oxide
structures carry real, used signal, since removing them hurts. Oversampling
shows that signal is **already fully extracted**: re-weighting the same 24
structures 4× adds zero information and zero performance, for the forward
model and the generator alike.

The oxide gap is therefore **not a sampling-weight problem**. It is bounded by
the information content of 24 unique oxide structures over the most diverse,
sharpest-featured targets in the dataset (§6.6). The levers that can plausibly
move it are new oxide training structures — more unique hosts and defects, not
duplicates — or model-side changes targeted at sharp localized states.

## Open questions

- A species-constrained or type-prior template variant, the clearest
  identified lever on the dominant template-mode failure (§5).
- Some defect labels (`V48_V44` and similar) remain to be clarified.

## Data files

| directory | contents |
|---|---|
| `dmx2_forward_eval/` | `metrics.csv`, `overlays/*.png` |
| `dmx2_generator_eval/` | `sweep.csv`, `sweep_w{0,1,2,3,5}.csv/.agg.json`, `test_metrics.csv`, `test_k20_w1.*`, `test_k20_uncond.*`, `val_template_k5_w1.*`, `val_template_k20_weak.*`, `template_vs_free.csv`, `template_vs_free_nonvacancy.csv`, `template_arm_geo.csv`, `free_arm_geo.csv`, `partD_weak_classes.csv` |
| `dmx2_relax_screen/` | `calibration.csv`, `generated.csv`, `dosdrift.csv`, `defect_stoich.csv`, `defect_geometry.csv`, `judge_locality.csv`, `reselect_a.csv`, `d2_subset.csv`, `d2_subset_v2.csv`, `relax_val_{free,template,template_k20}.csv` |
| `dmx2_family_benchmark/` | `forward_by_family.csv`, `generator_by_family.csv`, `recovery_by_family.csv`, `pristine_vs_defective.csv`, `distribution.csv`, `train_stats.csv`, `ablation/` |
| `dmx2_family_benchmark/ablation/` | `forward_ox4_metrics.csv`, `forward_ox4_by_family.csv`, `generator_ox4_test_k20_w1.csv`, `generator_ox4_by_family.csv`, `forward_loo_metrics.csv`, `forward_loo_by_family.csv` |

Generated-structure CIF dumps and prediction blobs (`eval/preds/*.pt`) are not
committed; they are regenerated by the commands in the top-level README.
