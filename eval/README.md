# Evaluation scripts

Every number and figure in [`results/RESULTS.md`](results/RESULTS.md) comes from
one of the scripts below. They are listed in pipeline order.

Run all of them **from the repository root** with the `dosmatgen-cu128`
environment active (see the top-level README), e.g. `python eval/dmx2_roundtrip.py ...`.

## What each script needs

| requirement | meaning |
|---|---|
| **GPU + checkpoints** | loads a trained model; needs a CUDA GPU and the run directories under `outputs/` (see the checkpoint table in the top-level README) |
| **dataset JSONs** | needs `data/dmx2_dos/{train,val,test}.json`, built by `build_dmx_dos_json.py` from the raw DOS data |
| **prediction blobs** | needs a `.pt` file of generated candidates under `eval/preds/`; not committed, produced by `generate_for_eval.py` or `dmx2_template_gen.py` |
| **committed CSVs** | re-analyses the CSVs already in `results/`; no GPU and no checkpoints |

The example commands name the run directories used for the published results
(`outputs/260714_*`). Download them as described under **Pretrained
checkpoints** in the top-level README and unpack into `outputs/`, and every
command below works verbatim; otherwise train your own (top-level README,
step 4) and substitute your run directory.

Two paths are hardcoded near the top of the model-loading scripts —
`JUDGE_DIR`/`JUDGE_CKPT` (forward model) and `GEN_DIR`/`GEN_CKPT` (generator).
If you retrain into a new timestamped run directory, either pass the
corresponding CLI flag where one exists or edit those constants.

---

## 1. `generate_for_eval.py` — DOS → structure generation

Generates *k* candidate structures per target in a split, conditioned on that
target's per-atom DOS, and writes them as a prediction blob keyed by
`structure_id` so each candidate stays paired with its ground truth. Use
`diff_ratio=1.0`: anything lower seeds the reverse diffusion from the noised
ground-truth structure and leaks its geometry into the result.

- **Needs:** GPU + generator checkpoint, dataset JSONs
- **Writes:** the `--out` blob (e.g. `eval/preds/dmx2_test_k20_w1.pt`)
- `--root_path` supplies `hparams.yaml` and the scalers; `--ckpt` picks the
  weights. Without `--ckpt` the script globs `*.ckpt` and requires exactly one,
  which fails if the directory also holds a `last.ckpt`.

```bash
mkdir -p eval/preds
python eval/generate_for_eval.py \
    --root_path outputs/260714_125010_dmx2_cfg_ft \
    --ckpt outputs/260714_125010_dmx2_cfg_ft/epoch=789-step=12640.ckpt \
    --split test --k 20 --w 1.0 --diff_ratio 1.0 \
    --out eval/preds/dmx2_test_k20_w1.pt
```

Add `--unconditional` for the DOS-blind control arm (RESULTS §2).

## 2. `dmx2_roundtrip.py` — round-trip scoring of generated structures

Runs the forward model as a judge over every candidate in a blob: predicts each
candidate's DOS and takes the MAE against the DOS the generation was
conditioned on, plus structural and compositional validity. Produces the
per-candidate CSV and the aggregate JSON behind RESULTS §2.

- **Needs:** GPU + forward checkpoint, dataset JSONs, a prediction blob
- **Writes:** `--out_csv` and a sibling `.agg.json`

```bash
python eval/dmx2_roundtrip.py \
    --pred_blob eval/preds/dmx2_test_k20_w1.pt \
    --target_json data/dmx2_dos/test.json \
    --out_csv eval/results/dmx2_generator_eval/test_k20_w1.csv
```

## 3. `dmx2_forward_eval.py` — forward-model test metrics

Predicts per-atom DOS for the test split, inverse-transforms to physical units,
and reports MAE in three energy windows at two levels against two training-set
baselines (global mean and per-element mean). Source of RESULTS §1 and the
overlay figures. Also exposes `predict_split()`, reused by other scripts.

- **Needs:** GPU + forward checkpoint, dataset JSONs
- **Writes:** `results/dmx2_forward_eval/metrics.csv`, `summary.md`, `overlays/*.png`

```bash
python eval/dmx2_forward_eval.py \
    --run_dir outputs/260714_121340_dmx2_forward_ft \
    --ckpt outputs/260714_121340_dmx2_forward_ft/epoch=89-step=1440.ckpt
```

## 4. `dmx2_relax_screen.py` — MACE relaxation screen

Relaxes structures with the MACE-MPA-0 medium potential (positions only, cell
fixed, FIRE, fmax 0.05 eV/Å, ≤200 steps) and records convergence, step count,
energy drop and RMSD. Run `--set gt` first: relaxing the ground-truth test
structures defines the in-band reference distribution that generated candidates
are judged against. Geometry screen only — the potential is non-magnetic.

- **Needs:** GPU, a MACE-MPA-0 medium checkpoint, and `mace-torch` installed
  (a separate environment from `dosmatgen-cu128` — see *Second environment:
  MACE* in the top-level README for how to install it and fetch the potential);
  dataset JSONs for `--set gt`, a prediction blob for `--set gen`. The
  checkpoint is located via `--mace_model`, else `$MACE_MODEL_PATH`, else the
  literal fallback `MACE-MPA-0-medium.model` in the working directory.
- **Writes:** `results/dmx2_relax_screen/calibration.csv` (gt) or `generated.csv`
  (gen), plus relaxed CIFs under `structures_relaxed/<tag>/`

```bash
export MACE_MODEL_PATH=/path/to/MACE-MPA-0-medium.model
python eval/dmx2_relax_screen.py --set gt      # calibration band, 62 structures
python eval/dmx2_relax_screen.py --set gen     # all generated candidates
```

## 5. `dmx2_relax_dosdrift.py` — DOS drift through relaxation

Re-runs the judge on the *relaxed* geometries and merges with the pre-relaxation
round-trip scores, giving the before/after comparison in RESULTS §3.3 and the
correlation between DOS drift and relaxation RMSD.

- **Needs:** GPU + forward checkpoint, dataset JSONs, relaxed CIFs from step 4,
  and the pre-relaxation round-trip CSV from step 2
- **Writes:** `results/dmx2_relax_screen/dosdrift.csv`

```bash
python eval/dmx2_relax_dosdrift.py
```

## 6. `dmx2_defect_stoich.py` — D1, defect stoichiometry recovery

For each candidate, compares its element-count delta against the host's pristine
supercell with the ground-truth delta. Recovery requires an exact species and
count match, which is stricter than plain composition matching. RESULTS §3.4.

- **Needs:** dataset JSONs and a prediction blob; no GPU
- **Writes:** `results/dmx2_relax_screen/defect_stoich.csv`

```bash
python eval/dmx2_defect_stoich.py
```

## 7. `dmx2_defect_geometry.py` — D2/D3, geometric defect verification

Maps a structure onto its pristine host reference by nearest-neighbour matching
in fractional space and classifies the resulting defect signature (vacancies,
additions, substitutions) plus host integrity. Two modes:

- `--mode gen_verify` — strict geometric recovery of the selected candidates
  (RESULTS §3.5). Needs an origin/lattice-operation alignment search, since
  diffusion output carries an arbitrary origin.
- `--mode gt_locality` — judge accuracy on defect-adjacent versus host-bulk
  atoms (RESULTS §3.6). This mode loads the forward model.

- **Needs:** dataset JSONs, relaxed CIFs from step 4, and a subset CSV for
  `gen_verify`; GPU + forward checkpoint for `gt_locality`
- **Writes:** `results/dmx2_relax_screen/defect_geometry.csv` (gen_verify) or
  `judge_locality.csv` (gt_locality)

```bash
python eval/dmx2_defect_geometry.py --mode gen_verify \
    --subset_csv eval/results/dmx2_relax_screen/d2_subset.csv
python eval/dmx2_defect_geometry.py --mode gt_locality
```

## 8. `dmx2_template_gen.py` — defect-template generation

Generates a defect inside a known host: pristine host sites are pinned
(coordinates, species and lattice held fixed at every denoising step) and only
the defect-involved atoms are free, starting from pure noise. Output is in the
same blob format as `generate_for_eval.py`, so steps 2–7 apply unchanged.

- **Needs:** GPU + generator checkpoint, dataset JSONs
- **Writes:** the `--out` blob

```bash
python eval/dmx2_template_gen.py --k 5 --w 1.0 \
    --out eval/preds/dmx2_val_template_k5_w1.pt
python eval/dmx2_template_gen.py --smoke    # 2 targets, asserts the pinning invariants
```

## 9. `dmx2_template_compare.py` — template versus free generation

Scores both arms on the same 53 defective val targets: pre- and
post-relaxation round-trip MAE, D1 stoichiometry and geometric recovery. Source
of the all-53 column in RESULTS §5.

- **Needs:** GPU + forward checkpoint, dataset JSONs, both arms' blobs and their
  relaxed CIFs
- **Writes:** `results/dmx2_generator_eval/template_vs_free.csv`,
  `template_arm_geo.csv`, `free_arm_geo.csv`

```bash
python eval/dmx2_template_compare.py
```

## 10. `dmx2_template_reframe.py` — non-vacancy reframing

Establishes that vacancy-class template targets have no free atoms — the
candidate is fully determined by template construction, not generated — and
recomputes the comparison restricted to targets where the model actually
generates something. Source of the non-vacancy and `n_free > 0` columns in
RESULTS §5.

- **Needs:** GPU + forward checkpoint, the template blob and the round-trip CSVs
- **Writes:** `results/dmx2_generator_eval/template_vs_free_nonvacancy.csv`

```bash
python eval/dmx2_template_reframe.py
```

## 11. `dmx2_family_benchmark.py` — per-family breakdown

Splits every stage by host anion (O/S/Se/Te) with structure-level bootstrap 95%
CIs, and computes the training-set statistics that explain the oxide gap.
Source of RESULTS §6.

- **Needs:** the committed CSVs in `results/` for tables 1-6, plus
  `data/dmx2_dos/train.json` for the training-set statistics table. No GPU
  and no checkpoints; everything else is re-analysis.
- **Writes:** `results/dmx2_family_benchmark/*.csv` (7 tables)

```bash
python eval/dmx2_family_benchmark.py
```

## 12. `dmx2_ablation_eval.py` — oxide ablation readout

Evaluates a retrained ablation checkpoint (oxide-oversampled or
leave-oxide-out) against the pre-registered criterion, reusing the committed
per-structure baselines and the same bootstrap procedure as the family
benchmark. Source of RESULTS §7.

- **Needs:** GPU + an ablation checkpoint, dataset JSONs. **The ablation
  checkpoints are not published** — reproduce them by retraining with the
  committed configs, which differ from the baseline only in the training set:
  `configs/dos_forward_dmx2_ox4.yml` (oxides oversampled 4x) and
  `configs/dos_forward_dmx2_loo.yml` (oxides removed). Build their datasets
  first, as described in RESULTS section 7, then run
  `run_forward_finetune.py --config <that config>`.
- **Writes:** `results/dmx2_family_benchmark/ablation/forward_<tag>_metrics.csv`
  and `forward_<tag>_by_family.csv`

```bash
# after training the ox4 arm into outputs/<your_run_dir>/
python eval/dmx2_ablation_eval.py \
    --run_dir outputs/<your_run_dir> \
    --ckpt outputs/<your_run_dir>/<best>.ckpt \
    --tag ox4
```

## `dmx_eval_utils.py` — shared helpers

Not runnable. Provides `smact_validity` and `structure_validity` (the validity
columns in the round-trip CSVs) and `lattices_to_params`, ported from DiffCSP's
evaluation utilities.

