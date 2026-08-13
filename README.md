# Total-DOS ↔ structure generation for 2D defect materials (EDOS arm)

Conditional generation and property prediction linking **total electronic
density of states** to **atomic structure**, for 2D transition-metal
dichalcogenide/oxide monolayers with point defects.

Two models, fine-tuned from the released [DOSMatGen](https://arxiv.org/abs/2504.06249)
MP-DOS checkpoints:

| model | direction | what it does |
|---|---|---|
| **forward** (`CSPProperty`) | structure → DOS | predicts each atom's total DOS curve |
| **generator** (`CSPDiffusion`, CFG) | DOS → structure | denoises coords+species+lattice conditioned on a target DOS |

The forward model doubles as the **judge** that scores generated structures
(round-trip: generate a structure from a DOS, predict its DOS, compare).

> **The conditioning is PER-ATOM.** The model takes an `[N, 400]` array — one
> 400-point DOS curve *per atom* — plus the atom count `N`. It does **not**
> take a single total-DOS vector for the cell. If you only have a cell-summed
> DOS you cannot use this model as-is; per-atom (site-projected) curves are
> required. The 400 points span **[−10, +10] eV relative to E_F**.

Dataset: 620 spin-polarized HSE06 static DOS calculations (44–49 atoms/cell).
**The data itself is not distributed with this repo** — see below for the
layout the pipeline expects.

> **This branch adds a spin-resolved option.** Sections 1–7 describe the
> released total-DOS models and apply unchanged. Section 8 covers the
> optional `[total(400) || m(400)]` extension: spin-resolved targets, an
> 800-d conditioning adapter for the generator, and an 800-d split output
> head for the forward model. With those options off, everything behaves
> exactly as the released total-DOS code.

---

## 1. Environment

Blackwell/B200-capable env. `build_cu128_env.sh` creates conda env
**`dosmatgen-cu128`**: python 3.12, **torch 2.7.0+cu128**, torch_geometric
2.7.0, PyG compiled extensions (`pyg_lib`, `torch_scatter`, `torch_sparse`,
`torch_cluster`, `torch_spline_conv`) built for `torch-2.7.0+cu128`, plus
lightning / ase / pymatgen / omegaconf / einops / wandb / scipy and the eval
deps (`matminer`, `smact`, `pyxtal`).

```bash
bash build_cu128_env.sh          # one time; ~15 min
conda activate dosmatgen-cu128
```

This env runs on **both** L4 (sm_89) and B200 (sm_100) — the compiled PyG
extensions are the part that actually needs the Blackwell build, which is why
the smoke test exercises them:

```bash
python gpu_smoke_b200.py         # asserts B200 + sm_100, cuda matmul+backward,
                                 # torch_scatter and a PyG GCNConv on cuda
```

`environment.yml` is the older CUDA-12.4 / torch-2.4.1 env (`dosmatgen`); it
works on L4 but **not** on Blackwell.

Training and eval are single-GPU. On an SBATCH cluster, request one GPU and
`conda activate dosmatgen-cu128` before the python line; nothing else is
needed.

### Second environment: MACE (relaxation screen only)

`eval/dmx2_relax_screen.py` is the one script that does not run in
`dosmatgen-cu128` — MACE pins its own PyTorch, so give it a separate
environment:

```bash
conda create -n mace python=3.12 -y && conda activate mace
pip install mace-torch          # this work used mace-torch 0.3.12
```

It needs the **MACE-MPA-0 medium** foundation potential. Fetch it once with
MACE's own downloader, which caches the file under `~/.cache/mace/`:

```bash
python -c "from mace.calculators import mace_mp; mace_mp(model='medium-mpa-0')"
```

The same checkpoint is published as a release asset of the MACE foundation
models (ACEsuit `mace-foundations`, file `mace-mpa-0-medium.model`) if you
prefer to download it directly.

Point the script at the file with **`MACE_MODEL_PATH`**, or per-run with
`--mace_model`:

```bash
export MACE_MODEL_PATH=$HOME/.cache/mace/<the downloaded file>
python eval/dmx2_relax_screen.py --set gt
```

Resolution order is `--mace_model` → `$MACE_MODEL_PATH` → the literal fallback
`MACE-MPA-0-medium.model`, which is only resolved relative to the working
directory. In other words, if you set neither, the run fails unless that file
sits in the repository root — there is no automatic download at run time.

Nothing else in the pipeline uses MACE, so you can skip this environment
entirely unless you are reproducing the relaxation screen (RESULTS section 3).

## 2. Expected raw-data layout

One directory per structure, named `<host>_<defect>`, directly under a single
root (referred to below as `$DMX`):

```
$DMX/
├── mos2-A_Defect-Free/
│   ├── vasprun.xml          <- required
│   └── OUTCAR               <- optional (audit only: magmoms, accuracy line)
├── mos2-A_Vacancy-Anion/
│   ├── vasprun.xml
│   └── OUTCAR
├── ws2-B_Doped-O/
│   └── ...
└── ...                      (624 folders in the reference dataset)
```

The `<host>_<defect>` naming is load-bearing: everything before the **first**
underscore is the host id, the rest is the defect label, and both are used for
stratified splitting and per-family analysis.

Each `vasprun.xml` must come from a **spin-polarized (`ISPIN=2`) static
(`NSW=0`) DOS run** — the reference set used HSE06 with `NEDOS=3000`. The
pipeline reads exactly four things from it:

| field | used for |
|---|---|
| `vasprun.pdos[i][orbital][spin]` | per-atom DOS, summed over all orbitals and both spins |
| `vasprun.tdos.energies`, `.efermi` | energy axis, shifted to `E − E_F` |
| `vasprun.structures[-1]` | positions (Cartesian), cell, atomic numbers |
| parent folder name | `structure_id` |

`OUTCAR` is optional and only read by the audit step.

## 3. Reproduction pipeline

All commands are run from the repo root with `dosmatgen-cu128` active. Set
`DMX=/path/to/your/dataset` first — `--dmx_dir` is required and has no default,
so every step that reads the raw data takes it explicitly.

### Step 1 — audit the raw data

Read-only. One row per folder: convergence, ISPIN/NEDOS/NSW, E_F, the actual
energy-window coverage, per-atom DOS min/max/finiteness, and md5-based
duplicate detection.

```bash
find $DMX -name vasprun.xml | sort | xargs md5sum > /tmp/new.md5
cp /tmp/new.md5 /tmp/old.md5          # no prior dataset to compare against

python audit_dmx2.py --dmx_dir $DMX --md5-new /tmp/new.md5 --md5-old /tmp/old.md5
```

Writes `data/dmx2_audit.csv`, `data/dmx2_magmoms.json`, `data/dmx2_errors.txt`.
(`--md5-new/--md5-old` populate the overlap columns comparing a new dataset
against an older one; pointing both at the same file is fine when there is no
older set. Note that the committed CSV's `in_old` / `identical_to_old` columns
were computed against an earlier version of this dataset, so with
`old.md5 == new.md5` those two columns will not match the committed file. The
other 25 columns, including `md5`, reproduce exactly.)

The committed `data/dmx2_audit.csv` and `data/dmx2_audit_summary.md` are the
audit of the reference dataset. It found **4 electronically unconverged
(NELM-saturated) runs**, listed in `data/dmx2_exclude.json`; the build skips
them, giving 620 usable structures.

### Step 2 — the frozen split

```bash
python make_dmx2_split.py             # writes splits/dmx2_v1.json
```

Design, in three lines:

- **Host-stratified**: split within each host's defect list, so every host
  appears in train *and* val *and* test — the task is defect-conditional
  generation in a known host, not host extrapolation.
- **Deterministic**: seed 2024, sorted id lists, reads `data/dmx2_audit.csv`;
  re-running reproduces the file byte-for-byte.
- **Committed as explicit id lists**, never re-randomized — the build script
  hard-fails if the parsed folder set ≠ the split id set.

**`splits/dmx2_v1.json` (496/62/62) is EXACTLY the split every published
result in `eval/results/` used.** Do not regenerate it to reproduce those
numbers — just use the committed file.

### Step 3 — build the training JSONs

```bash
python build_dmx_dos_json.py --dmx_dir $DMX
```

Writes `data/dmx2_dos/{train,val,test}.json` + `build_meta.json`.

Target recipe, per structure:

1. per-atom PDOS **summed over all orbitals and both spins** → `[N, nE]`
2. energy axis shifted: `x = tdos.energies − tdos.efermi`
3. **linear** `interp1d` onto `linspace(-10, 10, 400)`, `bounds_error=False`,
   **`fill_value=0`** (zero fill outside the source range)
4. positions/cell/atomic_numbers from the ASE atoms of `structures[-1]`

Record schema (the datamodule's expected format):

```python
{'structure_id': str, 'positions': (N,3), 'cell': (3,3),
 'atomic_numbers': (N,), 'y': (N,400)}
```

Options: `--out_dir`, `--split_file`, `--exclude_file` (see `--help`).

### Step 4 — training

Both models **warm-start from the released DOSMatGen MP-DOS checkpoints**,
downloadable from the upstream repo
([weights link](https://u.pcloud.link/publink/show?code=XZ0kE45ZOuEPUrHzQU5ciR2pXKCkDzNY3F9k)).
Place them so the config `pretrain_dir` keys resolve:

```
outputs/260518_164718_dos_cfg/    <- MP-DOS CFG diffusion ckpt + hparams.yaml + scalers
outputs/dos_forward_model/        <- MP-DOS forward ckpt + hparams.yaml + scalers
```

Both fine-tunes **reuse the pretrained scalers** (`scaler_path=pretrain_dir`)
rather than refitting, so y/lattice stay on the normalization the checkpoints
expect. Architecture is rebuilt from each checkpoint's own `hparams.yaml`,
guaranteeing a 1:1 weight load (the forward load is verified strict:
`missing == unexpected == []`). Fine-tune LR 1e-4 (pretrain used 1e-3), 1000
epochs, best-val checkpointing.

```bash
# forward model (structure -> DOS)
python run_forward_finetune.py --config configs/dos_forward_dmx2_ft.yml

# generator (DOS -> structure, classifier-free guidance)
python run_diffusion_CFG_finetune.py --config configs/dos_cfg_dmx2_ft.yml
```

Checkpoints land in `outputs/<YYMMDD_HHMMSS>_<run_name>/`. Roughly **2 s/epoch
on a B200**, so 1000 epochs is well under an hour per model. Progress is logged
to `wandb`; drop the `logging.wandb` block from the config to run offline. Add
`--resume path/to/last.ckpt` to continue an interrupted run.

The production checkpoints behind every committed result:

| model | run dir | checkpoint | val loss |
|---|---|---|---|
| forward / judge | `outputs/260714_121340_dmx2_forward_ft` | `epoch=89-step=1440.ckpt` | 0.2370 |
| generator | `outputs/260714_125010_dmx2_cfg_ft` | `epoch=789-step=12640.ckpt` | 0.6042 |

### Pretrained checkpoints

Both production run directories are published, so the example commands
throughout this repository work verbatim without retraining:

**Download:** `https://huggingface.co/paprakash/edos-dmx2`

Unpack into `outputs/` at the repository root, keeping the run-directory names:

```
outputs/
├── 260714_121340_dmx2_forward_ft/
│   ├── epoch=89-step=1440.ckpt
│   ├── hparams.yaml
│   ├── lattice_scaler.pt
│   └── prop_scaler.pt
└── 260714_125010_dmx2_cfg_ft/
    ├── epoch=789-step=12640.ckpt
    ├── hparams.yaml
    ├── lattice_scaler.pt
    └── prop_scaler.pt
```

Each run directory carries its own `hparams.yaml` (the architecture is rebuilt
from it, so the weights load strictly) and the two scalers the model was
calibrated with. The two `lattice_scaler.pt` / `prop_scaler.pt` pairs are
byte-identical across the run directories by design — both fine-tunes reuse the
pretrained MP-DOS scalers rather than refitting.


These paths are the **defaults** in the eval scripts. If you retrain into a new
timestamped directory instead, pass `--run_dir` / `--judge_dir` / `--ckpt`, or
edit the `JUDGE_DIR` / `GEN_DIR` constants at the top of the scripts that
hardcode them (`dmx2_template_compare.py`, `dmx2_relax_dosdrift.py`,
`dmx2_template_gen.py`).

### Step 5 — generation and evaluation

`eval/README.md` documents every evaluation script in pipeline order — what
each one does, its exact command, whether it needs a GPU and checkpoints or
just re-analyses the committed CSVs, and what it writes. The most common
entry points:

**Conditional generation** (`diff_ratio=1.0` = pure noise; anything lower
warm-starts from the noised ground-truth structure and leaks its geometry into
a "reconstruction" metric):

```bash
mkdir -p eval/preds
python eval/generate_for_eval.py \
    --root_path outputs/260714_125010_dmx2_cfg_ft \
    --ckpt outputs/260714_125010_dmx2_cfg_ft/epoch=789-step=12640.ckpt \
    --split test --k 20 --w 1.0 --diff_ratio 1.0 \
    --out eval/preds/dmx2_test_k20_w1.pt
```

`--w` is the guidance scale (1.0 chosen on val); `--k` candidates per target;
`--unconditional` gives the DOS-blind control. Atom count is fixed to the
target's.

**Template mode** — generate a defect *inside a known host*: pristine host
sites are pinned (fixed coords+species+lattice at every denoising step), only
the defect-involved atoms are free and start from pure noise:

```bash
python eval/dmx2_template_gen.py --k 5 --w 1.0 \
    --out eval/preds/dmx2_val_template_k5_w1.pt
python eval/dmx2_template_gen.py --smoke     # 2 targets, asserts pinning invariants
```

**Round-trip scoring** (judge predicts each candidate's DOS, MAE vs the
conditioning target; also structural/compositional validity):

```bash
python eval/dmx2_roundtrip.py \
    --pred_blob eval/preds/dmx2_test_k20_w1.pt \
    --target_json data/dmx2_dos/test.json \
    --out_csv eval/results/dmx2_generator_eval/test_k20_w1.csv
```

**Forward-model metrics** (three energy windows, two levels, two baselines):

```bash
python eval/dmx2_forward_eval.py
```

**Relaxation / realism screen** (MACE-MPA-0, positions-only FIRE, cell fixed,
fmax 0.05 eV/Å, ≤200 steps — a *geometry* screen; the potential is
non-magnetic, so it is not energetics truth). Needs `mace-torch` in a separate
environment and a local MACE-MPA-0 medium checkpoint:

```bash
export MACE_MODEL_PATH=/path/to/MACE-MPA-0-medium.model
python eval/dmx2_relax_screen.py --set gt      # calibration band on the 62 GT test structures
python eval/dmx2_relax_screen.py --set gen     # all generated candidates
```

**Per-family benchmark** (host anion O/S/Se/Te; bootstrap 95% CIs, 10k
resamples) — reuses the committed `dmx2_forward_eval/metrics.csv` and reads
`data/dmx2_dos/train.json` for the training-set statistics. No GPU needed; runs
in seconds:

```bash
python eval/dmx2_family_benchmark.py
```

## 4. Splits and data quality

Two split files are committed:

| file | sizes | status |
|---|---|---|
| `splits/dmx2_v1.json` | 496 / 62 / 62 | what **all committed results** used |
| `splits/dmx2_v1_1.json` | 485 / 62 / 61 | leak-repaired 2026-08-02 — **use this for new training** |

The v1 audit later turned up **duplicate host entries**: `ws2-B` and `ws2-c`
are the same host under two labels, producing identical-geometry structure
pairs, **6 of which straddled a train/eval boundary**. v1_1 removes one member
of each duplicate pair (12 ids total, including two mislabeled entries), keeping
the lower-energy member where the pair's DFT states differed. **No id ever
moves between splits and the seed is unchanged** — v1_1 is a pure subtraction
from v1, so it is directly comparable.

The leak inflates the committed v1 numbers by a small optimistic bias (6 eval
targets had a near-twin in train, out of 124). It was **not** re-run, so treat
the published figures as marginally optimistic rather than invalid.

```bash
python make_dmx2_split_v1_1.py    # writes splits/dmx2_v1_1.json and
                                  # data/dmx2_exclude_v1_1.json
python build_dmx_dos_json.py --dmx_dir $DMX \
    --split_file splits/dmx2_v1_1.json \
    --exclude_file data/dmx2_exclude_v1_1.json \
    --out_dir data/dmx2_dos_v1_1
```

`make_dmx2_split_v1_1.py` writes its own exclusion list to
`data/dmx2_exclude_v1_1.json` (the 4 v1 exclusions plus the 12 leak exclusions)
and leaves `data/dmx2_exclude.json` untouched, so the two split versions are
independent and a v1 build still works after generating v1_1. Both exclusion
files are committed.

(The per-pair decision table, including the DFT energies used to adjudicate
the state-mismatch twins, is reproduced inline in `make_dmx2_split_v1_1.py`.)

## 5. Known limitations

- **Per-atom conditioning is required.** The model consumes `[N, 400]` — one
  curve per atom — and the atom count. A single cell-summed DOS is not enough.
  This is the single biggest barrier to applying the model to a DOS you
  measured or sketched rather than computed.
- **Energy-window truncation.** Only ~18% of the source runs extend to +10 eV
  (median top of range +8.65 eV, 10th percentile +7.04 eV). Above roughly
  **+5.5 eV** the target for most structures is zero-fill from the interpolation,
  not physics. Metrics are therefore reported in three windows — full
  `[−10, 10]`, all-valid `[−10, +5.5]` (covered by every structure), and
  near-E_F `[−2, 2]` (always real data). Trust near-E_F; discount the top.
- **Oxide family gap.** Oxides are 6.5% of the data (40 structures, 8 in test).
  The forward model is ~1.8× worse on them (CI-disjoint from the chalcogenides),
  This is a **data** limit, not a weighting one: a 3× oxide-oversampling
  ablation (`ox4`) is a clean null (0.2118 vs 0.2109), while leave-oxide-out
  degrades oxides by 20% — the existing oxide data is fully used, so closing
  the gap needs new oxide calculations. See `eval/results/RESULTS.md`
  section 7.
- The judge is the forward model, so **all round-trip numbers inherit its blind
  spots**: round-trip MAE measures agreement with the forward model, not with
  DFT.
- The relaxation screen's potential (MACE-MPA-0) is non-magnetic: it certifies
  geometric plausibility, not energetic or magnetic correctness.

## 6. Results index

| directory | what's in it |
|---|---|
| `eval/results/RESULTS.md` | **start here** — every result, organised by experiment |
| `some_structures/` | four worked examples: dataset target + generated structure, with both DOS curves |
| `eval/results/dmx2_forward_eval/` | forward-model test metrics, 3 windows × 2 levels vs 2 baselines, + overlay plots |
| `eval/results/dmx2_generator_eval/` | guidance sweep, conditional vs unconditional, k=20 round-trip, template-vs-free per-candidate CSVs |
| `eval/results/dmx2_relax_screen/` | MACE relaxation screen, GT calibration band, defect geometry/stoichiometry, DOS drift |
| `eval/results/dmx2_family_benchmark/` | per-anion-family breakdown with bootstrap CIs; `ablation/` has the ox4 + leave-oxide-out results |

Headline numbers (test split, states/eV/atom):

| | model | per-element baseline |
|---|---|---|
| forward, full window | **0.131** | 0.168 |
| forward, near-E_F | **0.168** | 0.221 |

| | round-trip mean-of-k | best-of-20 | exact composition |
|---|---|---|---|
| conditional (w=1) | **0.158** | **0.139** | 0.70 |
| unconditional control | 0.269 | 0.227 | 0.00 |

## 7. Repo layout

```
├── audit_dmx2.py               step 1: read-only data audit
├── make_dmx2_split.py          step 2: frozen host-stratified split (v1)
├── make_dmx2_split_v1_1.py     step 2': leak repair -> v1_1
├── build_dmx_dos_json.py       step 3: vaspruns -> train/val/test JSON
├── run_forward_finetune.py     step 4: forward model fine-tune
├── run_diffusion_CFG_finetune.py  step 4: generator fine-tune
├── run_forward_finetune_spin.py   section 8: 800-d split-head fine-tune
├── run_diffusion_CFG_finetune_spin.py  section 8: 800-d generator fine-tune
├── run_diffusion_CFG.py        upstream: MP-DOS pretraining from scratch
├── generate_CFG_{conditional,unconditional}.py   upstream generation entry points
├── gpu_smoke_b200.py           Blackwell smoke test
├── build_cu128_env.sh          env builder (torch 2.7.0+cu128)
├── configs/                    dos_*_dmx2_*.yml (+ dos_cfg.yml = MP-DOS pretrain)
├── dosmatgen/                  model/dataset/diffusion library
├── data/                       audit CSV, exclusions, build_meta, magmoms (committed);
│                               the built dmx2_dos/*.json + *_ori.pt caches are gitignored
├── splits/                     dmx2_v1.json, dmx2_v1_1.json
├── some_structures/            four worked examples (see its README)
└── eval/                       evaluation scripts (see eval/README.md)
    └── results/RESULTS.md      all results, organised by experiment
```

## 8. Spin-resolved (800-d) extension

An optional extension that carries the spin asymmetry alongside the total DOS.

### What the targets look like

Per atom, `y = [total(400) || m(400)]`:

- `total = up + down` — identical to the 400-d targets used everywhere above
- `m = up - down` — the spin asymmetry, zero for a non-spin-polarised calculation

Both halves share the same energy grid, Fermi reference and interpolation as the
released build: 400 bins on `[-10, +10]` eV in `E - E_F`, linear interpolation, zero
fill outside the source range. The first 400 columns of a spin-split build are
therefore identical to a default build of the same structures.

### What this branch adds

The model code for both options already ships in the released branch, default-off:

- **`mag_zero_mixin`** (generator, `dosmatgen/models/cspnet_cfg.py`) splits the DOS
  condition into a total pathway and an m pathway and adds their projections. The m
  pathway is zero-initialised, so at step 0 the model is exactly the released
  total-DOS generator, while the m pathway still receives gradient and can lift off
  on its own.
- **`node_out_split`** (forward model, `dosmatgen/models/cspnet.py`) splits the
  per-atom output head into a total half and an m half. The m half is
  zero-initialised, so the predicted m is zero at step 0 and the backbone is
  unchanged.

What this branch adds on top of that is the parts that were missing: the
spin-resolved target build, the two training entry points that perform the
warm-start surgery, and the configs that switch the options on. That is why the diff
against the released branch is small.

The generator model also contains an alternative tanh-gated conditioning variant
(`mag_gate`); the configs here use `mag_zero_mixin`.

### Step 1 — build spin-resolved targets

```bash
python build_dmx_dos_json.py --dmx_dir $DMX --spin_split
```

Writes `data/dmx2_dos_spin/{train,val,test}.json` plus `build_meta.json`, using the
same frozen split and exclusion list as the total-only build (override with
`--split_file` / `--exclude_file` / `--out_dir`). Every `y` row is 800 wide.

### Step 2 — train the generator

```bash
python run_diffusion_CFG_finetune_spin.py --config configs/dos_cfg_dmx2_spin_ft.yml
```

### Step 3 — train the forward model

```bash
python run_forward_finetune_spin.py --config configs/dos_forward_dmx2_spin_ft.yml
```

Both warm-start from the **released EDOS checkpoints** — the same run directories
described under *Pretrained checkpoints* in section 3. Set `pretrain_dir` in the
config to the run directory you downloaded; each entry point expects that directory
to hold exactly one `.ckpt` alongside `prop_scaler.pt` and `lattice_scaler.pt`, which
is the layout the released checkpoints ship in.

Both scripts rebuild the property scaler as `[released total stats || dataset m
stats]`, so the grafted total pathway keeps the normalisation it was trained with,
and both give the freshly initialised m pathway a higher learning rate than the
inherited weights (`optim.new_pathway_lr`). Checkpoints land in
`outputs/<YYMMDD_HHMMSS>_<run_name>/` as usual.

### Step 4 — forward inference

```python
from glob import glob
import torch
from omegaconf import OmegaConf
from dosmatgen.diffusion.property import CSPProperty

run_dir = "outputs/<your_spin_forward_run>"
cfg = OmegaConf.load(f"{run_dir}/hparams.yaml")
model = CSPProperty(**cfg)
sd = torch.load(glob(f"{run_dir}/*.ckpt")[0], map_location="cpu",
                weights_only=False)["state_dict"]
model.load_state_dict(sd, strict=True)
model.eval()

scaler = torch.load(f"{run_dir}/prop_scaler.pt", map_location="cpu",
                    weights_only=False)
pred_node, _ = model.infer(batch)              # [n_atoms, 800], scaled
pred = scaler.inverse_transform(pred_node)     # physical units
total, m = pred[:, :400], pred[:, 400:]
```

`batch` is a PyG batch from `CrystalDataModule`, exactly as for the 400-d model.

### No checkpoints are published for this branch

The released EDOS checkpoints are 400-d. There are no 800-d checkpoints to download —
train them with the configs above.

### Compatibility with the total-DOS code

`mag_zero_mixin` and `node_out_split` default to off. With them off, the model classes
construct exactly as in the released branch and the released configs load and train
unchanged; `build_dmx_dos_json.py` without `--spin_split` produces the same 400-d
targets as before.

What works at 800-d:

- **training** — both entry points above
- **forward inference** — as in step 4; the prediction is `[n_atoms, 800]`
- **conditional generation** — `generate_CFG_conditional.py` runs against an 800-d
  generator run directory and writes structures. It reads the condition width from
  the data and the model, so it needs no flag for this

What does not:

- the evaluation scripts under `eval/` assume a 400-d energy grid and have not been
  extended. Against an 800-d model they fail with a NumPy `IndexError` raised by a
  400-length window mask rather than returning wrong numbers, but the message does
  not name the cause. Run them against 400-d models.

`generate_CFG_unconditional.py` fills the DOS condition slot with an arbitrary
constant placeholder rather than a real spectrum; it now sizes that placeholder from
`pred_dim`, so at 800-d it spans both halves. Nothing should be read into its value,
and 400-d behaviour is unchanged.

### A note on the m channel

The spin-up/spin-down assignment in a DFT calculation is an arbitrary global
convention rather than a physical orientation, and in our experiments conditioning the
generator on the signed m channel did not improve round-trip fidelity over the
total-only baseline, so the m channel should be treated as experimental.

## Acknowledgements and citation

Built on **DOSMatGen** (Jia, Ganesh & Fung), which is itself informed by
**DiffCSP**. The `dosmatgen/` library and the `run_diffusion_CFG.py` /
`generate_CFG_*.py` entry points are upstream code; the `dmx2` pipeline,
configs, and eval suite are this work.

```
@article{jia2025electronic,
  title={Electronic Structure Guided Inverse Design Using Generative Models},
  author={Jia, Shuyi and Ganesh, Panchapakesan and Fung, Victor},
  journal={arXiv preprint arXiv:2504.06249},
  year={2025}
}
```

See `LICENSE` for licensing.
