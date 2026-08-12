# Example structures

Four worked examples of the DOS → structure task: a dataset target, the
structure the generator produced for it, and the two DOS curves side by side.
They are meant to be opened and looked at — nothing here is needed to run the
pipeline.

| example | host | defect type | composition | round-trip MAE |
|---|---|---|---|---|
| `crte2_Ti_adatom` | CrTe₂ | adatom (Ti) | Cr₁₆Te₃₂Ti | 0.0720 |
| `snte2_Cu_doped` | SnTe₂ | substitution (Cu on a metal site) | CuSn₁₅Te₃₂ | 0.0815 |
| `mnse2-A_Vacancy-Metal` | MnSe₂ (polymorph A) | vacancy (metal site) | Mn₁₅Se₃₂ | 0.1273 |
| `rus2_Int-Anion` | RuS₂ | interstitial (extra anion) | Ru₁₆S₃₃ | 0.1289 |

Round-trip MAE is the post-relaxation, full-window `[-10, 10]` eV value in
states/eV/atom, taken from the selection table
`eval/results/dmx2_relax_screen/reselect_a.csv`. For scale, the forward model's
own error on ground-truth test structures is 0.1311 (see `eval/results/RESULTS.md`
section 1), so all four sit at or below the level at which the model can
distinguish a generated structure from a real one.

Each candidate was chosen by the selection rule described in RESULTS section 4:
among candidates whose MACE relaxation stayed inside the ground-truth band and
whose defect stoichiometry was correct, the one with the lowest post-relaxation
round-trip MAE. In all four the generated composition matches the target
exactly.

## Files in each folder

| file | what it is |
|---|---|
| `target.cif` | the dataset structure — the ground-truth geometry the DOS was computed from |
| `target_dos_dft.csv` | the DOS that structure actually has |
| `generated_relaxed.cif` | what the generator produced from that DOS, after a MACE relaxation |
| `generated_dos_predicted.csv` | the DOS the forward model predicts for the generated structure |

Both CSVs have 400 rows on the same grid, `linspace(-10, 10, 400)` in `E − E_F`,
with columns `energy_eV_minus_EF` and `dos_states_per_eV_per_atom`. The DOS is
the cell DOS divided by the atom count, so the two curves in a folder are
directly comparable.

## Provenance — the two curves are not the same kind of quantity

**`target_dos_dft.csv` is DFT.** It is the spin-polarised HSE06 static-DOS
calculation that ships with the dataset for that structure, summed over
orbitals and both spins and averaged over atoms. It is a first-principles
result.

**`generated_dos_predicted.csv` is a model prediction, not a DFT
calculation.** It is the forward model (`outputs/260714_121340_dmx2_forward_ft`,
`epoch=89-step=1440.ckpt`) evaluated on the MACE-relaxed generated geometry. No
DFT was run on any generated structure in this repository, so agreement between
the two curves means the generated structure reproduces the target DOS *as
judged by the forward model* — it is not independent first-principles
confirmation.

Generated geometries were relaxed with the non-magnetic MACE-MPA-0 potential
(positions only, cell fixed), which certifies geometric plausibility and not
energetics or magnetism.
