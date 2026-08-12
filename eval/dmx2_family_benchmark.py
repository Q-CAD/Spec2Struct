"""Per-family (host anion: O/S/Se/Te) benchmark across the whole dmx2 pipeline.

Motivation: the most visible failure cases are oxides. This quantifies family
effects at every level as committed artifacts, with bootstrap 95% CIs
(oxides are only 8 test structures — CIs are mandatory, not optional).

Per-family tables (results/dmx2_family_benchmark/*.csv):
  1 distribution         family x split, family x pristine/defective
  2 forward_by_family    model + both baselines + model/baseline ratios,
                         cell & per-atom, full/allvalid/near-E_F, boot CIs
  3 generator_by_family  round-trip mean-k / best-of-k (test k20 w=1),
                         unconditional control, selected post-relax MAE
  4 pristine_vs_defective within family (forward + generator)
  5 recovery_by_family   D1 stoichiometry (per-cand + best-of-20), D2 status

Train-set statistics ("oxides are intrinsically harder", quantified):
  7 train_stats          per-family target diversity (mean pairwise cell-DOS
                         MAE), peak sharpness (mean |dDOS/dE|, max/mean),
                         DOS max, energy-window coverage (emax_data)

All curves cell-level = mean of per-atom y over atoms, states/eV/atom.
"""
import json
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "eval/results/dmx2_family_benchmark"
GEV = REPO / "eval/results/dmx2_generator_eval"
RLX = REPO / "eval/results/dmx2_relax_screen"
EGRID = np.linspace(-10, 10, 400)
RNG = np.random.default_rng(0)
NBOOT = 10000


def family(host):
    base = re.sub(r"-[A-Za-z0-9]+$", "", host)
    for suf, fam in (("se2", "Se"), ("te2", "Te"), ("s2", "S"), ("o2", "O")):
        if base.endswith(suf):
            return fam
    raise ValueError(f"no family for host {host}")


def boot_ci(vals, stat=np.mean, n=NBOOT):
    vals = np.asarray(vals, float)
    idx = RNG.integers(0, len(vals), (n, len(vals)))
    s = np.array([stat(vals[i]) for i in idx])
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def boot_ratio_ci(num, den, n=NBOOT):
    """CI of mean(num)/mean(den) under paired structure resampling."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    idx = RNG.integers(0, len(num), (n, len(num)))
    r = num[idx].mean(1) / den[idx].mean(1)
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(REPO / "data/dmx2_audit.csv").set_index("id")
    with open(REPO / "splits/dmx2_v1.json") as f:
        sp = json.load(f)
    split_of = {i: s for s in ("train", "val", "test") for i in sp[s]}
    fam_of = {i: family(i.split("_")[0]) for i in split_of}

    # ---- 1 distribution
    d = pd.DataFrame({"sid": list(split_of), "split": list(split_of.values())})
    d["family"] = d.sid.map(fam_of)
    d["pristine"] = [audit.loc[s, "defect"] == "Defect-Free" for s in d.sid]
    t1a = d.pivot_table(index="family", columns="split", values="sid",
                        aggfunc="count").fillna(0).astype(int)
    t1a["total"] = t1a.sum(1)
    t1a["frac"] = (t1a.total / t1a.total.sum()).round(4)
    t1b = d.pivot_table(index="family", columns="pristine", values="sid",
                        aggfunc="count").fillna(0).astype(int)
    t1b.columns = ["defective", "pristine"]
    t1 = pd.concat([t1a, t1b], axis=1)
    t1.to_csv(OUT / "distribution.csv")

    # ---- 2 forward by family
    fm = pd.read_csv(REPO / "eval/results/dmx2_forward_eval/metrics.csv")
    fm["family"] = fm.host.map(family)
    rows = []
    for fam, sub in fm.groupby("family"):
        for level in ("peratom", "cell"):
            for win in ("full", "allvalid", "nearef"):
                m = sub[f"model_{level}_{win}"].values
                bg = sub[f"base_global_{level}_{win}"].values
                be = sub[f"base_elem_{level}_{win}"].values
                lo, hi = boot_ci(m)
                rlo, rhi = boot_ratio_ci(m, be)
                rows.append(dict(
                    family=fam, n=len(sub), level=level, window=win,
                    model=m.mean(), model_ci_lo=lo, model_ci_hi=hi,
                    base_global=bg.mean(), base_elem=be.mean(),
                    ratio_vs_global=m.mean() / bg.mean(),
                    ratio_vs_elem=m.mean() / be.mean(),
                    ratio_vs_elem_ci_lo=rlo, ratio_vs_elem_ci_hi=rhi))
    t2 = pd.DataFrame(rows)
    t2.to_csv(OUT / "forward_by_family.csv", index=False)

    # ---- 3 generator by family
    def rt(csvpath):
        g = pd.read_csv(csvpath)
        g["family"] = g.structure_id.map(lambda s: family(s.split("_")[0]))
        agg = g.groupby("family").agg(
            n_targets=("structure_id", "nunique"),
            meank_full=("mae_full", "mean"), meank_nearef=("mae_nearef", "mean"))
        best = g.groupby(["family", "structure_id"])[["mae_full", "mae_nearef"]].min()
        agg["bestofk_full"] = best.groupby("family").mae_full.mean()
        agg["bestofk_nearef"] = best.groupby("family").mae_nearef.mean()
        ci = {f: boot_ci(b.mae_full.values)
              for f, b in best.reset_index().groupby("family")}
        agg["bestofk_full_ci_lo"] = [ci[f][0] for f in agg.index]
        agg["bestofk_full_ci_hi"] = [ci[f][1] for f in agg.index]
        return agg
    cond = rt(GEV / "test_k20_w1.csv")
    unc = rt(GEV / "test_k20_uncond.csv")
    resel = pd.read_csv(RLX / "reselect_a.csv")
    resel["family"] = resel.structure_id.map(lambda s: family(s.split("_")[0]))
    sel = resel.groupby("family").post_mae_full.agg(["mean", "count"])
    t3 = cond.join(unc, rsuffix="_uncond")
    t3["uncond_over_cond_bestofk"] = t3.bestofk_full_uncond / t3.bestofk_full
    t3["selected_postrelax_mae"] = sel["mean"]
    t3.to_csv(OUT / "generator_by_family.csv")

    # ---- 4 pristine vs defective within family
    fm["pristine"] = [audit.loc[s, "defect"] == "Defect-Free"
                      for s in fm.structure_id]
    g = pd.read_csv(GEV / "test_k20_w1.csv")
    g["family"] = g.structure_id.map(lambda s: family(s.split("_")[0]))
    g["pristine"] = [audit.loc[s, "defect"] == "Defect-Free"
                     for s in g.structure_id]
    gb = (g.groupby(["family", "pristine", "structure_id"]).mae_full.min()
           .groupby(["family", "pristine"]).agg(["mean", "count"])
           .rename(columns={"mean": "gen_bestof20_full", "count": "n"}))
    fb = (fm.groupby(["family", "pristine"])
            .agg(n=("structure_id", "count"),
                 fwd_peratom_full=("model_peratom_full", "mean"),
                 fwd_peratom_nearef=("model_peratom_nearef", "mean")))
    t4 = fb.join(gb.drop(columns="n"))
    t4.to_csv(OUT / "pristine_vs_defective.csv")

    # ---- 5 recovery by family
    st = pd.read_csv(RLX / "defect_stoich.csv")
    st = st[st.has_reference.fillna(False)]
    st["family"] = st.host.map(family)
    s_agg = st.groupby("family").agg(n_targets=("sid", "nunique"),
                                     stoich_percand=("recovered", "mean"))
    s_agg["stoich_bestof20"] = (st.groupby(["family", "sid"]).recovered.max()
                                  .groupby("family").mean())
    ge = pd.read_csv(RLX / "defect_geometry.csv")
    ge = ge[ge.status != "no_reference_or_cif"].copy()
    ge["family"] = ge.sid.map(lambda s: family(s.split("_")[0]))
    s_agg["d2_n"] = ge.groupby("family").sid.count()
    s_agg["d2_recovered"] = (ge.status == "recovered").groupby(ge.family).mean()
    s_agg.to_csv(OUT / "recovery_by_family.csv")

    # ---- 6 train-set statistics
    with open(REPO / "data/dmx2_dos/train.json") as f:
        train = json.load(f)
    meta = {m["structure_id"]: m for m in
            json.load(open(REPO / "data/dmx2_dos/build_meta.json"))}
    rows, curves = [], {}
    for r in train:
        y = np.asarray(r["y"])
        c = y.mean(0)
        fam = fam_of[r["structure_id"]]
        curves.setdefault(fam, []).append(c)
        grad = np.abs(np.gradient(c, EGRID))
        rows.append(dict(family=fam, ymax_cell=c.max(), ymax_atom=y.max(),
                         sharp_meangrad=grad.mean(),
                         sharp_maxmean=c.max() / max(c.mean(), 1e-9),
                         emax_data=meta[r["structure_id"]]["emax_data"]))
    tr = pd.DataFrame(rows)
    t7 = tr.groupby("family").agg(
        n=("family", "count"), ymax_cell=("ymax_cell", "mean"),
        ymax_atom=("ymax_atom", "mean"),
        sharp_meangrad=("sharp_meangrad", "mean"),
        sharp_maxmean=("sharp_maxmean", "mean"),
        emax_data_mean=("emax_data", "mean"),
        emax_below10_frac=("emax_data", lambda s: (s < 10).mean()))
    div = {}
    for fam, cs in curves.items():
        cs = np.asarray(cs)
        pairs = [np.abs(a - b).mean() for a, b in combinations(cs, 2)]
        div[fam] = (np.mean(pairs), *boot_ci(pairs))
    t7["pairwise_mae"] = [div[f][0] for f in t7.index]
    t7["pairwise_mae_ci_lo"] = [div[f][1] for f in t7.index]
    t7["pairwise_mae_ci_hi"] = [div[f][2] for f in t7.index]
    t7.to_csv(OUT / "train_stats.csv")

    for name, t in [("1 distribution", t1), ("2 forward", t2.round(4)),
                    ("3 generator", t3.round(4)), ("4 pris-vs-def", t4.round(4)),
                    ("5 recovery", s_agg.round(4)),
                    ("6 train stats", t7.round(4))]:
        print(f"\n=== {name} ===")
        print(t.to_string())


if __name__ == "__main__":
    main()
