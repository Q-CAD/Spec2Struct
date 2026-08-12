"""Oxide ablation eval: oxide-4x-oversampled + leave-oxide-out, vs the
pre-registered criterion recorded in eval/results/RESULTS.md (section 7):

  success (Arm 1) = oxide forward per-atom full MAE mean drops below 0.154
  (outside the baseline model's 95% CI) while S/Se/Te stay inside their
  baseline CIs; also check the model/per-elem ratio CI moves off 1.0.

Same frozen split, same eval pipeline: model predictions via
dmx2_forward_eval.predict_split (run's own prop_scaler), per-structure
baselines REUSED from the committed forward_eval/metrics.csv (train-mean
baselines don't depend on the model), bootstrap CIs identical to
dmx2_family_benchmark (seeded rng, 10k resamples).

Usage: python eval/dmx2_ablation_eval.py --run_dir outputs/<...> --ckpt <...>
       --tag ox4|loo
Writes eval/results/dmx2_family_benchmark/ablation/forward_<tag>_by_family.csv
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))
from dmx2_family_benchmark import family, boot_ci, boot_ratio_ci
from dmx2_forward_eval import predict_split, EGRID

OUT = REPO / "eval/results/dmx2_family_benchmark/ablation"
NEAREF = (EGRID >= -2) & (EGRID <= 2)


def main(args):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(REPO / "data/dmx2_dos/test.json") as f:
        test = json.load(f)
    base = pd.read_csv(REPO / "eval/results/dmx2_forward_eval/metrics.csv")
    preds = predict_split(Path(args.run_dir), args.ckpt)

    rows = []
    for r in test:
        sid = r["structure_id"]
        err = np.abs(preds[sid] - np.asarray(r["y"]))
        rows.append(dict(structure_id=sid,
                         family=family(sid.split("_")[0]),
                         model_peratom_full=float(err.mean()),
                         model_peratom_nearef=float(err[:, NEAREF].mean()),
                         model_cell_full=float(np.abs(
                             preds[sid].mean(0) - np.asarray(r["y"]).mean(0)).mean())))
    df = pd.DataFrame(rows).merge(
        base[["structure_id", "base_global_peratom_full",
              "base_elem_peratom_full"]], on="structure_id")
    df.to_csv(OUT / f"forward_{args.tag}_metrics.csv", index=False)

    fam_rows = []
    for fam, sub in df.groupby("family"):
        m = sub.model_peratom_full.values
        be = sub.base_elem_peratom_full.values
        lo, hi = boot_ci(m)
        rlo, rhi = boot_ratio_ci(m, be)
        fam_rows.append(dict(
            family=fam, n=len(sub), model_peratom_full=m.mean(),
            ci_lo=lo, ci_hi=hi,
            model_peratom_nearef=sub.model_peratom_nearef.mean(),
            model_cell_full=sub.model_cell_full.mean(),
            base_elem=be.mean(), ratio_vs_elem=m.mean() / be.mean(),
            ratio_ci_lo=rlo, ratio_ci_hi=rhi))
    out = pd.DataFrame(fam_rows)
    out.to_csv(OUT / f"forward_{args.tag}_by_family.csv", index=False)
    print(f"[{args.tag}] ckpt {args.ckpt}")
    print(out.round(4).to_string(index=False))

    # verdict vs the pre-registered criterion (baseline-model CIs)
    ref = pd.read_csv(REPO / "eval/results/dmx2_family_benchmark/forward_by_family.csv")
    ref = ref[(ref.level == "peratom") & (ref.window == "full")].set_index("family")
    print("\nvs pre-registered criterion (baseline model CIs):")
    for fam in ("O", "S", "Se", "Te"):
        cur = out[out.family == fam].iloc[0]
        r = ref.loc[fam]
        if fam == "O":
            print(f"  O: {cur.model_peratom_full:.4f} "
                  f"{'<' if cur.model_peratom_full < 0.154 else '>='} 0.154 "
                  f"(baseline 0.2109, CI [{r.model_ci_lo:.3f}, {r.model_ci_hi:.3f}])")
        else:
            inside = r.model_ci_lo <= cur.model_peratom_full <= r.model_ci_hi
            print(f"  {fam}: {cur.model_peratom_full:.4f} "
                  f"{'inside' if inside else 'OUTSIDE'} baseline CI "
                  f"[{r.model_ci_lo:.3f}, {r.model_ci_hi:.3f}]")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tag", required=True, choices=["ox4", "loo"])
    main(p.parse_args())
