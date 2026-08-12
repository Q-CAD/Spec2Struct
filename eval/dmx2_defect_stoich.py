"""D1: defect-stoichiometry recovery for all conditional test candidates.

For each generated candidate, compute the element-count delta vs the PRISTINE
supercell composition of the target's host (the host's Defect-Free entry in the
audit table). The ground-truth target defines the reference delta the same way.
Recovery = generated delta EXACTLY equals the GT delta (species + counts).
Stricter than comp_exact_match: a composition that matches some other host, or
the right elements in wrong proportions, cannot pass.

Output: eval/results/dmx2_relax_screen/defect_stoich.csv + printed by-class rates.
"""
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from make_dmx2_split import coarse_class

from pymatgen.core.periodic_table import Element


def formula_to_counter(formula):
    return Counter({el: int(n) for el, n in re.findall(r"([A-Z][a-z]?)(\d+)", formula)})


def znums_to_counter(zs):
    return Counter(Element.from_Z(int(z)).symbol for z in zs)


def delta(comp, pristine):
    d = {}
    for el in set(comp) | set(pristine):
        v = comp.get(el, 0) - pristine.get(el, 0)
        if v:
            d[el] = v
    return tuple(sorted(d.items()))


def main():
    audit = pd.read_csv(REPO / "data/dmx2_audit.csv")
    pristine = {r.host: formula_to_counter(r.formula)
                for r in audit.itertuples() if r.defect == "Defect-Free"}
    hosts_all = set(audit.host)
    print(f"pristine references: {len(pristine)}/{len(hosts_all)} hosts "
          f"(missing: {sorted(hosts_all - set(pristine))})")

    gt_formula = audit.set_index("id").formula.to_dict()
    gt_defect = audit.set_index("id").defect.to_dict()

    blob = torch.load(REPO / "eval/preds/dmx2_test_k20_w1.pt", weights_only=False)
    rows = []
    for k, cands in enumerate(blob["preds"]):
        for c in cands:
            sid = c["structure_id"]
            host = sid.split("_")[0]
            if host not in pristine:
                rows.append(dict(sid=sid, k=k, host=host,
                                 coarse_class=coarse_class(gt_defect[sid]),
                                 has_reference=False, recovered=np.nan,
                                 gen_delta="", gt_delta=""))
                continue
            d_gen = delta(znums_to_counter(c["atom_types"]), pristine[host])
            d_gt = delta(formula_to_counter(gt_formula[sid]), pristine[host])
            rows.append(dict(sid=sid, k=k, host=host,
                             coarse_class=coarse_class(gt_defect[sid]),
                             has_reference=True, recovered=(d_gen == d_gt),
                             gen_delta=str(d_gen), gt_delta=str(d_gt)))
    df = pd.DataFrame(rows)
    out = REPO / "eval/results/dmx2_relax_screen/defect_stoich.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    ok = df[df.has_reference]
    print(f"\ncandidates with pristine reference: {len(ok)}/{len(df)}")
    print(f"DEFECT STOICHIOMETRY RECOVERY (delta_gen == delta_gt): {ok.recovered.mean():.4f}")
    print("\nby coarse class:")
    print(ok.groupby("coarse_class").recovered.agg(["mean", "count"]).round(4).to_string())
    per = ok.groupby("sid").recovered.max()
    print(f"\nbest-of-20 recovery (any candidate correct): {per.mean():.4f}")


if __name__ == "__main__":
    main()
