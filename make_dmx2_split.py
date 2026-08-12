"""Build the frozen host-stratified split for the raw DOS dataset.

Inputs : data/dmx2_audit.csv, data/dmx2_exclude.json
Output : splits/dmx2_v1.json  (explicit id lists; consumed by build_dmx_dos_json.py)

Design (deterministic, SEED=2024, same seed convention as build_dmx_dos_json.py):
  - unit = structure_id, stratum = host (audit 'host' column)
  - global 80/10/10 on the 620 post-exclusion ids -> 496/62/62
  - every host with n >= 3 gets >= 1 val and >= 1 test id
  - host with n == 2 (rete2): both -> train (explicit decision)
  - remaining val/test slots go to largest hosts by largest remainder (0.1*n - 1),
    ties broken by host name
  - light balancing pass: every fine defect type with global count >= 10 must
    appear in val and in test; fixed by within-host train<->val/test swaps
"""
import csv
import json
import random
from collections import Counter, defaultdict

SEED = 2024
AUDIT = "data/dmx2_audit.csv"
EXCLUDE = "data/dmx2_exclude.json"
OUT = "splits/dmx2_v1.json"
FRAC_VAL = FRAC_TEST = 0.10
MIN_TYPE_COUNT = 10  # balancing threshold for fine defect types


def coarse_class(defect):
    if defect == "Defect-Free":
        return "pristine"
    if defect.startswith("Vacancy"):
        return "vacancy"
    if defect.startswith("Anti"):
        return "antisite"
    if defect.endswith("_doped") or defect.startswith("Doped"):
        return "substitution"
    if defect.endswith("_adatom") or defect.startswith("Adatom"):
        return "adatom"
    if defect.startswith("Int"):
        return "interstitial"
    return "other"  # V48_* and other odd labels — semantics TBD with mentor


def main():
    rows = list(csv.DictReader(open(AUDIT)))
    excl = set(json.load(open(EXCLUDE))["exclude_all"])
    rows = [r for r in rows if r["id"] not in excl]
    n = len(rows)
    defect_of = {r["id"]: r["defect"] for r in rows}
    by_host = defaultdict(list)
    for r in rows:
        by_host[r["host"]].append(r["id"])
    for h in by_host:
        by_host[h].sort()

    n_val = n_test = round(FRAC_VAL * n)
    print(f"post-exclusion n={n}; targets train/val/test = "
          f"{n - n_val - n_test}/{n_val}/{n_test}; hosts={len(by_host)}")

    rng = random.Random(SEED)
    assign = {}  # id -> split

    # pass 1: guaranteed 1 val + 1 test per host with n >= 3
    small_train_only = []
    for h in sorted(by_host):
        ids = by_host[h][:]
        rng.shuffle(ids)
        if len(ids) < 3:
            small_train_only.append((h, ids))
            for i in ids:
                assign[i] = "train"
            continue
        assign[ids[0]] = "val"
        assign[ids[1]] = "test"
        for i in ids[2:]:
            assign[i] = "train"
    if small_train_only:
        print("hosts with n<3, all -> train:",
              [(h, len(i)) for h, i in small_train_only])

    # pass 2: distribute remaining val/test slots by largest remainder
    used_val = Counter(assign.values())["val"]
    used_test = Counter(assign.values())["test"]
    rem = sorted(
        ((0.10 * len(by_host[h]) - 1.0, h) for h in by_host if len(by_host[h]) >= 3),
        key=lambda t: (-t[0], t[1]),
    )

    def promote(split, k):
        got = 0
        for _, h in rem:
            if got == k:
                break
            cands = [i for i in by_host[h] if assign[i] == "train"]
            if not cands:
                continue
            pick = rng.choice(sorted(cands))
            assign[pick] = split
            got += 1
        if got != k:
            raise RuntimeError(f"could not place {k} extra {split} ids")

    promote("val", n_val - used_val)
    promote("test", n_test - used_test)

    # pass 3: defect-type coverage balancing (within-host swaps only)
    def split_types(s):
        return Counter(defect_of[i] for i, sp in assign.items() if sp == s)

    global_types = Counter(defect_of.values())
    big_types = {t for t, c in global_types.items() if c >= MIN_TYPE_COUNT}
    swaps = []
    for s in ("val", "test"):
        for t in sorted(big_types - set(split_types(s))):
            done = False
            for i in sorted(i for i, d in defect_of.items()
                            if d == t and assign[i] == "train"):
                h = i.split("_")[0]
                # partner: same-host member of s whose type stays covered in s
                for j in sorted(k for k, sp in assign.items()
                                if sp == s and k.split("_")[0] == h):
                    if split_types(s)[defect_of[j]] >= 2 or defect_of[j] not in big_types:
                        assign[i], assign[j] = s, "train"
                        swaps.append((t, s, i, j))
                        done = True
                        break
                if done:
                    break
            if not done:
                print(f"WARN: type {t!r} ({global_types[t]}x) uncovered in {s}, "
                      f"no legal same-host swap found")
    for t, s, i, j in swaps:
        print(f"swap for {t!r} in {s}: {i} <- {s}, {j} -> train")

    # verification
    splits = {s: sorted(i for i, sp in assign.items() if sp == s)
              for s in ("train", "val", "test")}
    sizes = {s: len(v) for s, v in splits.items()}
    assert sum(sizes.values()) == n
    assert not (set(splits["train"]) & set(splits["val"]))
    assert not (set(splits["train"]) & set(splits["test"]))
    assert not (set(splits["val"]) & set(splits["test"]))
    print("sizes:", sizes)

    hosts_in = {s: {i.split("_")[0] for i in splits[s]} for s in splits}
    ge3 = {h for h in by_host if len(by_host[h]) >= 3}
    for s in ("val", "test"):
        missing = ge3 - hosts_in[s]
        print(f"hosts(n>=3) missing from {s}: {sorted(missing) if missing else 'none'}")

    print("\ncoarse-class marginals (frac of split):")
    for s in splits:
        cc = Counter(coarse_class(defect_of[i]) for i in splits[s])
        tot = sizes[s]
        print(f"  {s:5s} " + "  ".join(f"{k}:{v/tot:.2f}" for k, v in sorted(cc.items())))
    print("\nbig fine types missing anywhere:",
          [(t, s) for s in ("val", "test") for t in sorted(big_types)
           if t not in {defect_of[i] for i in splits[s]}] or "none")

    out = {
        "version": "dmx2_v1",
        "seed": SEED,
        "source_audit": AUDIT,
        "excluded": sorted(excl),
        "sizes": sizes,
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
    }
    import os
    os.makedirs("splits", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
