#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {int(r["sample_id"]): r for r in csv.DictReader(f)}


def norm(x):
    return str(x).strip().upper()


def kappa(a, b):
    keep = [(x, y) for x, y in zip(a, b) if x and y]
    if not keep:
        return None
    a, b = zip(*keep)
    n = len(a)
    po = sum(x == y for x, y in keep) / n
    ca, cb = Counter(a), Counter(b)
    labels = set(ca) | set(cb)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    if abs(1 - pe) < 1e-12:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def majority(vals):
    vals = [v for v in vals if v]
    if not vals:
        return ""
    c = Counter(vals)
    top = c.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return "TIE"
    return top[0][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", type=Path, required=True)
    ap.add_argument("--annotations", type=Path, nargs="+", required=True)
    args = ap.parse_args()

    key = read_csv(args.key)
    anns = [read_csv(p) for p in args.annotations]
    common = sorted(set(key).intersection(*(set(a) for a in anns)))
    if not common:
        raise RuntimeError("No common sample_id values")

    fields = [
        ("human_negation_supported", None),
        ("human_actionable", "auto_actionable"),
        ("human_semantic_type", "auto_semantic_type"),
    ]

    print("=" * 88)
    print("HUMAN VALIDATION")
    print("=" * 88)
    print("annotators:", len(anns), "common items:", len(common))

    for field, auto_field in fields:
        print("\n", field)
        vectors = [[norm(a[i].get(field, "")) for i in common] for a in anns]
        for (ia, va), (ib, vb) in itertools.combinations(enumerate(vectors), 2):
            valid = [(x, y) for x, y in zip(va, vb) if x and y]
            agree = sum(x == y for x, y in valid) / len(valid) if valid else float("nan")
            kap = kappa(va, vb)
            print(f" pair {ia+1}-{ib+1}: agreement={agree:.3f} kappa={kap if kap is not None else 'NA'}")

        if auto_field:
            correct = total = 0
            primary_correct = primary_total = 0
            per_group = defaultdict(lambda: [0, 0])
            for pos, sid in enumerate(common):
                mv = majority([v[pos] for v in vectors])
                auto = norm(key[sid].get(auto_field, ""))
                if field == "human_actionable":
                    auto = "YES" if auto in {"TRUE", "1", "YES"} else "NO"
                if not mv or mv in {"TIE", "UNCLEAR"} or not auto:
                    continue
                total += 1
                correct += int(mv == auto)
                group = key[sid].get("sampling_group", "")
                per_group[group][1] += 1
                per_group[group][0] += int(mv == auto)
                if group == "PRIMARY_RANDOM":
                    primary_total += 1
                    primary_correct += int(mv == auto)
            if total:
                print(f" majority-vs-auto overall: {correct}/{total} = {correct/total:.3f}")
            if primary_total:
                print(f" PRIMARY_RANDOM majority-vs-auto: {primary_correct}/{primary_total} = {primary_correct/primary_total:.3f}")
            for g, (c, n) in sorted(per_group.items()):
                if n:
                    print(f" {g}: {c}/{n} = {c/n:.3f}")

    # Resource precision proxy: majority says the extracted negative is supported.
    vals = []
    for sid in common:
        mv = majority([norm(a[sid].get("human_negation_supported", "")) for a in anns])
        if mv in {"YES", "NO"}:
            vals.append(mv == "YES")
    if vals:
        print("\nNEGATIVE EXTRACTION SUPPORT RATE (majority):", f"{sum(vals)}/{len(vals)} = {sum(vals)/len(vals):.3f}")


if __name__ == "__main__":
    main()
