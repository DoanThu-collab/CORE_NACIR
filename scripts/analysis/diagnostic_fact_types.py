import json
import numpy as np
from pathlib import Path
from collections import defaultdict

DATA = Path(
    "/mlcv1/WorkingSpace/Personal/core_baotg/thu/sthingnew/"
    "result_frozen_final/dataset_nacir_frozen_final.jsonl"
)

RUNS = {
    "H0": Path("runs_final/diagnostic_1653_blip_h0/ranks.npz"),
    "Current": Path("runs_final/diagnostic_1653_blip_current_turn/ranks.npz"),
    "Persistent": Path("runs_final/diagnostic_1653_blip_persistent/ranks.npz"),
}

rows = [
    json.loads(x)
    for x in DATA.read_text(encoding="utf-8").splitlines()
    if x.strip()
]

ranks = {
    k: np.load(v)["ranks"]
    for k, v in RUNS.items()
}

def r10(x):
    return 100.0 * np.mean(x < 10)

def mr(x):
    return float(np.mean(x))

types_per_dialog = defaultdict(set)
neg_turns_per_dialog = defaultdict(list)

for i, row in enumerate(rows):
    for t in row["dialogue"]:
        md = t["metadata"]
        negs = md.get("negative_concepts", [])
        if not negs:
            continue

        ft = md.get("fact_type", "UNKNOWN")
        types_per_dialog[i].add(ft)
        neg_turns_per_dialog[i].append(int(t["turn_idx"]))

def summarize(mask, label):
    n = int(mask.sum())
    if n == 0:
        return

    print("\n" + "=" * 72)
    print(label, "N =", n)
    print("=" * 72)

    for name, r in ranks.items():
        final = r[-1, mask]
        avg = r[1:, mask]

        print(
            f"{name:10s} "
            f"Final R@10={r10(final):7.3f} "
            f"Avg-turn R@10={r10(avg):7.3f} "
            f"Final MR={mr(final):8.3f}"
        )

    h0 = ranks["H0"][-1, mask]
    cur = ranks["Current"][-1, mask]
    per = ranks["Persistent"][-1, mask]

    print(f"Δ Current-H0    = {r10(cur)-r10(h0):+.3f} pp")
    print(f"Δ Persistent-H0 = {r10(per)-r10(h0):+.3f} pp")
    print(f"Δ Persist-Curr  = {r10(per)-r10(cur):+.3f} pp")

N = len(rows)

for ft in ["CATEGORY", "ATTRIBUTE", "RELATION", "SCENE"]:
    mask = np.array([
        ft in types_per_dialog[i]
        for i in range(N)
    ])
    summarize(mask, f"Has negative {ft}")

# exclusive single-type dialogues
for ft in ["CATEGORY", "ATTRIBUTE", "RELATION", "SCENE"]:
    mask = np.array([
        types_per_dialog[i] == {ft}
        for i in range(N)
    ])
    summarize(mask, f"ONLY negative {ft}")

# first-negative position
first_neg = np.array([
    min(neg_turns_per_dialog[i])
    if neg_turns_per_dialog[i]
    else -1
    for i in range(N)
])

summarize((first_neg >= 0) & (first_neg <= 2), "First negative EARLY (turn 0-2)")
summarize((first_neg >= 3) & (first_neg <= 5), "First negative MID (turn 3-5)")
summarize(first_neg >= 6, "First negative LATE (turn 6-9)")
