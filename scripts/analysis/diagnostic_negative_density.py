import json
import numpy as np
from pathlib import Path
from collections import Counter

ROOT = Path(".")
BELIEFS = Path(
    "artifacts_final/diagnostic_frozen/"
    "beliefs_diagnostic_schema_v2.json"
)

RUNS = {
    "H0": ROOT / "runs_final/diagnostic_1653_blip_h0/ranks.npz",
    "Current": ROOT / "runs_final/diagnostic_1653_blip_current_turn/ranks.npz",
    "Persistent": ROOT / "runs_final/diagnostic_1653_blip_persistent/ranks.npz",
}

b = json.load(open(BELIEFS, encoding="utf-8"))

neg_counts = []

for d in b["dialogs"]:
    c = sum(
        len(t["negatives"])
        for t in d["turns"]
    )
    neg_counts.append(c)

neg_counts = np.asarray(neg_counts)

print("Negative-count distribution:")
print(Counter(neg_counts.tolist()))

ranks = {
    name: np.load(path)["ranks"]
    for name, path in RUNS.items()
}

def r10(x):
    return 100.0 * np.mean(x < 10)

def mean_rank(x):
    return float(np.mean(x))

def summarize(mask, label):
    print("\n" + "=" * 72)
    print(label, "N =", int(mask.sum()))
    print("=" * 72)

    for name, r in ranks.items():
        final = r[-1, mask]
        avg_feedback = r[1:, mask]

        print(
            f"{name:10s} "
            f"Final R@10={r10(final):7.3f} "
            f"Avg-turn R@10={r10(avg_feedback):7.3f} "
            f"Final mean-rank={mean_rank(final):8.3f}"
        )

    h0 = ranks["H0"][-1, mask]
    cur = ranks["Current"][-1, mask]
    per = ranks["Persistent"][-1, mask]

    print(
        f"Δ Current-H0 Final R@10    = "
        f"{r10(cur)-r10(h0):+.3f} pp"
    )
    print(
        f"Δ Persistent-H0 Final R@10 = "
        f"{r10(per)-r10(h0):+.3f} pp"
    )
    print(
        f"Δ Persistent-Current       = "
        f"{r10(per)-r10(cur):+.3f} pp"
    )

summarize(neg_counts == 0, "0 negatives")
summarize(neg_counts == 1, "1 negative")
summarize(neg_counts == 2, "2 negatives")
summarize(neg_counts >= 3, "3+ negatives")
summarize(neg_counts >= 2, "2+ negatives")
summarize(neg_counts > 0, "Any negative")
