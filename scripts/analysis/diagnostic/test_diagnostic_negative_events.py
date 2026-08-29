import json
import math
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

SEED = 42
N_BOOT = 20000

rows = [
    json.loads(x)
    for x in DATA.read_text(encoding="utf-8").splitlines()
    if x.strip()
]

ranks = {
    k: np.load(v)["ranks"]
    for k, v in RUNS.items()
}

events = defaultdict(list)

for sid, row in enumerate(rows):
    for turn in row["dialogue"]:
        md = turn["metadata"]

        if not md.get("negative_concepts"):
            continue

        feedback_turn = int(turn["turn_idx"])
        state = feedback_turn + 1
        ft = md.get("fact_type", "UNKNOWN")

        events[ft].append((sid, state))


def exact_mcnemar(a_hit, b_hit):
    # paired discordants
    b = int(np.sum(a_hit & ~b_hit))
    c = int(np.sum(~a_hit & b_hit))

    n = b + c
    if n == 0:
        return 1.0, b, c

    k = min(b, c)

    # exact two-sided binomial p-value p=.5
    prob = sum(
        math.comb(n, i) * (0.5 ** n)
        for i in range(k + 1)
    )

    p = min(1.0, 2.0 * prob)
    return p, b, c


def bootstrap_delta(a, b, fn, seed):
    rng = np.random.default_rng(seed)

    obs = fn(b) - fn(a)

    vals = np.empty(N_BOOT, dtype=float)

    n = len(a)

    for i in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        vals[i] = fn(b[idx]) - fn(a[idx])

    lo, hi = np.percentile(vals, [2.5, 97.5])

    return obs, float(lo), float(hi)


def mean_log_rank(x):
    return float(np.mean(np.log1p(x)))


def median_rank(x):
    return float(np.median(x))


for ft in ["ATTRIBUTE", "RELATION", "SCENE"]:

    ev = events[ft]

    sid = np.asarray([x[0] for x in ev])
    state = np.asarray([x[1] for x in ev])

    h0 = ranks["H0"][state, sid]
    cur = ranks["Current"][state, sid]
    per = ranks["Persistent"][state, sid]

    print("\n" + "=" * 78)
    print(ft, "N events =", len(ev))
    print("=" * 78)

    for name, cand in [
        ("Current", cur),
        ("Persistent", per),
    ]:

        h0_hit = h0 < 10
        cand_hit = cand < 10

        delta_r10 = 100.0 * (
            cand_hit.mean() - h0_hit.mean()
        )

        p, h0_only, cand_only = exact_mcnemar(
            h0_hit,
            cand_hit,
        )

        log_obs, log_lo, log_hi = bootstrap_delta(
            h0,
            cand,
            mean_log_rank,
            SEED,
        )

        med_obs, med_lo, med_hi = bootstrap_delta(
            h0,
            cand,
            median_rank,
            SEED + 1,
        )

        print(f"\n{name} vs H0")
        print(
            f"Δ R@10            = {delta_r10:+.3f} pp"
        )
        print(
            f"McNemar discordant = "
            f"H0-only {h0_only}, "
            f"{name}-only {cand_only}"
        )
        print(
            f"McNemar exact p    = {p:.6g}"
        )
        print(
            "Δ mean log(1+rank) = "
            f"{log_obs:+.6f} "
            f"[{log_lo:+.6f}, {log_hi:+.6f}]"
        )
        print(
            "Δ median rank      = "
            f"{med_obs:+.3f} "
            f"[{med_lo:+.3f}, {med_hi:+.3f}]"
        )
