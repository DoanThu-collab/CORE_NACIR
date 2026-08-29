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

SEED = 42
N_BOOT = 20000

rows = [
    json.loads(line)
    for line in DATA.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

ranks = {
    name: np.load(path)["ranks"]
    for name, path in RUNS.items()
}

# type -> dialog_id -> retrieval states after negatives of that type
events = defaultdict(lambda: defaultdict(list))

for sid, row in enumerate(rows):
    for turn in row["dialogue"]:
        md = turn["metadata"]

        if not md.get("negative_concepts"):
            continue

        ft = md.get("fact_type", "UNKNOWN")
        state = int(turn["turn_idx"]) + 1

        events[ft][sid].append(state)


def collect_for_dialogs(ft, sampled_dialogs, candidate):
    """
    Cluster bootstrap:
    resample dialogues with replacement, then retain all events belonging
    to each sampled dialogue. Therefore dependence within a dialogue is
    preserved.
    """
    h0_values = []
    cand_values = []

    mapping = events[ft]

    for sid in sampled_dialogs:
        states = mapping[sid]

        for state in states:
            h0_values.append(ranks["H0"][state, sid])
            cand_values.append(ranks[candidate][state, sid])

    return (
        np.asarray(h0_values),
        np.asarray(cand_values),
    )


def delta_r10(h0, cand):
    return 100.0 * (
        np.mean(cand < 10) -
        np.mean(h0 < 10)
    )


def delta_logrank(h0, cand):
    return float(
        np.mean(np.log1p(cand)) -
        np.mean(np.log1p(h0))
    )


def delta_meanrank(h0, cand):
    return float(
        np.mean(cand) -
        np.mean(h0)
    )


def cluster_bootstrap(ft, candidate):
    dialog_ids = np.asarray(
        sorted(events[ft].keys()),
        dtype=int,
    )

    # observed effect
    h0_obs, cand_obs = collect_for_dialogs(
        ft,
        dialog_ids,
        candidate,
    )

    obs_r10 = delta_r10(h0_obs, cand_obs)
    obs_log = delta_logrank(h0_obs, cand_obs)
    obs_mr = delta_meanrank(h0_obs, cand_obs)

    rng = np.random.default_rng(SEED)

    boot_r10 = np.empty(N_BOOT)
    boot_log = np.empty(N_BOOT)
    boot_mr = np.empty(N_BOOT)

    n_dialogs = len(dialog_ids)

    for b in range(N_BOOT):
        sampled = rng.choice(
            dialog_ids,
            size=n_dialogs,
            replace=True,
        )

        h0, cand = collect_for_dialogs(
            ft,
            sampled,
            candidate,
        )

        boot_r10[b] = delta_r10(h0, cand)
        boot_log[b] = delta_logrank(h0, cand)
        boot_mr[b] = delta_meanrank(h0, cand)

    r10_ci = np.percentile(boot_r10, [2.5, 97.5])
    log_ci = np.percentile(boot_log, [2.5, 97.5])
    mr_ci = np.percentile(boot_mr, [2.5, 97.5])

    # Two-sided bootstrap sign p-value.
    def sign_p(x):
        p_lo = np.mean(x <= 0)
        p_hi = np.mean(x >= 0)
        return min(1.0, 2 * min(p_lo, p_hi))

    return {
        "dialogs": n_dialogs,
        "events": len(h0_obs),

        "r10": (
            obs_r10,
            r10_ci[0],
            r10_ci[1],
            sign_p(boot_r10),
        ),

        "logrank": (
            obs_log,
            log_ci[0],
            log_ci[1],
            sign_p(boot_log),
        ),

        "meanrank": (
            obs_mr,
            mr_ci[0],
            mr_ci[1],
            sign_p(boot_mr),
        ),
    }


for ft in ["ATTRIBUTE", "RELATION", "SCENE"]:

    print("\n" + "=" * 80)
    print(ft)
    print("=" * 80)

    for candidate in ["Current", "Persistent"]:

        result = cluster_bootstrap(ft, candidate)

        print(
            f"\n{candidate} vs H0"
            f" | dialogs={result['dialogs']}"
            f" events={result['events']}"
        )

        obs, lo, hi, p = result["r10"]
        print(
            f"ΔR@10 = {obs:+.3f} pp "
            f"[{lo:+.3f}, {hi:+.3f}] "
            f"bootstrap-p={p:.5f}"
        )

        obs, lo, hi, p = result["logrank"]
        print(
            f"Δ mean log(1+rank) = {obs:+.6f} "
            f"[{lo:+.6f}, {hi:+.6f}] "
            f"bootstrap-p={p:.5f}"
        )

        obs, lo, hi, p = result["meanrank"]
        print(
            f"Δ mean rank = {obs:+.3f} "
            f"[{lo:+.3f}, {hi:+.3f}] "
            f"bootstrap-p={p:.5f}"
        )
