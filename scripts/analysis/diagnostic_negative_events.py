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

events = defaultdict(list)

for dialog_id, row in enumerate(rows):
    for t in row["dialogue"]:
        md = t["metadata"]

        if not md.get("negative_concepts"):
            continue

        feedback_turn = int(t["turn_idx"])

        # Belief from feedback turn t affects retrieval state t+1.
        retrieval_state = feedback_turn + 1

        ft = md.get("fact_type", "UNKNOWN")

        events[ft].append(
            (dialog_id, feedback_turn, retrieval_state)
        )


def summarize(label, ev):
    if not ev:
        return

    dh0 = []
    dcur = []
    dper = []

    cur_vs_h0 = []
    per_vs_h0 = []
    per_vs_cur = []

    h0_hit = []
    cur_hit = []
    per_hit = []

    for sid, ft, state in ev:
        # Change inside each method from the state before feedback
        # to the state after feedback.
        prev_state = state - 1

        h0_before = ranks["H0"][prev_state, sid]
        h0_after = ranks["H0"][state, sid]

        cur_before = ranks["Current"][prev_state, sid]
        cur_after = ranks["Current"][state, sid]

        per_before = ranks["Persistent"][prev_state, sid]
        per_after = ranks["Persistent"][state, sid]

        dh0.append(h0_after - h0_before)
        dcur.append(cur_after - cur_before)
        dper.append(per_after - per_before)

        cur_vs_h0.append(cur_after - h0_after)
        per_vs_h0.append(per_after - h0_after)
        per_vs_cur.append(per_after - cur_after)

        h0_hit.append(h0_after < 10)
        cur_hit.append(cur_after < 10)
        per_hit.append(per_after < 10)

    dh0 = np.asarray(dh0)
    dcur = np.asarray(dcur)
    dper = np.asarray(dper)

    cur_vs_h0 = np.asarray(cur_vs_h0)
    per_vs_h0 = np.asarray(per_vs_h0)
    per_vs_cur = np.asarray(per_vs_cur)

    print("\n" + "=" * 78)
    print(label, "events =", len(ev))
    print("=" * 78)

    print("Mean within-method rank change after negative")
    print("  H0        :", float(dh0.mean()))
    print("  Current   :", float(dcur.mean()))
    print("  Persistent:", float(dper.mean()))
    print("  (negative = target rank improved)")

    print("\nRank difference AT post-negative state")
    print("  Current - H0       :", float(cur_vs_h0.mean()))
    print("  Persistent - H0    :", float(per_vs_h0.mean()))
    print("  Persistent - Current:", float(per_vs_cur.mean()))
    print("  (negative = first method is better)")

    print("\nPost-negative R@10")
    print("  H0        :", 100*np.mean(h0_hit))
    print("  Current   :", 100*np.mean(cur_hit))
    print("  Persistent:", 100*np.mean(per_hit))

    print("\nWin / tie / loss in target rank vs H0")

    for name, x in [
        ("Current", cur_vs_h0),
        ("Persistent", per_vs_h0),
    ]:
        wins = int((x < 0).sum())
        ties = int((x == 0).sum())
        losses = int((x > 0).sum())

        print(
            f"  {name:10s}: "
            f"win={wins} tie={ties} loss={losses}"
        )


all_events = []

for ft in sorted(events):
    summarize(ft, events[ft])
    all_events.extend(events[ft])

summarize("ALL NEGATIVE EVENTS", all_events)


# Position buckets.
position = {
    "EARLY_0_2": [],
    "MID_3_5": [],
    "LATE_6_9": [],
}

for ev in all_events:
    _, feedback_turn, _ = ev

    if feedback_turn <= 2:
        position["EARLY_0_2"].append(ev)
    elif feedback_turn <= 5:
        position["MID_3_5"].append(ev)
    else:
        position["LATE_6_9"].append(ev)

for name, ev in position.items():
    summarize(name, ev)
