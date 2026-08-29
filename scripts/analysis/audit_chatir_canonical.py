import json
import argparse
from pathlib import Path

import numpy as np


def load_report(run_dir):
    p = Path(run_dir) / "report.json"
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_ranks(run_dir):
    p = Path(run_dir) / "ranks.npz"
    x = np.load(p, allow_pickle=False)
    if "ranks" not in x.files:
        raise ValueError(f"{p}: missing 'ranks'")
    return x["ranks"]


def load_traces(run_dir):
    p = Path(run_dir) / "turn_traces.jsonl"
    traces = {}

    with p.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)

            sid = int(item["session_id"])
            turn = int(item["turn_index"])
            key = (sid, turn)

            if key in traces:
                raise ValueError(
                    f"{p}:{lineno}: duplicate trace key {key}"
                )

            traces[key] = item

    return traces


def memory_negative(trace):
    """
    Return active negative-memory count if available.
    """

    diag = trace.get("diagnostics", {})
    mem = diag.get("memory", {})

    if "negative" not in mem:
        return None

    return int(mem["negative"])


def audit_triplet(name, h0_dir, current_dir, persistent_dir):
    print()
    print("=" * 78)
    print(f"AUDITING {name}")
    print("=" * 78)

    runs = {
        "H0": h0_dir,
        "Current": current_dir,
        "Persistent": persistent_dir,
    }

    reports = {
        k: load_report(v)
        for k, v in runs.items()
    }

    ranks = {
        k: load_ranks(v)
        for k, v in runs.items()
    }

    traces = {
        k: load_traces(v)
        for k, v in runs.items()
    }

    # ------------------------------------------------------------
    # 1. Basic report / shape checks
    # ------------------------------------------------------------

    print("\n[1] REPORT / SHAPE AUDIT")

    for k in runs:
        r = reports[k]
        a = ranks[k]

        print(
            f"{k:10s} "
            f"method={r.get('method')} "
            f"config={r.get('config_path')} "
            f"shape={a.shape} "
            f"BRI={r['metrics']['bri']:.8f}"
        )

        assert r["status"] == "complete"
        assert r["num_sessions"] == 2064
        assert r["metrics"]["num_queries"] == 2064
        assert r["metrics"]["num_rounds"] == 11
        assert a.shape == (11, 2064)

    assert ranks["H0"].shape == ranks["Current"].shape
    assert ranks["H0"].shape == ranks["Persistent"].shape

    # ------------------------------------------------------------
    # 2. Trace key alignment
    # ------------------------------------------------------------

    print("\n[2] TRACE ALIGNMENT")

    h0_keys = set(traces["H0"])
    cur_keys = set(traces["Current"])
    per_keys = set(traces["Persistent"])

    print("H0 traces        :", len(h0_keys))
    print("Current traces   :", len(cur_keys))
    print("Persistent traces:", len(per_keys))

    assert h0_keys == cur_keys, "H0/Current trace keys mismatch"
    assert h0_keys == per_keys, "H0/Persistent trace keys mismatch"

    expected = 2064 * 11

    assert len(h0_keys) == expected, (
        f"Expected {expected} traces, got {len(h0_keys)}"
    )

    # ------------------------------------------------------------
    # 3. Turn-0 equality
    # ------------------------------------------------------------

    print("\n[3] ROUND-0 EQUALITY")

    assert np.array_equal(
        ranks["H0"][0],
        ranks["Current"][0]
    ), "Current differs from H0 at round 0"

    assert np.array_equal(
        ranks["H0"][0],
        ranks["Persistent"][0]
    ), "Persistent differs from H0 at round 0"

    print("PASS: H0 == Current == Persistent at round 0")

    # ------------------------------------------------------------
    # 4. Current-turn no-op invariant
    #
    # If active negative count for this turn is zero:
    # Current rank MUST equal H0 rank.
    # ------------------------------------------------------------

    print("\n[4] CURRENT-TURN ZERO-NEGATIVE NO-OP")

    current_tested = 0
    current_mismatch = []

    for (sid, turn), trace in traces["Current"].items():
        neg = memory_negative(trace)

        if neg is None:
            continue

        if neg == 0:
            current_tested += 1

            h0_rank = int(ranks["H0"][turn, sid])
            cur_rank = int(ranks["Current"][turn, sid])

            if h0_rank != cur_rank:
                current_mismatch.append(
                    {
                        "session_id": sid,
                        "turn": turn,
                        "h0_rank": h0_rank,
                        "current_rank": cur_rank,
                    }
                )

    print("zero-negative turn states tested:", current_tested)
    print("mismatches:", len(current_mismatch))

    if current_mismatch:
        print("First mismatches:")
        for x in current_mismatch[:20]:
            print(x)

    # ------------------------------------------------------------
    # 5. Persistent no-op invariant
    #
    # If persistent memory contains zero negative concepts:
    # Persistent rank MUST equal H0.
    # ------------------------------------------------------------

    print("\n[5] PERSISTENT ZERO-NEGATIVE NO-OP")

    persistent_tested = 0
    persistent_mismatch = []

    for (sid, turn), trace in traces["Persistent"].items():
        neg = memory_negative(trace)

        if neg is None:
            continue

        if neg == 0:
            persistent_tested += 1

            h0_rank = int(ranks["H0"][turn, sid])
            per_rank = int(ranks["Persistent"][turn, sid])

            if h0_rank != per_rank:
                persistent_mismatch.append(
                    {
                        "session_id": sid,
                        "turn": turn,
                        "h0_rank": h0_rank,
                        "persistent_rank": per_rank,
                    }
                )

    print("zero-negative persistent states tested:", persistent_tested)
    print("mismatches:", len(persistent_mismatch))

    if persistent_mismatch:
        print("First mismatches:")
        for x in persistent_mismatch[:20]:
            print(x)

    # ------------------------------------------------------------
    # 6. Exact top-k invariant on no-op states
    #
    # Stronger than target-rank equality.
    # ------------------------------------------------------------

    print("\n[6] TOP-K NO-OP INVARIANT")

    current_topk_tested = 0
    current_topk_mismatch = 0

    persistent_topk_tested = 0
    persistent_topk_mismatch = 0

    for key in h0_keys:
        h0_trace = traces["H0"][key]

        cur_trace = traces["Current"][key]
        cur_neg = memory_negative(cur_trace)

        if cur_neg == 0:
            current_topk_tested += 1

            if (
                h0_trace.get("top_k_indices")
                != cur_trace.get("top_k_indices")
            ):
                current_topk_mismatch += 1

        per_trace = traces["Persistent"][key]
        per_neg = memory_negative(per_trace)

        if per_neg == 0:
            persistent_topk_tested += 1

            if (
                h0_trace.get("top_k_indices")
                != per_trace.get("top_k_indices")
            ):
                persistent_topk_mismatch += 1

    print(
        "Current top-k:",
        current_topk_tested,
        "tested /",
        current_topk_mismatch,
        "mismatches"
    )

    print(
        "Persistent top-k:",
        persistent_topk_tested,
        "tested /",
        persistent_topk_mismatch,
        "mismatches"
    )

    # ------------------------------------------------------------
    # 7. Final verdict
    # ------------------------------------------------------------

    pass_current = (
        len(current_mismatch) == 0
        and current_topk_mismatch == 0
    )

    pass_persistent = (
        len(persistent_mismatch) == 0
        and persistent_topk_mismatch == 0
    )

    print()
    print("=" * 78)

    if pass_current and pass_persistent:
        print(f"{name}: CANONICAL NO-OP AUDIT PASS")
    else:
        print(f"{name}: CANONICAL NO-OP AUDIT FAIL")

    print("=" * 78)

    return pass_current and pass_persistent


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--h0", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--persistent", required=True)
    parser.add_argument("--name", default="ChatIR")

    args = parser.parse_args()

    ok = audit_triplet(
        args.name,
        args.h0,
        args.current,
        args.persistent,
    )

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
