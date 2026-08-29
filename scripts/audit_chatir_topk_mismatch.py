import json
from pathlib import Path
from collections import Counter

H0 = Path("runs_final/chatir_clip_vitl14_h0/turn_traces.jsonl")
CUR = Path("runs_final/chatir_clip_vitl14_nacir_current_turn/turn_traces.jsonl")
PER = Path("runs_final/chatir_clip_vitl14_nacir_minus/turn_traces.jsonl")


def load(path):
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            x = json.loads(line)
            out[(int(x["session_id"]), int(x["turn_index"]))] = x
    return out


def neg_count(x):
    return (
        x.get("diagnostics", {})
         .get("memory", {})
         .get("negative", None)
    )


def analyze(name, h0, cand):
    mismatches = []

    for key, c in cand.items():
        if neg_count(c) != 0:
            continue

        a = h0[key]["top_k_indices"]
        b = c["top_k_indices"]

        if a == b:
            continue

        sa, sb = set(a), set(b)

        # How different are the sets?
        removed = sa - sb
        added = sb - sa

        # Compare important prefixes
        top10_a = a[:10]
        top10_b = b[:10]

        top100_a = a[:100]
        top100_b = b[:100]

        mismatches.append({
            "session_id": key[0],
            "turn": key[1],

            "same_set": sa == sb,

            "set_symmetric_diff": len(sa ^ sb),

            "top10_exact": top10_a == top10_b,
            "top10_same_set": set(top10_a) == set(top10_b),

            "top100_exact": top100_a == top100_b,
            "top100_same_set": set(top100_a) == set(top100_b),

            "removed_count": len(removed),
            "added_count": len(added),

            "h0_final_rank": h0[key]["final_rank"],
            "candidate_final_rank": c["final_rank"],
        })

    print("\n" + "=" * 72)
    print(name)
    print("=" * 72)
    print("mismatches:", len(mismatches))

    if not mismatches:
        return

    same_full_set = sum(x["same_set"] for x in mismatches)
    same_top10_set = sum(x["top10_same_set"] for x in mismatches)
    exact_top10 = sum(x["top10_exact"] for x in mismatches)

    same_top100_set = sum(x["top100_same_set"] for x in mismatches)
    exact_top100 = sum(x["top100_exact"] for x in mismatches)

    rank_mismatch = sum(
        x["h0_final_rank"] != x["candidate_final_rank"]
        for x in mismatches
    )

    print(f"same full top-k set : {same_full_set}/{len(mismatches)}")
    print(f"exact top-10        : {exact_top10}/{len(mismatches)}")
    print(f"same top-10 set     : {same_top10_set}/{len(mismatches)}")
    print(f"exact top-100       : {exact_top100}/{len(mismatches)}")
    print(f"same top-100 set    : {same_top100_set}/{len(mismatches)}")
    print(f"target-rank mismatch: {rank_mismatch}/{len(mismatches)}")

    diffs = Counter(x["set_symmetric_diff"] for x in mismatches)

    print("\nFull-set symmetric difference distribution:")
    for k in sorted(diffs):
        print(f"  diff={k}: {diffs[k]}")

    print("\nFirst 20 mismatches:")
    for x in mismatches[:20]:
        print(x)


def main():
    h0 = load(H0)
    cur = load(CUR)
    per = load(PER)

    assert set(h0) == set(cur) == set(per)

    analyze("CURRENT vs H0", h0, cur)
    analyze("PERSISTENT vs H0", h0, per)


if __name__ == "__main__":
    main()
