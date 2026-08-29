import json
import numpy as np
from pathlib import Path

H0 = Path("runs_final/diagnostic_1653_blip_h0")
CUR = Path("runs_final/diagnostic_1653_blip_current_turn")
PER = Path("runs_final/diagnostic_1653_blip_persistent")

EXPECTED_SESSIONS = 1653
EXPECTED_STATES = 11
EXPECTED_TRACES = EXPECTED_SESSIONS * EXPECTED_STATES


def load_ranks(root):
    x = np.load(root / "ranks.npz")
    return x["ranks"]


def load_report(root):
    return json.load(open(root / "report.json"))


def load_traces(root):
    out = {}
    with open(root / "turn_traces.jsonl") as f:
        for line in f:
            x = json.loads(line)
            key = (int(x["session_id"]), int(x["turn_index"]))
            assert key not in out
            out[key] = x
    return out


def neg_count(x):
    return (
        x.get("diagnostics", {})
         .get("memory", {})
         .get("negative", None)
    )


h0r = load_ranks(H0)
curr = load_ranks(CUR)
perr = load_ranks(PER)

h0rep = load_report(H0)
currep = load_report(CUR)
perrep = load_report(PER)

h0 = load_traces(H0)
cur = load_traces(CUR)
per = load_traces(PER)

print("=" * 72)
print("DIAGNOSTIC-1653 BLIP CANONICAL AUDIT")
print("=" * 72)

print("\n[1] SHAPES")
print("H0        :", h0r.shape)
print("Current   :", curr.shape)
print("Persistent:", perr.shape)

assert h0r.shape == (EXPECTED_STATES, EXPECTED_SESSIONS)
assert curr.shape == h0r.shape
assert perr.shape == h0r.shape

assert h0rep["num_sessions"] == EXPECTED_SESSIONS
assert currep["num_sessions"] == EXPECTED_SESSIONS
assert perrep["num_sessions"] == EXPECTED_SESSIONS

print("\n[2] TRACE ALIGNMENT")
print(len(h0), len(cur), len(per))

assert len(h0) == EXPECTED_TRACES
assert len(cur) == EXPECTED_TRACES
assert len(per) == EXPECTED_TRACES
assert set(h0) == set(cur) == set(per)

print("\n[3] ROUND-0 EQUALITY")
assert np.array_equal(h0r[0], curr[0])
assert np.array_equal(h0r[0], perr[0])
print("PASS")

print("\n[4] CURRENT ZERO-NEGATIVE NO-OP")

tested = 0
rank_mm = 0
top10_mm = 0
fullset_mm = 0

for key, c in cur.items():
    if neg_count(c) != 0:
        continue

    tested += 1
    a = h0[key]
    b = c

    if a["final_rank"] != b["final_rank"]:
        rank_mm += 1

    ta = a["top_k_indices"]
    tb = b["top_k_indices"]

    if ta[:10] != tb[:10]:
        top10_mm += 1

    if set(ta) != set(tb):
        fullset_mm += 1

print("tested            :", tested)
print("target-rank mm    :", rank_mm)
print("exact-top10 mm    :", top10_mm)
print("full-topk-set mm  :", fullset_mm)

print("\n[5] PERSISTENT ZERO-NEGATIVE NO-OP")

tested_p = 0
rank_mm_p = 0
top10_mm_p = 0
fullset_mm_p = 0

for key, p in per.items():
    if neg_count(p) != 0:
        continue

    tested_p += 1
    a = h0[key]
    b = p

    if a["final_rank"] != b["final_rank"]:
        rank_mm_p += 1

    ta = a["top_k_indices"]
    tb = b["top_k_indices"]

    if ta[:10] != tb[:10]:
        top10_mm_p += 1

    if set(ta) != set(tb):
        fullset_mm_p += 1

print("tested            :", tested_p)
print("target-rank mm    :", rank_mm_p)
print("exact-top10 mm    :", top10_mm_p)
print("full-topk-set mm  :", fullset_mm_p)

ok = (
    rank_mm == 0
    and top10_mm == 0
    and fullset_mm == 0
    and rank_mm_p == 0
    and top10_mm_p == 0
    and fullset_mm_p == 0
)

print("\n" + "=" * 72)
print("PASS" if ok else "NEEDS MISMATCH CLASSIFICATION")
print("=" * 72)
