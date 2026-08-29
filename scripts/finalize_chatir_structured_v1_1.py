import json
import hashlib
from pathlib import Path
from collections import Counter

SRC = Path(
    "artifacts_final/typed_nacir/"
    "chatir_structured_negative_final_v1.json"
)

OUT = Path(
    "artifacts_final/typed_nacir/"
    "chatir_structured_negative_final_v1_1.json"
)

MANIFEST = Path(
    "artifacts_final/typed_nacir/"
    "chatir_structured_negative_final_v1_1_manifest.json"
)

PARENT_SHA = (
    "2d438c7a9d5f402d5a8b850483118c2cbebfe17cb909df2267d2032bc510e776"
)

TYPE_FIX = {
    "288:3:0": "ATTRIBUTE",
    "304:3:0": "RELATION",
    "871:3:0": "ATTRIBUTE",
    "931:6:0": "ATTRIBUTE",
    "1221:7:1": "GLOBAL",
    "1461:2:0": "ATTRIBUTE",
    "1849:5:0": "ATTRIBUTE",
}

DROP_FIX = {
    "439:7:0",
    "1433:1:0",
}

assert len(TYPE_FIX) == 7
assert len(DROP_FIX) == 2


def key(x):
    return (
        f"{x['dialog_id']}:"
        f"{x['turn']}:"
        f"{x.get('negative_index', 0)}"
    )


artifact = json.load(open(SRC, encoding="utf-8"))
items = artifact["items"]

assert len(items) == 6464

seen_type = set()
seen_drop = set()

for x in items:
    k = key(x)

    if k in TYPE_FIX:
        assert x["actionable_negative"] is True
        assert isinstance(x.get("typing"), dict)
        assert x["typing"].get("type") is None

        x["typing"]["type"] = TYPE_FIX[k]

        # Make status internally consistent with eligibility.
        x["typing"]["status"] = "REJECTED"

        x["semantic_type_source"] = (
            "manual_pre_retrieval_final_adjudication"
        )

        seen_type.add(k)

    if k in DROP_FIX:
        assert x["actionable_negative"] is True

        x["actionable_negative"] = False
        x["eligibility_source"] = (
            "manual_pre_retrieval_final_null_type_audit_drop"
        )

        # Preserve semantic uncertainty.
        if isinstance(x.get("typing"), dict):
            x["typing"]["status"] = "UNRESOLVED"
            x["typing"]["type"] = None

        seen_drop.add(k)


assert seen_type == set(TYPE_FIX)
assert seen_drop == DROP_FIX

# ------------------------------------------------------------
# Final consistency audits
# ------------------------------------------------------------

actionable = [
    x for x in items
    if x["actionable_negative"]
]

dropped = [
    x for x in items
    if not x["actionable_negative"]
]

actionable_none = [
    x for x in actionable
    if not isinstance(x.get("typing"), dict)
    or x["typing"].get("type") is None
]

assert len(actionable_none) == 0, (
    f"Still have {len(actionable_none)} actionable type=None"
)

allowed_types = {
    "EXISTENCE",
    "ATTRIBUTE",
    "RELATION",
    "GLOBAL",
}

for x in actionable:
    assert x["typing"]["type"] in allowed_types

type_counts = Counter(
    x["typing"]["type"]
    for x in actionable
)

drop_type_counts = Counter(
    x["typing"].get("type")
    if isinstance(x.get("typing"), dict)
    else None
    for x in dropped
)

source_counts = Counter(
    x["eligibility_source"]
    for x in items
)

artifact["items"] = items
artifact["status"] = (
    "FROZEN_PRE_RETRIEVAL_STRUCTURED_NEGATIVE_V1_1"
)

artifact["final_null_type_audit"] = {
    "num_actionable_type_none_before": 9,
    "typed_actionable": 7,
    "newly_dropped_non_actionable": 2,
    "retrieval_results_used": False,
    "reason": (
        "Final pre-retrieval audit of actionable items lacking "
        "semantic type. Seven contained actionable propositions "
        "and were manually typed; two were found to lack evidence "
        "against the exact candidate concept and were dropped."
    ),
}

OUT.write_text(
    json.dumps(
        artifact,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

sha = hashlib.sha256(
    OUT.read_bytes()
).hexdigest()

manifest = {
    "artifact": str(OUT),
    "sha256": sha,
    "parent_artifact": str(SRC),
    "parent_sha256": PARENT_SHA,
    "num_negatives": len(items),
    "num_actionable": len(actionable),
    "num_non_actionable": len(dropped),
    "actionable_type_counts": dict(type_counts),
    "actionable_type_none": len(actionable_none),
    "final_manual_type_repairs": 7,
    "final_manual_eligibility_drops": 2,
    "retrieval_results_used_for_design_or_adjudication": False,
    "status": "FROZEN_BEFORE_STRUCTURED_RETRIEVAL",
}

MANIFEST.write_text(
    json.dumps(
        manifest,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print("=" * 72)
print("FINAL STRUCTURED NEGATIVE V1.1")
print("=" * 72)

print("total             :", len(items))
print("actionable        :", len(actionable))
print("non_actionable    :", len(dropped))
print("actionable types  :", dict(type_counts))
print("dropped types     :", dict(drop_type_counts))
print("actionable None   :", len(actionable_none))
print("eligibility source:", dict(source_counts))
print("sha256            :", sha)
print("saved             :", OUT)
print("manifest          :", MANIFEST)

print("\nPASS")
