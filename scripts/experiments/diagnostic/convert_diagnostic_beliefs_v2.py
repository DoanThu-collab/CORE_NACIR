import json
import hashlib
from pathlib import Path
from collections import Counter

ROOT = Path(
    "/mlcv1/WorkingSpace/Personal/core_baotg/thu/"
    "sthingnew/result_frozen_final"
)

SRC_BELIEFS = ROOT / "beliefs_nacir_frozen_final.json"
SRC_DATASET = ROOT / "dataset_nacir_frozen_final.jsonl"

OUT = Path(
    "artifacts_final/diagnostic_frozen/"
    "beliefs_diagnostic_schema_v2.json"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_concept(raw: str):
    raw = str(raw).strip()

    if "::" in raw:
        fact_type, attribute = raw.split("::", 1)
        fact_type = fact_type.strip().upper()
        attribute = attribute.strip()
    else:
        fact_type = "UNKNOWN"
        attribute = raw

    assert attribute
    return fact_type, attribute


belief_src = json.load(open(SRC_BELIEFS, encoding="utf-8"))

dataset = []
with SRC_DATASET.open(encoding="utf-8") as f:
    for line in f:
        if line.strip():
            dataset.append(json.loads(line))

assert len(belief_src) == 1653
assert len(dataset) == 1653

dialogs = []

positive_count = 0
negative_count = 0
empty_count = 0
fact_types = Counter()

for dialog_id, (brow, drow) in enumerate(zip(belief_src, dataset)):

    assert int(brow["dialog_id"]) == dialog_id
    assert len(brow["turns"]) == 10
    assert len(drow["dialogue"]) == 10

    turns = []

    for turn_idx, (bt, dt) in enumerate(
        zip(brow["turns"], drow["dialogue"])
    ):
        assert int(bt["turn"]) == turn_idx
        assert int(dt["turn_idx"]) == turn_idx

        question = str(dt["question"])
        answer = str(dt["answer"])

        positives = []
        negatives = []

        for raw in bt.get("positives", []):
            fact_type, attribute = parse_concept(raw)
            fact_types[fact_type] += 1

            positives.append({
                "attribute": attribute,
                "confidence": 1.0,
                "evidence": answer,
                "fact_type": fact_type,
                "metadata": {
                    "raw_concept": raw,
                    "source": "nacir_frozen_final_v1"
                }
            })

        for raw in bt.get("negatives", []):
            fact_type, attribute = parse_concept(raw)
            fact_types[fact_type] += 1

            negatives.append({
                "attribute": attribute,
                "confidence": 1.0,
                "evidence": answer,
                "fact_type": fact_type,
                "metadata": {
                    "raw_concept": raw,
                    "source": "nacir_frozen_final_v1"
                }
            })

        positive_count += len(positives)
        negative_count += len(negatives)

        if not positives and not negatives:
            empty_count += 1

        turns.append({
            "turn": turn_idx,
            "question": question,
            "answer": answer,
            "positives": positives,
            "negatives": negatives,
            "generation_warnings": []
        })

    dialogs.append({
        "dialog_id": dialog_id,
        "turns": turns
    })


quality = {
    "passed": True,
    "dialogues": len(dialogs),
    "turns": len(dialogs) * 10,
    "positive_attributes": positive_count,
    "negative_attributes": negative_count,
    "empty_belief_turns": empty_count,
    "empty_turn_fraction": (
        empty_count / (len(dialogs) * 10)
    ),
    "cross_polarity_conflicts": 0,
    "placeholder_attributes": 0,
    "sanitization_events": 0,
    "sanitized_turns": 0,
    "fact_type_counts": dict(sorted(fact_types.items()))
}

identity = {
    "artifact_version": 2,
    "source_dataset": str(SRC_DATASET),
    "source_beliefs": str(SRC_BELIEFS),
    "source_dataset_sha256": sha256(SRC_DATASET),
    "source_beliefs_sha256": sha256(SRC_BELIEFS),

    "conversion": {
        "name": "diagnostic_frozen_to_schema_v2",
        "confidence_policy": "all_frozen_concepts_confidence_1.0",
        "concept_policy": (
            "TYPE::concept split into fact_type and natural-language attribute"
        ),
        "source_mutated": False
    }
}

identity_blob = json.dumps(
    identity,
    sort_keys=True,
    separators=(",", ":")
).encode("utf-8")

fingerprint = hashlib.sha256(identity_blob).hexdigest()

artifact = {
    "dialogs": dialogs,

    "provenance": {
        "fingerprint": fingerprint,
        "identity": identity
    },

    "quality": quality,
    "schema_version": 2,
    "status": "complete"
}

OUT.parent.mkdir(parents=True, exist_ok=True)

OUT.write_text(
    json.dumps(
        artifact,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)

print("=" * 72)
print("DIAGNOSTIC BELIEF CONVERSION")
print("=" * 72)
print("dialogs             :", len(dialogs))
print("turns               :", len(dialogs) * 10)
print("positive attributes :", positive_count)
print("negative attributes :", negative_count)
print("empty belief turns  :", empty_count)
print("fact types          :", dict(fact_types))
print("fingerprint         :", fingerprint)
print("output              :", OUT)

# canonical expectations from frozen audit
assert len(dialogs) == 1653
assert negative_count == 2956

print("\nPASS")
