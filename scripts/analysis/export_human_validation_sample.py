#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


FIELDS = [
    "sample_id", "sampling_group", "dialog_id", "turn", "negative_index",
    "question", "answer", "negative_attribute", "evidence",
    "human_negation_supported", "human_actionable",
    "human_semantic_type", "human_notes",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structured", type=Path, default=Path(
        "artifacts_final/typed_nacir/chatir_structured_negative_final_v1_1.json"
    ))
    ap.add_argument("--random-n", type=int, default=200)
    ap.add_argument("--per-type", type=int, default=25)
    ap.add_argument("--non-actionable-n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts_final/human_validation"))
    args = ap.parse_args()

    artifact = json.loads(args.structured.read_text(encoding="utf-8"))
    items = artifact["items"]
    rng = random.Random(args.seed)

    # Primary unbiased sample for overall annotation-quality estimates.
    primary = rng.sample(items, min(args.random_n, len(items)))

    # Diagnostic enrichment: balanced semantic types + rare non-actionable cases.
    by_type = defaultdict(list)
    non_actionable = []
    for x in items:
        if not x.get("actionable_negative", True):
            non_actionable.append(x)
        typ = (x.get("typing") or {}).get("type")
        if x.get("actionable_negative", True) and typ:
            by_type[typ].append(x)

    extra = []
    for typ in ["EXISTENCE", "ATTRIBUTE", "RELATION", "GLOBAL"]:
        pool = by_type.get(typ, [])
        extra.extend(rng.sample(pool, min(args.per_type, len(pool))))
    extra.extend(rng.sample(non_actionable, min(args.non_actionable_n, len(non_actionable))))

    selected = []
    seen = set()
    for group, seq in [("PRIMARY_RANDOM", primary), ("STRATIFIED_EXTRA", extra)]:
        for x in seq:
            key = (int(x["dialog_id"]), int(x["turn"]), int(x.get("negative_index", 0)))
            if key in seen:
                continue
            seen.add(key)
            selected.append((group, x))

    rng.shuffle(selected)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    blind = args.out_dir / "human_validation_blind.csv"
    keyfile = args.out_dir / "human_validation_key.csv"
    guide = args.out_dir / "ANNOTATION_GUIDE.txt"

    with blind.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for i, (group, x) in enumerate(selected):
            w.writerow({
                "sample_id": i,
                "sampling_group": group,
                "dialog_id": x["dialog_id"],
                "turn": x["turn"],
                "negative_index": x.get("negative_index", 0),
                "question": x.get("question", ""),
                "answer": x.get("answer", ""),
                "negative_attribute": x.get("negative_attribute", ""),
                "evidence": x.get("evidence", ""),
                "human_negation_supported": "",
                "human_actionable": "",
                "human_semantic_type": "",
                "human_notes": "",
            })

    with keyfile.open("w", newline="", encoding="utf-8") as f:
        fields = ["sample_id", "sampling_group", "auto_actionable", "auto_semantic_type"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, (group, x) in enumerate(selected):
            w.writerow({
                "sample_id": i,
                "sampling_group": group,
                "auto_actionable": bool(x.get("actionable_negative", True)),
                "auto_semantic_type": (x.get("typing") or {}).get("type") or "",
            })

    guide.write_text(
        """NACIR human-validation guide

Annotate independently. Do NOT inspect human_validation_key.csv before finishing.

1) human_negation_supported: YES / NO / UNCLEAR
Does the question-answer pair provide evidence that the exact extracted
negative concept should be rejected for the target image?
Judge the exact concept, not a corrected fact introduced only by the answer.

2) human_actionable: YES / NO / UNCLEAR
Is that negative evidence usable to exclude images in retrieval?
Pure inability to see/tell/know is normally non-actionable when it does not
establish that the candidate concept is false.

3) human_semantic_type:
EXISTENCE / ATTRIBUTE / RELATION / GLOBAL / UNCLEAR

EXISTENCE: presence/absence of an entity/object.
ATTRIBUTE: property of an entity (color, state, material, etc.).
RELATION: relation between entities.
GLOBAL: scene-level/global proposition not naturally reducible to one entity.

Each annotator should fill a separate copy of the blind CSV.
""",
        encoding="utf-8",
    )

    print("Total selected:", len(selected))
    print("Blind sheet:", blind)
    print("Hidden key:", keyfile)
    print("Guide:", guide)
    print("IMPORTANT: duplicate blind CSV for each annotator; keep key hidden until annotation is complete.")


if __name__ == "__main__":
    main()
