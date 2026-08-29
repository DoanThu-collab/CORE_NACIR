import ast
import json
import hashlib
from pathlib import Path

import torch
from tqdm import tqdm

from nacir.adapters.plugir_blip import (
    load_blip_text_encoder,
)


ROOT = Path(__file__).resolve().parents[3]

FROZEN = Path(
    "/mlcv1/WorkingSpace/Personal/core_baotg/thu/"
    "sthingnew/result_frozen_final"
)

QUERIES = (
    FROZEN
    / "NACIR_queries_frozen_final.json"
)

GALLERY = (
    ROOT
    / "artifacts_final/diagnostic_frozen/"
      "diagnostic_gallery_1653.json"
)

OUT = (
    ROOT
    / "artifacts_final/diagnostic_frozen/"
      "sessions_diagnostic_1653_blip.pt"
)

PROV = OUT.with_suffix(".provenance.json")

DEVICE = "cuda"


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_context(raw):
    """
    Frozen query context is serialized like:
      "{'type': 'scene_summary',
        'text': 'A person ...'}"

    Use only actual scene-summary text.
    """

    raw = str(raw).strip()

    try:
        parsed = ast.literal_eval(raw)

        if (
            isinstance(parsed, dict)
            and isinstance(parsed.get("text"), str)
        ):
            return parsed["text"].strip()

    except Exception:
        pass

    return raw


queries = json.load(
    open(
        QUERIES,
        encoding="utf-8",
    )
)

gallery = json.load(
    open(
        GALLERY,
        encoding="utf-8",
    )
)

assert len(queries) == 1653
assert len(gallery) == 1653

encoder = load_blip_text_encoder(
    DEVICE,
    allow_download=False,
)

sessions = []

for dialog_id, row in enumerate(
    tqdm(
        queries,
        desc="Building diagnostic BLIP sessions",
    )
):

    dialog = row["dialog"]

    # context + ten QA pairs
    assert len(dialog) == 11

    context = normalize_context(dialog[0])

    assert context

    states = [context]

    history = [context]

    for qa in dialog[1:]:

        qa = str(qa).strip()
        assert qa

        history.append(qa)

        # Explicit cumulative conversational history.
        states.append(
            " ".join(history)
        )

    assert len(states) == 11

    vectors = encoder.encode(states)
    vectors = vectors.detach().cpu().float()

    assert vectors.shape == (11, 256)
    assert torch.isfinite(vectors).all()

    # Closed-world frozen mapping:
    # dialog i -> gallery image i.
    target_index = dialog_id

    expected_rel = gallery[
        target_index
    ]["relative_path"]

    assert row["img"] == expected_rel, (
        dialog_id,
        row["img"],
        expected_rel,
    )

    sessions.append({
        "session_id": dialog_id,
        "target_index": target_index,
        "query_vectors": vectors,
        "query_texts": states,
    })


assert len(sessions) == 1653

for i, s in enumerate(sessions):
    assert s["session_id"] == i
    assert s["target_index"] == i
    assert s["query_vectors"].shape == (11, 256)
    assert len(s["query_texts"]) == 11


OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

torch.save(
    sessions,
    OUT,
)

prov = {
    "status": "complete",
    "benchmark": "NACIR-Diagnostic-1653",
    "sessions": 1653,
    "retrieval_states_per_session": 11,
    "feedback_turns": 10,
    "embedding_dim": 256,

    "query_protocol": (
        "state0=scene_summary; "
        "state_t=scene_summary plus all QA pairs "
        "through feedback turn t-1"
    ),

    "target_protocol": (
        "dialog_i -> frozen gallery_index_i"
    ),

    "queries": str(QUERIES),
    "queries_sha256": sha256(QUERIES),

    "gallery": str(GALLERY),
    "gallery_sha256": sha256(GALLERY),

    "output": str(OUT),
}

PROV.write_text(
    json.dumps(
        prov,
        indent=2,
    ),
    encoding="utf-8",
)

print("\n[PASS]")
print("sessions:", len(sessions))
print(
    "first shape:",
    tuple(sessions[0]["query_vectors"].shape),
)
print(
    "first target:",
    sessions[0]["target_index"],
)
print("\nSTATE 0:")
print(sessions[0]["query_texts"][0])

print("\nSTATE 1:")
print(sessions[0]["query_texts"][1])

print("\nSTATE 10 chars:")
print(len(sessions[0]["query_texts"][10]))

print("\nsaved:", OUT)
print("provenance:", PROV)
