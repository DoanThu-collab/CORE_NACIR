from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from nacir.config import MemoryConfig, NACIRMinusConfig
from nacir.evaluation import evaluate_session, rank_matrix
from nacir.pipeline import NACIRMinusPipeline
from nacir.pipeline_current_turn import NACIRCurrentTurnPipeline
from nacir.provenance import build_provenance, read_rank_archive, verify_pairing
from nacir.schema import Belief, BeliefBundle, DialogTurn, RetrievalSession


class _FakeEncoder:
    def encode(self, texts):
        rows = []
        for text in texts:
            if text == "blue":
                rows.append(torch.tensor([0.0, 1.0]))
            else:
                rows.append(torch.tensor([1.0, 0.0]))
        return torch.stack(rows)


def _corpus() -> torch.Tensor:
    rows = [
        [1.0, -0.30],  # target: helped by subtracting the negative blue direction
        [1.0, 0.00],
        [0.8, 0.4],
        [0.6, 0.8],
        [0.0, 1.0],
        [-0.4, 0.9],
        [-0.8, 0.6],
        [-1.0, 0.0],
        [-0.8, -0.6],
        [-0.4, -0.9],
        [0.0, -1.0],
        [0.5, -0.8],
    ]
    return F.normalize(torch.tensor(rows, dtype=torch.float32), dim=-1)


def _session() -> RetrievalSession:
    query = torch.tensor([1.0, 0.0])
    negative = BeliefBundle(
        negative=[Belief(attribute="blue", confidence=1.0, fact_type="negative")],
        source_turn=0,
    )
    return RetrievalSession(
        session_id=0,
        target_index=0,
        turns=[
            DialogTurn(0, "initial", query.clone(), BeliefBundle.empty()),
            DialogTurn(1, "negative introduced", query.clone(), negative),
            DialogTurn(2, "no current negative", query.clone(), BeliefBundle.empty()),
        ],
    )


def _serialized_session(session: RetrievalSession) -> list[dict]:
    return [
        {
            "session_id": int(session.session_id),
            "target_index": int(session.target_index),
            "query_vectors": torch.stack([turn.query_vector for turn in session.turns]),
            "query_texts": [turn.query_text for turn in session.turns],
        }
    ]


def _write_fixture_archive(path: Path, ranks: np.ndarray, provenance: dict) -> None:
    """Write the documented public ranks.npz schema without importing CLI code."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        key: value
        for key, value in provenance.items()
        if key not in {"session_ids", "target_indices"}
    }
    metadata["provenance_status"] = "emitted_by_evaluator"
    np.savez_compressed(
        path,
        ranks=np.asarray(ranks, dtype=np.int64),
        session_ids=np.asarray(provenance["session_ids"], dtype=np.int64),
        target_indices=np.asarray(provenance["target_indices"], dtype=np.int64),
        pairing_fingerprint=np.asarray(provenance["pairing_fingerprint"]),
        evaluation_fingerprint=np.asarray(provenance["evaluation_fingerprint"]),
        provenance_status=np.asarray("emitted_by_evaluator"),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def test_h0_current_persistent_history_only_and_provenance(tmp_path: Path):
    corpus = _corpus()
    session = _session()
    encoder = _FakeEncoder()
    config = NACIRMinusConfig(
        memory=MemoryConfig(
            negative_weight=0.275,
            recency_decay=0.10,
            max_concepts=50,
            semantic_merge=False,
        ),
        top_k=12,
    )

    persistent_pipeline = NACIRMinusPipeline(
        config=config,
        corpus_vectors=corpus,
        text_encoder=encoder,
        device="cpu",
    )
    current_pipeline = NACIRCurrentTurnPipeline(
        config=config,
        corpus_vectors=corpus,
        text_encoder=encoder,
        device="cpu",
    )

    h0 = evaluate_session(persistent_pipeline, session, "h0")
    current = evaluate_session(current_pipeline, session, "nacir")
    persistent = evaluate_session(persistent_pipeline, session, "nacir")

    h0_ranks = np.asarray(rank_matrix([h0]), dtype=np.int64)
    current_ranks = np.asarray(rank_matrix([current]), dtype=np.int64)
    persistent_ranks = np.asarray(rank_matrix([persistent]), dtype=np.int64)

    # Turn 0 is pre-feedback. Turn 1 sees the same current negative in both adapted
    # methods. Turn 2 has no current negative, so Current collapses exactly to H0,
    # while Persistent can still use the historical exclusion.
    assert h0_ranks[0, 0] == current_ranks[0, 0] == persistent_ranks[0, 0]
    assert current_ranks[1, 0] == persistent_ranks[1, 0]
    assert current_ranks[2, 0] == h0_ranks[2, 0]
    assert persistent_ranks[2, 0] < h0_ranks[2, 0]

    # Release traces are deliberately capped at the R@10 boundary even when the
    # configured ranking top-k is larger.
    assert all(len(turn.top_k_indices) == 10 for turn in h0.turns)
    assert all(len(turn.top_k_indices) == 10 for turn in current.turns)
    assert all(len(turn.top_k_indices) == 10 for turn in persistent.turns)

    sessions_path = tmp_path / "sessions.pt"
    corpus_path = tmp_path / "corpus.pt"
    corpus_alt_path = tmp_path / "corpus_alt.pt"
    beliefs_path = tmp_path / "beliefs.json"
    config_path = tmp_path / "config.json"

    torch.save(_serialized_session(session), sessions_path)
    torch.save(corpus, corpus_path)
    corpus_alt = corpus.clone()
    corpus_alt[3] = F.normalize(torch.tensor([0.7, 0.7]), dim=0)
    torch.save(corpus_alt, corpus_alt_path)
    beliefs_path.write_text('{"fixture": true}', encoding="utf-8")
    config_path.write_text(
        json.dumps(
            {
                "memory": {
                    "negative_weight": 0.275,
                    "recency_decay": 0.10,
                    "max_concepts": 50,
                    "semantic_merge": False,
                },
                "top_k": 12,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    h0_provenance = build_provenance(
        sessions_path=sessions_path,
        corpus_path=corpus_path,
        beliefs_path=beliefs_path,
        method="h0",
        config_path=config_path,
    )
    persistent_provenance = build_provenance(
        sessions_path=sessions_path,
        corpus_path=corpus_path,
        beliefs_path=beliefs_path,
        method="persistent",
        config_path=config_path,
        adapter_module="fixture.fake_adapter",
        adapter_func="load",
        model_revision="fake-v1",
    )
    alt_provenance = build_provenance(
        sessions_path=sessions_path,
        corpus_path=corpus_alt_path,
        beliefs_path=beliefs_path,
        method="h0",
        config_path=config_path,
    )

    h0_path = tmp_path / "h0" / "ranks.npz"
    persistent_path = tmp_path / "persistent" / "ranks.npz"
    alt_path = tmp_path / "alt" / "ranks.npz"
    _write_fixture_archive(h0_path, h0_ranks, h0_provenance)
    _write_fixture_archive(persistent_path, persistent_ranks, persistent_provenance)
    _write_fixture_archive(alt_path, h0_ranks, alt_provenance)

    h0_archive = read_rank_archive(h0_path)
    persistent_archive = read_rank_archive(persistent_path)
    alt_archive = read_rank_archive(alt_path)

    assert verify_pairing(h0_archive, persistent_archive)["verified"] is True
    mismatch = verify_pairing(h0_archive, alt_archive)
    assert mismatch["verified"] is False
    assert "pairing_fingerprint_mismatch" in mismatch["reasons"]

    with np.load(persistent_path, allow_pickle=False) as archive:
        assert {
            "ranks",
            "session_ids",
            "target_indices",
            "pairing_fingerprint",
            "evaluation_fingerprint",
            "provenance_status",
            "metadata_json",
        }.issubset(archive.files)
        metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
        assert metadata["model_revision"] == "fake-v1"
        assert str(np.asarray(archive["provenance_status"]).item()) == "emitted_by_evaluator"
