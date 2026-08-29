import torch

from nacir.config import NACIRMinusConfig
from nacir.evaluation import evaluate_session
from nacir.pipeline import NACIRMinusPipeline
from nacir.schema import DialogTurn, RetrievalSession


class NeverEncode:
    def encode(self, texts):
        raise AssertionError(
            "H0 must not encode beliefs"
        )


def test_h0_ranks_without_beliefs() -> None:
    corpus = torch.eye(
        10,
        dtype=torch.float32,
    )

    pipeline = NACIRMinusPipeline(
        config=NACIRMinusConfig(
            top_k=10,
        ),
        corpus_vectors=corpus,
        text_encoder=NeverEncode(),
        device="cpu",
    )

    session = RetrievalSession(
        session_id=7,
        target_index=2,
        turns=[
            DialogTurn(
                turn_index=0,
                query_text="",
                query_vector=corpus[2],
            )
        ],
    )

    output = evaluate_session(
        pipeline,
        session,
        "h0",
    )

    assert output.turns[0].final_rank == 0
    assert output.turns[0].top_k_indices[0] == 2
