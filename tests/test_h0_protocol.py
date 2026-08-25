import torch

from nacir.config import F1Config
from nacir.evaluation import evaluate_session, rank_matrix
from nacir.pipeline import F1Pipeline
from nacir.schema import DialogTurn, RetrievalSession


class NeverEncode:
    def encode(self, texts):
        raise AssertionError("H0 must not encode beliefs")


def test_h0_ranks_without_beliefs() -> None:
    corpus = torch.eye(10, dtype=torch.float32)
    pipeline = F1Pipeline(
        config=F1Config(),
        corpus_vectors=corpus,
        text_encoder=NeverEncode(),
        device="cpu",
    )
    session = RetrievalSession(
        session_id=7,
        target_index=2,
        turns=[DialogTurn(turn_index=0, query_vector=corpus[2])],
    )
    output = evaluate_session(pipeline, session, "h0")
    assert output.turns[0].final_rank == 0
    assert rank_matrix([output]) == [[0]]
