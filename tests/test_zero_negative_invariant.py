import torch

from nacir.config import MemoryConfig
from nacir.core.memory import ConceptMemory
from nacir.schema import Belief, BeliefBundle


class FixedEncoder:
    def encode(self, texts):
        vectors = {
            "red car": torch.tensor([1.0, 0.0, 0.0]),
        }
        return torch.stack([vectors[text] for text in texts])


def test_empty_negative_bundle_is_noop_up_to_normalization():
    memory = ConceptMemory(MemoryConfig(), FixedEncoder())
    query = torch.tensor([3.0, 4.0, 0.0])

    stats = memory.add_bundle(BeliefBundle.empty(), turn=1)
    result = memory.synthesize(query)

    assert stats == {
        "added": 0,
        "updated": 0,
        "overridden": 0,
        "evicted": 0,
    }
    assert len(memory.entries) == 0
    assert torch.allclose(result, torch.nn.functional.normalize(query, dim=0))


def test_positive_only_bundle_does_not_change_canonical_memory():
    memory = ConceptMemory(MemoryConfig(), FixedEncoder())
    query = torch.tensor([3.0, 4.0, 0.0])
    bundle = BeliefBundle(positive=[Belief("red car", confidence=0.9)])

    stats = memory.add_bundle(bundle, turn=1)
    result = memory.synthesize(query)

    assert stats == {
        "added": 0,
        "updated": 0,
        "overridden": 0,
        "evicted": 0,
    }
    assert len(memory.entries) == 0
    assert torch.allclose(result, torch.nn.functional.normalize(query, dim=0))


def test_negative_bundle_changes_query_direction():
    config = MemoryConfig(negative_weight=0.275)
    memory = ConceptMemory(config, FixedEncoder())
    query = torch.tensor([0.0, 1.0, 0.0])
    bundle = BeliefBundle(negative=[Belief("red car", confidence=1.0)])

    memory.add_bundle(bundle, turn=1)
    result = memory.synthesize(query)

    assert len(memory.entries) == 1
    assert not torch.allclose(result, torch.nn.functional.normalize(query, dim=0))
    assert result[0] < 0
