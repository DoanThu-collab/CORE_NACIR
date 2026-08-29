import numpy as np
import pytest

from nacir.config import NACIRMinusConfig
from nacir.statistics import holm_adjust, paired_bri_delta_ci


def test_frozen_configuration_is_valid() -> None:
    config = NACIRMinusConfig()
    config.validate()

    assert config.memory.negative_weight > 0
    assert config.memory.recency_decay >= 0
    assert config.memory.max_concepts > 0
    assert config.top_k > 0


def test_invalid_negative_weight_is_rejected() -> None:
    config = NACIRMinusConfig()
    config.memory.negative_weight = -1.0

    with pytest.raises(ValueError):
        config.validate()


def test_paired_bri_and_holm_are_deterministic() -> None:
    baseline = np.array(
        [[10, 10, 10], [10, 10, 10]],
        dtype=np.int64,
    )
    candidate = np.array(
        [[0, 0, 0], [0, 0, 0]],
        dtype=np.int64,
    )

    delta, low, high = paired_bri_delta_ci(
        baseline,
        candidate,
        samples=200,
        seed=7,
    )

    assert delta < 0
    assert low <= delta <= high

    adjusted = holm_adjust(
        [0.01, 0.04, 0.03]
    )

    assert adjusted == pytest.approx(
        [0.03, 0.06, 0.06]
    )
