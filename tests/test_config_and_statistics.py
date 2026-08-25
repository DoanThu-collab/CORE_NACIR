import numpy as np
import pytest

from nacir.config import F1Config
from nacir.statistics import holm_adjust, paired_bri_delta_ci


def test_frozen_configuration_is_valid() -> None:
    config = F1Config()
    config.validate()
    assert config.asymmetric_constraint.mode == "dual_route_trust"
    assert config.asymmetric_constraint.candidate_k == 500
    assert config.asymmetric_constraint.max_kl == 0.002
    assert config.memory.retain_history is True


def test_non_frozen_router_mode_is_rejected() -> None:
    config = F1Config()
    config.asymmetric_constraint.mode = "unconstrained"
    with pytest.raises(ValueError, match="dual_route_trust"):
        config.validate()


def test_paired_bri_and_holm_are_deterministic() -> None:
    baseline = np.array([[10, 10, 10], [10, 10, 10]], dtype=np.int64)
    candidate = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.int64)
    delta, low, high = paired_bri_delta_ci(baseline, candidate, samples=200, seed=7)
    assert delta < 0
    assert low <= delta <= high
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])
