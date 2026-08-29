import sys
import types

import numpy as np

from nacir.provenance import _declared_model_revision, verify_pairing


def _obj(fp, sids=(0, 1), targets=(4, 5)):
    return {
        "ranks": np.zeros((11, 2), dtype=np.int64),
        "pairing_fingerprint": fp,
        "evaluation_fingerprint": "run-" + fp,
        "session_ids": np.array(sids, dtype=np.int64),
        "target_indices": np.array(targets, dtype=np.int64),
    }


def test_same_pairing_verifies():
    assert verify_pairing(_obj("same"), _obj("same"))["verified"]


def test_cross_space_fingerprint_rejected():
    v = verify_pairing(_obj("blip"), _obj("clip"))
    assert not v["verified"]
    assert "pairing_fingerprint_mismatch" in v["reasons"]


def test_target_mismatch_rejected():
    v = verify_pairing(_obj("same"), _obj("same", targets=(4, 6)))
    assert not v["verified"]
    assert "target_indices_mismatch" in v["reasons"]


def test_missing_metadata_rejected():
    a = _obj("same")
    b = _obj("same")
    b["pairing_fingerprint"] = None
    assert not verify_pairing(a, b)["verified"]


def test_declared_revision_is_loaded_for_nacir(monkeypatch):
    module = types.ModuleType("fake_nacir_adapter")
    module.MODEL_REVISION = "revision-123"
    monkeypatch.setitem(sys.modules, "fake_nacir_adapter", module)

    assert (
        _declared_model_revision("persistent", "fake_nacir_adapter")
        == "revision-123"
    )


def test_h0_does_not_import_unused_adapter():
    assert _declared_model_revision("h0", "module.that.does.not.exist") is None
