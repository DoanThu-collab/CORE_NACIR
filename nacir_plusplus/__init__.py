"""
NACIR++ Plug-and-Play
========================
Xem README.md ở thư mục gốc để biết hợp đồng Input/Output thống nhất.

Import nhanh:
    from nacir_plusplus import NACIRPlusPlusPipeline, NACIRPlusPlusConfig
    from nacir_plusplus.schema import RetrievalSession, DialogTurn, BeliefBundle, Belief
    from nacir_plusplus.interfaces import TextEncoder, BeliefSource, ImageScorer
"""

from .config import (
    DynamicScheduleConfig,
    NACIRPlusPlusConfig,
    OPTIMAL_CONFIG,
    OPTIMAL_SCHEDULE,
    default_dynamic_schedule,
)
from .pipeline import NACIRPlusPlusPipeline
from .schema import (
    Belief,
    BeliefBundle,
    DialogTurn,
    RetrievalSession,
    SessionOutput,
    TurnOutput,
)

__all__ = [
    "NACIRPlusPlusPipeline",
    "NACIRPlusPlusConfig",
    "DynamicScheduleConfig",
    "OPTIMAL_CONFIG",
    "OPTIMAL_SCHEDULE",
    "default_dynamic_schedule",
    "Belief",
    "BeliefBundle",
    "DialogTurn",
    "RetrievalSession",
    "SessionOutput",
    "TurnOutput",
]
