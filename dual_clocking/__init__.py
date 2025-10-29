"""Dual clocking simulator package with hardware-specific backends."""

from .scheduler import (
    DualClockConfig,
    DualClockSchedule,
    DualClockScheduler,
    PulseOp,
)
from .simulate import (
    SimulatorConfig,
    SimResult,
    run_schedule,
    require_sqi_edge_fraction,
    decode_majority,
)
from .backends.base import DualClockingBackend, ProbeResult
from .backends.superconducting import SuperconductingBackend
from .backends.ion import TrappedIonBackend
from .backends.nmr import NMRBackend

__all__ = [
    "DualClockConfig",
    "DualClockSchedule",
    "DualClockScheduler",
    "PulseOp",
    "SimulatorConfig",
    "SimResult",
    "run_schedule",
    "require_sqi_edge_fraction",
    "decode_majority",
    "DualClockingBackend",
    "ProbeResult",
    "SuperconductingBackend",
    "TrappedIonBackend",
    "NMRBackend",
]
