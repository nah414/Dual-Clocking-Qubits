"""Dual clocking simulator package with hardware-specific backends."""

from .scheduler import (
    DualClockConfig,
    DualClockSchedule,
    DualClockScheduler,
    PulseOp,
    TeleportationConfig,
)
from .simulate import (
    SimulatorConfig,
    SimResult,
    run_schedule,
    require_sqi_edge_fraction,
    decode_majority,
)
from .calibration import (
    ProbePoint,
    ProbeData,
    CalibrationMetrics,
    ProbeCalibrator,
    mutual_info_bits,
    qnd_correlation,
    compute_snr,
    compute_separation,
    compute_metrics,
    pareto_front,
    analyze_probe_sweep,
)
from .backends.base import DualClockingBackend, ProbeResult
from .backends.superconducting import SuperconductingBackend
from .backends.ion import TrappedIonBackend
from .backends.nmr import NMRBackend
from .backends.telecom import TelecomPhotonicsBackend
from .backends.plugins import (
    BUILTIN_BACKENDS,
    ENTRY_POINT_GROUP as BACKEND_ENTRY_POINT_GROUP,
    available_backends,
    get_backend_class,
)

__all__ = [
    "DualClockConfig",
    "DualClockSchedule",
    "DualClockScheduler",
    "PulseOp",
    "TeleportationConfig",
    "SimulatorConfig",
    "SimResult",
    "run_schedule",
    "require_sqi_edge_fraction",
    "decode_majority",
    "ProbePoint",
    "ProbeData",
    "CalibrationMetrics",
    "ProbeCalibrator",
    "mutual_info_bits",
    "qnd_correlation",
    "compute_snr",
    "compute_separation",
    "compute_metrics",
    "pareto_front",
    "analyze_probe_sweep",
    "DualClockingBackend",
    "ProbeResult",
    "SuperconductingBackend",
    "TrappedIonBackend",
    "NMRBackend",
    "TelecomPhotonicsBackend",
    "BUILTIN_BACKENDS",
    "BACKEND_ENTRY_POINT_GROUP",
    "available_backends",
    "get_backend_class",
]
