"""Typed pulse operations and schedule builder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .backends.base import DualClockingBackend


@dataclass
class PulseOp:
    """Single pulse entry in the schedule."""

    kind: str
    duration: float
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class DualClockConfig:
    """Timing and control parameters shared across hardware backends."""

    t_drive_1: float = 40e-9
    t_probe: float = 20e-9
    t_drive_2: float = 40e-9
    rabi_rate: float = 2.0 * 3.141592653589793 * 25e6
    probe_threshold: float = 0.0
    two_tone: bool = False
    teleport: Optional["TeleportationConfig"] = None


@dataclass
class TeleportationConfig:
    """Parameters governing photonic teleportation interactions."""

    pair_id: str = "link-0"
    fidelity_target: float = 0.9
    coincidence_window: float = 1e-9
    entanglement_source: str = "spdc"
    enable_feedforward: bool = True


@dataclass
class DualClockSchedule:
    """Full schedule with ordered pulse operations."""

    backend: DualClockingBackend
    operations: List[PulseOp] = field(default_factory=list)

    def append(self, op: PulseOp) -> None:
        self.operations.append(op)


class DualClockScheduler:
    """Create schedules tailored to the selected backend."""

    def __init__(self, backend: DualClockingBackend) -> None:
        self.backend = backend

    def build_schedule(self, config: DualClockConfig) -> DualClockSchedule:
        schedule = DualClockSchedule(backend=self.backend)

        drive_meta_1 = dict(self.backend.drive_metadata(two_tone=config.two_tone))
        drive_meta_1.update({"rabi_rate": config.rabi_rate})
        schedule.append(PulseOp(kind="drive_1", duration=config.t_drive_1, metadata=drive_meta_1))

        if config.teleport is not None:
            teleport_meta = self.backend.teleport_metadata(
                pair_id=config.teleport.pair_id,
                fidelity_target=config.teleport.fidelity_target,
            )
            if teleport_meta is not None:
                teleport_meta = dict(teleport_meta)
                teleport_meta.update(
                    {
                        "entanglement_source": config.teleport.entanglement_source,
                        "feedforward": config.teleport.enable_feedforward,
                        "coincidence_window": config.teleport.coincidence_window,
                    }
                )
                schedule.append(
                    PulseOp(
                        kind="teleport",
                        duration=config.teleport.coincidence_window,
                        metadata=teleport_meta,
                    )
                )

        probe_meta = dict(self.backend.probe_metadata(threshold=config.probe_threshold))
        schedule.append(PulseOp(kind="probe", duration=config.t_probe, metadata=probe_meta))

        drive_meta_2 = dict(self.backend.drive_metadata(two_tone=config.two_tone))
        drive_meta_2.update({"rabi_rate": config.rabi_rate})
        schedule.append(PulseOp(kind="drive_2", duration=config.t_drive_2, metadata=drive_meta_2))

        return schedule
