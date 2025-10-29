from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

from .backends.base import Backend, ProbeResult

@dataclass
class DrivePulse:
    qubit: int
    duration: float
    amp: float
    phase: float = 0.0
    freq: Optional[float] = None
    domain: str = "drive"

@dataclass
class ProbePulse:
    qubit: int
    duration: float
    strength: float
    detuning: float = 0.0
    weak: bool = True
    domain: str = "probe"

@dataclass
class PhaseShift:
    qubit: int
    dphi: float
    domain: str = "drive"

@dataclass
class GradientPulse:
    """Gradient pulse for backends that support spatial encoding (e.g. NMR)."""

    qubit: int
    duration: float
    strength: float
    axis: str = "z"

@dataclass
class Delay:
    duration: float

@dataclass
class Barrier:
    label: str = "global"

Pulse = Union[DrivePulse, ProbePulse, PhaseShift, GradientPulse, Delay, Barrier]

class DualClockScheduler:
    def __init__(self, backend: Backend):
        self.backend = backend
        self.timeline: List[Pulse] = []

    def add(self, pulse: Pulse) -> None:
        self.timeline.append(pulse)

    def clear(self) -> None:
        self.timeline.clear()

    def run(self) -> List[ProbeResult]:
        results: List[ProbeResult] = []
        t = 0.0
        dt = self.backend.dt

        for p in self.timeline:
            if isinstance(p, Barrier):
                self.backend.barrier()
                continue
            if isinstance(p, Delay):
                t += max(dt, round(p.duration / dt) * dt)
                continue
            if isinstance(p, PhaseShift):
                self.backend.shift_phase(p.qubit, p.dphi, domain=p.domain)
                continue

            snapped = max(dt, round(p.duration / dt) * dt)

            if isinstance(p, DrivePulse):
                self.backend.play_pulse(p.qubit, duration=snapped, amp=p.amp, phase=p.phase, freq=p.freq, domain=p.domain)
            elif isinstance(p, ProbePulse):
                res = self.backend.probe(p.qubit, duration=snapped, strength=p.strength, detuning=p.detuning, weak=p.weak, t_start=t)
                results.append(res)
            elif isinstance(p, GradientPulse):
                if not hasattr(self.backend, "apply_gradient"):
                    raise AttributeError(f"Backend '{self.backend.name}' does not support gradient pulses")
                self.backend.apply_gradient(p.qubit, duration=snapped, strength=p.strength, axis=p.axis)

            t += snapped

        return results
