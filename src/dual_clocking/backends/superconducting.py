from __future__ import annotations

import math
from typing import Dict, Any, Optional

from .base import Backend, ProbeResult

class SuperconductingBackend(Backend):
    def __init__(self, dt: float = 1e-9, **kwargs):
        super().__init__(dt=dt, name="superconducting", **kwargs)
        self.frames = {}

    def calibrate(self, qubit: int, **kwargs) -> Dict[str, Any]:
        omega_ge = kwargs.get("omega_ge", 5.0e9)
        omega_r  = kwargs.get("omega_r", 7.0e9)
        self.frames[(qubit, "drive")] = (omega_ge, 0.0)
        self.frames[(qubit, "probe")] = (omega_r, 0.0)
        return {"omega_ge": omega_ge, "omega_r": omega_r, "dt": self.dt}

    def set_frame(self, qubit: int, freq: float, phase: float = 0.0, domain: str = "drive") -> None:
        self.frames[(qubit, domain)] = (float(freq), float(phase))

    def shift_phase(self, qubit: int, dphi: float, domain: str = "drive") -> None:
        f, phi = self.frames.get((qubit, domain), (0.0, 0.0))
        self.frames[(qubit, domain)] = (f, (phi + dphi) % (2*math.pi))

    def play_pulse(self, qubit: int, duration: float, amp: float, phase: float = 0.0, freq: Optional[float] = None, domain: str = "drive", **kwargs) -> None:
        frame_freq, frame_phase = self.frames.get((qubit, domain), (0.0, 0.0))
        eff_freq = frame_freq if freq is None else freq
        eff_phase = frame_phase + phase
        _ = (eff_freq, eff_phase, duration, amp, domain)

    def probe(self, qubit: int, duration: float, strength: float, detuning: float = 0.0, weak: bool = True, **kwargs) -> ProbeResult:
        k = kwargs.get("k_dephasing", 2e6)
        gamma_phi = k * max(0.0, strength)**2
        signal = strength * duration
        return ProbeResult(counts=None, signal=signal, p01=None, gamma_phi=gamma_phi, metadata={"detuning": detuning, "weak": weak})

    def measure(self, qubit: int, duration: float, **kwargs) -> ProbeResult:
        return self.probe(qubit, duration=duration, strength=1.0, weak=False, **kwargs)
