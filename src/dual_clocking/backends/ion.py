from __future__ import annotations

import math
from typing import Dict, Any, Optional

from .base import Backend, ProbeResult

class TrappedIonBackend(Backend):
    def __init__(self, dt: float = 1e-6, **kwargs):
        super().__init__(dt=dt, name="trapped_ion", **kwargs)
        self.frames = {}

    def calibrate(self, qubit: int, **kwargs) -> Dict[str, Any]:
        omega_0 = kwargs.get("omega_0", 2 * math.pi * 12.642e6)
        omega_p = kwargs.get("omega_probe", omega_0 + 2e4)
        self.frames[(qubit, "drive")] = (omega_0, 0.0)
        self.frames[(qubit, "probe")] = (omega_p, 0.0)
        return {"omega_0": omega_0, "omega_probe": omega_p, "dt": self.dt}

    def set_frame(self, qubit: int, freq: float, phase: float = 0.0, domain: str = "drive") -> None:
        self.frames[(qubit, domain)] = (float(freq), float(phase))

    def shift_phase(self, qubit: int, dphi: float, domain: str = "drive") -> None:
        f, phi = self.frames.get((qubit, domain), (0.0, 0.0))
        self.frames[(qubit, domain)] = (f, (phi + dphi) % (2*math.pi))

    def play_pulse(self, qubit: int, duration: float, amp: float, phase: float = 0.0, freq: Optional[float] = None, domain: str = "drive", **kwargs) -> None:
        frame_freq, frame_phase = self.frames.get((qubit, domain), (0.0, 0.0))
        eff_freq = frame_freq if freq is None else freq
        eff_phase = frame_phase + phase
        aom_latency = kwargs.get("aom_latency", 1e-6)
        _ = (eff_freq, eff_phase, duration + aom_latency, amp, domain)

    def probe(self, qubit: int, duration: float, strength: float, detuning: float = 0.0, weak: bool = True, **kwargs) -> ProbeResult:
        R = kwargs.get("rate_scale", 5e4) * max(0.0, strength)**2
        counts = int(R * max(0.0, duration))
        beta = kwargs.get("beta_dephasing", 1e-6)
        gamma_phi = beta * R
        signal = counts / max(duration, 1e-9)
        return ProbeResult(counts=counts, signal=signal, p01=None, gamma_phi=gamma_phi, metadata={"detuning": detuning, "weak": weak})

    def measure(self, qubit: int, duration: float, **kwargs) -> ProbeResult:
        return self.probe(qubit, duration=duration, strength=1.0, weak=False, **kwargs)
