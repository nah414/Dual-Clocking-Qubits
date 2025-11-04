"""NMR backend description."""

from __future__ import annotations

from typing import Dict

from .base import DualClockingBackend


class NMRBackend(DualClockingBackend):
    """Backend tuned for nuclear magnetic resonance control hardware."""

    def __init__(
        self,
        *,
        carrier_frequency: float,
        b1_field: float,
        dt: float = 5e-6,
    ) -> None:
        super().__init__(name="nmr", dt=dt)
        self.carrier_frequency = carrier_frequency
        self.b1_field = b1_field

    def drive_metadata(self, *, two_tone: bool) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "frame": "rf_transmit",
            "frame_frequency": self.carrier_frequency,
            "units": "MHz",
            "nutation_field": self.b1_field,
            "two_tone": two_tone,
            "envelope": "composite" if two_tone else "hard",
        }
        self.annotate(payload)
        return payload

    def probe_metadata(self, *, threshold: float) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "method": "inductive",
            "receiver_bandwidth": 1.0 / self.dt,
            "threshold": threshold,
            "units": "arb",
        }
        self.annotate(payload)
        return payload
