"""Superconducting circuit backend description."""

from __future__ import annotations

from typing import Dict

from .base import DualClockingBackend


class SuperconductingBackend(DualClockingBackend):
    """Metadata tuned for dispersive readout on superconducting qubits."""

    def __init__(self, *, drive_frequency: float, readout_frequency: float, dt: float = 1e-9) -> None:
        super().__init__(name="superconducting", dt=dt)
        self.drive_frequency = drive_frequency
        self.readout_frequency = readout_frequency

    def drive_metadata(self, *, two_tone: bool) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "frame": "drive_q",
            "frame_frequency": self.drive_frequency,
            "units": "GHz",
            "envelope": "two-tone" if two_tone else "single-tone",
            "two_tone": two_tone,
        }
        self.annotate(payload)
        return payload

    def probe_metadata(self, *, threshold: float) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "method": "dispersive",
            "readout_frequency": self.readout_frequency,
            "threshold": threshold,
            "units": "GHz",
        }
        self.annotate(payload)
        return payload
