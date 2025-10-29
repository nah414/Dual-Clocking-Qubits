"""Trapped-ion backend description."""

from __future__ import annotations

from typing import Dict

from .base import DualClockingBackend


class TrappedIonBackend(DualClockingBackend):
    """Backend tuned for Raman/AOM control of trapped-ion qubits."""

    def __init__(self, *, raman_frequency: float, photon_collection_efficiency: float, dt: float = 1e-6) -> None:
        super().__init__(name="trapped-ion", dt=dt)
        self.raman_frequency = raman_frequency
        self.photon_collection_efficiency = photon_collection_efficiency

    def drive_metadata(self, *, two_tone: bool) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "frame": "raman_beam",
            "frame_frequency": self.raman_frequency,
            "units": "MHz",
            "aom_sidebands": 2 if two_tone else 1,
            "two_tone": two_tone,
        }
        self.annotate(payload)
        return payload

    def probe_metadata(self, *, threshold: float) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "method": "photon-counting",
            "collection_efficiency": self.photon_collection_efficiency,
            "threshold": threshold,
            "units": "photons",
        }
        self.annotate(payload)
        return payload
