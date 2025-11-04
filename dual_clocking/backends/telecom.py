"""Telecom photonics backend description."""

from __future__ import annotations

from typing import Dict, Optional

from .base import DualClockingBackend


class TelecomPhotonicsBackend(DualClockingBackend):
    """Backend tuned for fiber-optic photonic entanglement distribution."""

    def __init__(
        self,
        *,
        central_frequency_thz: float,
        fiber_length_km: float,
        entanglement_rate: float,
        dt: float = 5e-12,
    ) -> None:
        super().__init__(name="telecom-photonics", dt=dt)
        self.central_frequency_thz = central_frequency_thz
        self.fiber_length_km = fiber_length_km
        self.entanglement_rate = entanglement_rate

    def drive_metadata(self, *, two_tone: bool) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "frame": "telecom_carrier",
            "frame_frequency": self.central_frequency_thz,
            "units": "THz",
            "dispersion_compensation": "dual-tone" if two_tone else "single-tone",
            "two_tone": two_tone,
            "fiber_length_km": self.fiber_length_km,
        }
        self.annotate(payload)
        return payload

    def probe_metadata(self, *, threshold: float) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "method": "superconducting-nanowire",
            "coincidence_threshold": threshold,
            "units": "counts",  # photon detection counts
            "fiber_length_km": self.fiber_length_km,
        }
        self.annotate(payload)
        return payload

    def teleport_metadata(
        self,
        *,
        pair_id: str,
        fidelity_target: float,
    ) -> Optional[Dict[str, object]]:
        payload: Dict[str, object] = {
            "method": "entanglement-swapping",
            "link_pair": pair_id,
            "fidelity_target": fidelity_target,
            "fiber_length_km": self.fiber_length_km,
            "entanglement_rate": self.entanglement_rate,
        }
        self.annotate(payload)
        return payload
