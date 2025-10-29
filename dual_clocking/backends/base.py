"""Backend abstractions for the dual clocking simulator."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Mapping, MutableMapping


@dataclass
class ProbeResult:
    """Container describing the outcome of a probe step."""

    outcome: int
    estimator: float
    metadata: Mapping[str, object]


class DualClockingBackend(ABC):
    """Abstract backend describing hardware-specific scheduling metadata."""

    name: str
    dt: float

    def __init__(self, name: str, dt: float) -> None:
        self.name = name
        self.dt = dt

    @abstractmethod
    def drive_metadata(self, *, two_tone: bool) -> Dict[str, object]:
        """Return metadata for a drive pulse."""

    @abstractmethod
    def probe_metadata(self, *, threshold: float) -> Dict[str, object]:
        """Return metadata for a probe/readout pulse."""

    def annotate(self, payload: MutableMapping[str, object]) -> None:
        """Apply backend-wide annotations to a metadata payload in-place."""

        payload.setdefault("backend", self.name)
        payload.setdefault("dt", self.dt)
