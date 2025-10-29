from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class ProbeResult:
    counts: Optional[int] = None
    signal: Optional[float] = None
    p01: Optional[float] = None
    gamma_phi: Optional[float] = None
    metadata: Dict[str, Any] = None

class Backend(ABC):
    def __init__(self, dt: float, name: str, **kwargs):
        self.dt = float(dt)
        self.name = name
        self.capabilities = kwargs

    @abstractmethod
    def calibrate(self, qubit: int, **kwargs) -> Dict[str, Any]: ...

    @abstractmethod
    def set_frame(self, qubit: int, freq: float, phase: float = 0.0, domain: str = "drive") -> None: ...

    @abstractmethod
    def shift_phase(self, qubit: int, dphi: float, domain: str = "drive") -> None: ...

    @abstractmethod
    def play_pulse(self, qubit: int, duration: float, amp: float, phase: float = 0.0, freq: Optional[float] = None, domain: str = "drive", **kwargs) -> None: ...

    @abstractmethod
    def probe(self, qubit: int, duration: float, strength: float, detuning: float = 0.0, weak: bool = True, **kwargs) -> ProbeResult: ...

    @abstractmethod
    def measure(self, qubit: int, duration: float, **kwargs) -> ProbeResult: ...

    def barrier(self) -> None:
        return
