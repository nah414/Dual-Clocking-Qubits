from __future__ import annotations

import math
from typing import Tuple

def bloch_evolve(theta: float, phi: float, gamma_phi: float, duration: float) -> Tuple[float, float]:
    shrink = math.exp(-gamma_phi * max(duration, 0.0))
    return theta, phi

def estimate_dephasing_from_probe(gamma_phi: float, duration: float) -> float:
    return gamma_phi * max(duration, 0.0)
