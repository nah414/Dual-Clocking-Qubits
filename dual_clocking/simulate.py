"""Lightweight Bloch-vector simulator for dual clocking schedules."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .backends.base import ProbeResult
from .scheduler import DualClockSchedule

Bloch = Tuple[float, float, float]


def _rz(state: Bloch, theta: float) -> Bloch:
    x, y, z = state
    c, s = math.cos(theta), math.sin(theta)
    return (x * c - y * s, x * s + y * c, z)


def _rx(state: Bloch, theta: float) -> Bloch:
    x, y, z = state
    c, s = math.cos(theta), math.sin(theta)
    return (x, y * c - z * s, y * s + z * c)


def _dephase(state: Bloch, gamma: float) -> Bloch:
    x, y, z = state
    k = max(0.0, 1.0 - gamma)
    return (x * k, y * k, z)


def _measure_z(state: Bloch, rng: random.Random) -> Tuple[int, Bloch, float]:
    x, y, z = state
    p1 = (1.0 + z) * 0.5
    outcome = 1 if rng.random() < p1 else 0
    post = (0.0, 0.0, 1.0 if outcome == 1 else -1.0)
    return outcome, post, z


@dataclass
class SimulatorConfig:
    seed: int = 7
    gamma_dephasing: float = 0.01
    physical_correction: bool = False
    virtual_phase: float = 0.1 * math.pi
    physical_theta: float = 0.15 * math.pi


@dataclass
class SimResult:
    final_state: Bloch
    probe: ProbeResult
    feed_forward_events: List[Dict[str, object]] = field(default_factory=list)


def run_schedule(schedule: DualClockSchedule, config: SimulatorConfig | None = None) -> SimResult:
    """Simulate the provided schedule and return the final Bloch state."""

    cfg = config or SimulatorConfig()
    rng = random.Random(cfg.seed)
    state: Bloch = (0.0, 0.0, 1.0)
    feed_events: List[Dict[str, object]] = []
    probe_result = ProbeResult(outcome=0, estimator=1.0, metadata={})

    for op in schedule.operations:
        if op.kind.startswith("drive"):
            rabi_rate = float(op.metadata.get("rabi_rate", 1.0))
            theta = rabi_rate * op.duration
            if op.metadata.get("two_tone"):
                state = _rz(state, 0.25 * theta)
                state = _rx(state, theta)
                state = _rz(state, -0.15 * theta)
            else:
                state = _rx(state, theta)
            state = _dephase(state, cfg.gamma_dephasing)

        elif op.kind == "probe":
            outcome, post, est = _measure_z(state, rng)
            state = post
            threshold = float(op.metadata.get("threshold", 0.0))
            if est < threshold:
                if cfg.physical_correction:
                    state = _rx(state, cfg.physical_theta)
                    feed_events.append({
                        "mode": "physical",
                        "theta": cfg.physical_theta,
                        "backend": op.metadata.get("backend"),
                    })
                else:
                    feed_events.append({
                        "mode": "virtual",
                        "phase": cfg.virtual_phase,
                        "backend": op.metadata.get("backend"),
                    })
            probe_result = ProbeResult(outcome=outcome, estimator=est, metadata=dict(op.metadata))

    return SimResult(final_state=state, probe=probe_result, feed_forward_events=feed_events)


def require_sqi_edge_fraction(edge_fraction: float, *, threshold: float) -> None:
    if edge_fraction < threshold:
        raise ValueError(
            f"SQI edge_fraction {edge_fraction:.2f} below required threshold {threshold:.2f}"
        )


def decode_majority(measurements: List[int]) -> int:
    if not measurements:
        raise ValueError("Cannot decode majority of empty measurement list")
    ones = sum(measurements)
    zeros = len(measurements) - ones
    return 1 if ones >= zeros else 0
