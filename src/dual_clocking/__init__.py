"""Dual-clocking control & simulation scaffolding.

This package adds a **backend abstraction** for running the Drive–Probe–Drive
method on heterogeneous hardware (superconducting or trapped-ion).

Version: 0.3.1
"""

from .scheduler import DualClockScheduler, DrivePulse, ProbePulse, Barrier, PhaseShift, GradientPulse, Delay

__all__ = [
    "DualClockScheduler",
    "DrivePulse",
    "ProbePulse",
    "Barrier",
    "PhaseShift",
    "GradientPulse",
    "Delay",
]
