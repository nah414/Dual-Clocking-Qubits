"""Gradient echo demonstration for the NMR backend."""

from __future__ import annotations

import argparse
import math

from dual_clocking.backends import NMRBackend
from dual_clocking.scheduler import (
    DualClockScheduler,
    DrivePulse,
    GradientPulse,
    ProbePulse,
    Barrier,
    Delay,
)


def pi_half_duration(backend: NMRBackend, qubit: int, amp: float) -> float:
    params = getattr(backend, "params", None) or backend.calibrate(qubit)
    gamma = params["gamma_rad_per_t"]
    b1_max = params["B1_max"]
    amp = max(min(amp, 1.0), 1e-6)
    return (0.5 * math.pi) / (gamma * b1_max * amp)


def pi_duration(backend: NMRBackend, qubit: int, amp: float) -> float:
    params = getattr(backend, "params", None) or backend.calibrate(qubit)
    gamma = params["gamma_rad_per_t"]
    b1_max = params["B1_max"]
    amp = max(min(amp, 1.0), 1e-6)
    return math.pi / (gamma * b1_max * amp)


def run_gradient_echo(gradient: float, gtime: float, tau: float, acq: float, amp: float) -> None:
    backend = NMRBackend()
    backend.calibrate(0)

    schedule = DualClockScheduler(backend)
    t90 = pi_half_duration(backend, 0, amp)
    t180 = pi_duration(backend, 0, amp)

    schedule.add(DrivePulse(qubit=0, duration=t90, amp=amp, phase=math.pi / 2))
    schedule.add(Barrier())
    schedule.add(GradientPulse(qubit=0, duration=gtime, strength=gradient))
    schedule.add(Delay(duration=tau))
    schedule.add(DrivePulse(qubit=0, duration=t180, amp=amp))
    schedule.add(GradientPulse(qubit=0, duration=gtime, strength=-gradient))
    schedule.add(Delay(duration=tau))
    schedule.add(ProbePulse(qubit=0, duration=acq, strength=0.8, weak=False))

    results = schedule.run()
    if not results:
        print("No acquisition returned")
        return

    res = results[0]
    fid = res.metadata.get("fid", []) if res.metadata else []
    print("Gradient echo acquisition complete")
    print(f"Echo samples: {len(fid)} rms={res.signal:.6g}")
    print(f"Residual spread: {getattr(backend, '_phase_spread', {}).get(0, 0.0):.6g} rad")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-lobe gradient echo demo")
    parser.add_argument("--G", type=float, default=0.8, help="Gradient strength (T/m)")
    parser.add_argument("--gtime", type=float, default=1e-3, help="Gradient lobe duration (s)")
    parser.add_argument("--tau", type=float, default=1e-3, help="Free evolution interval (s)")
    parser.add_argument("--acq", type=float, default=2.56e-3, help="Acquisition window (s)")
    parser.add_argument("--amp", type=float, default=0.6, help="Normalized RF amplitude (0..1)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_gradient_echo(
        gradient=args.G,
        gtime=args.gtime,
        tau=args.tau,
        acq=args.acq,
        amp=args.amp,
    )
