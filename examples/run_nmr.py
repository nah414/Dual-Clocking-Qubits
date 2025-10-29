"""Simple FID example for the NMR backend."""

from __future__ import annotations

import argparse
import math
from typing import Sequence

from dual_clocking.backends import NMRBackend
from dual_clocking.scheduler import DualClockScheduler, DrivePulse, ProbePulse, Barrier


def pi_half_duration(backend: NMRBackend, qubit: int, amp: float) -> float:
    params = getattr(backend, "params", None) or backend.calibrate(qubit)
    gamma = params["gamma_rad_per_t"]
    b1_max = params["B1_max"]
    amp = max(min(amp, 1.0), 1e-6)
    return (0.5 * math.pi) / (gamma * b1_max * amp)


def run_fid(amp: float, strength: float, acq: float) -> None:
    backend = NMRBackend()
    backend.calibrate(0)

    schedule = DualClockScheduler(backend)
    t90 = pi_half_duration(backend, qubit=0, amp=amp)
    schedule.add(DrivePulse(qubit=0, duration=t90, amp=amp, phase=math.pi / 2))
    schedule.add(Barrier())
    schedule.add(ProbePulse(qubit=0, duration=acq, strength=strength, weak=True))

    results = schedule.run()
    if not results:
        print("No acquisition returned")
        return

    fid = results[0].metadata.get("fid", []) if results[0].metadata else []
    preview = format_preview(fid, 8)
    print("FID acquisition complete")
    print(f"Samples: {len(fid)} dt={results[0].metadata.get('rx_dt', 'n/a')} s")
    print(f"RMS amplitude: {results[0].signal:.6g}")
    print(f"gamma_phi: {results[0].gamma_phi:.6g}")
    print(f"First samples: {preview}")


def format_preview(values: Sequence[float], limit: int) -> str:
    if not values:
        return "[]"
    shown = ", ".join(f"{v:.3g}" for v in values[:limit])
    suffix = ", ..." if len(values) > limit else ""
    return f"[{shown}{suffix}]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a 90° excitation followed by an FID acquisition")
    parser.add_argument("--amp", type=float, default=0.6, help="Normalized RF amplitude (0..1)")
    parser.add_argument("--strength", type=float, default=0.5, help="Receiver gain scaling (0..1)")
    parser.add_argument("--acq", type=float, default=2.56e-3, help="Acquisition window (s)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_fid(amp=args.amp, strength=args.strength, acq=args.acq)
