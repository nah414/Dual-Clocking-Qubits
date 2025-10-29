"""Command-line entry point for running a dual-clock Ramsey experiment simulation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Tuple

from dual_clocking import (
    DualClockConfig,
    DualClockScheduler,
    SimulatorConfig,
    SuperconductingBackend,
    TrappedIonBackend,
    run_schedule,
)


def _build_backends() -> Tuple[SuperconductingBackend, TrappedIonBackend]:
    sc_backend = SuperconductingBackend(drive_frequency=5.0, readout_frequency=6.5)
    ion_backend = TrappedIonBackend(raman_frequency=80.0, photon_collection_efficiency=0.35)
    return sc_backend, ion_backend


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("superconducting", "ion"),
        default="superconducting",
        help="Select the hardware backend abstraction to use.",
    )
    parser.add_argument(
        "--t-wait",
        type=float,
        default=None,
        help="Wait/probe duration between the drive pulses (seconds).",
    )
    parser.add_argument(
        "--probe-strength",
        type=float,
        default=None,
        help="Probe threshold that triggers feed-forward corrections.",
    )
    parser.add_argument(
        "--two-tone",
        action="store_true",
        help="Enable two-tone driving when supported by the backend.",
    )
    parser.add_argument(
        "--physical-correction",
        action="store_true",
        help="Apply physical feed-forward corrections instead of virtual phases.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Seed for the stochastic measurement process.",
    )
    args = parser.parse_args(argv)

    sc_backend, ion_backend = _build_backends()
    backend = sc_backend if args.backend == "superconducting" else ion_backend

    cfg = DualClockConfig()
    if args.t_wait is not None:
        cfg.t_probe = float(args.t_wait)
    if args.probe_strength is not None:
        cfg.probe_threshold = float(args.probe_strength)
    cfg.two_tone = bool(args.two_tone)

    scheduler = DualClockScheduler(backend)
    schedule = scheduler.build_schedule(cfg)

    sim_cfg = SimulatorConfig(seed=args.seed, physical_correction=args.physical_correction)
    result = run_schedule(schedule, sim_cfg)

    print("Backend:", backend.name)
    print("Schedule operations:")
    for op in schedule.operations:
        print(f"  - {op.kind} ({op.duration:.3e}s): {op.metadata}")
    print("Simulator config:", asdict(sim_cfg))
    print("Probe result:", result.probe)
    print("Feed-forward events:", result.feed_forward_events)
    print("Final state:", result.final_state)


if __name__ == "__main__":
    main()
