"""Command-line entry point for running simplified NMR-style sequences."""

from __future__ import annotations

import argparse
from dataclasses import asdict

from dual_clocking import (
    DualClockConfig,
    DualClockScheduler,
    NMRBackend,
    SimulatorConfig,
    run_schedule,
)


def _build_backend() -> NMRBackend:
    return NMRBackend(carrier_frequency=150.0, b1_field=0.025)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("fid", "echo"),
        default="fid",
        help="Choose between a free-induction decay or Hahn-echo style sequence.",
    )
    parser.add_argument(
        "--amp",
        type=float,
        default=1.0,
        help="Drive amplitude scaling applied to the RF pulses.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.0,
        help="Probe threshold magnitude that triggers feed-forward.",
    )
    parser.add_argument(
        "--acq",
        type=float,
        default=1.0e-3,
        help="Acquisition window / probe duration in seconds.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Free-precession duration (seconds) used for Hahn-echo drive legs.",
    )
    parser.add_argument(
        "--physical-correction",
        action="store_true",
        help="Apply physical rotations instead of virtual frame updates when probing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=11,
        help="Seed for the pseudo-random measurement outcomes.",
    )
    args = parser.parse_args(argv)

    backend = _build_backend()

    cfg = DualClockConfig()
    cfg.rabi_rate = float(args.amp)
    cfg.probe_threshold = float(args.strength)
    cfg.t_probe = float(args.acq)
    cfg.two_tone = args.mode == "echo"

    if args.mode == "echo":
        if args.tau is None:
            raise SystemExit("--tau is required when --mode echo is selected")
        cfg.t_drive_1 = float(args.tau) * 0.5
        cfg.t_drive_2 = float(args.tau) * 0.5
    elif args.tau is not None:
        cfg.t_drive_2 = float(args.tau)

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
