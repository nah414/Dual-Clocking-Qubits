"""Drive–probe–drive demonstration for both hardware classes."""

from __future__ import annotations

from .. import (
    DualClockConfig,
    DualClockScheduler,
    SimulatorConfig,
    run_schedule,
    SuperconductingBackend,
    TrappedIonBackend,
)


def build_and_run_superconducting() -> None:
    backend = SuperconductingBackend(drive_frequency=5.0, readout_frequency=6.5)
    scheduler = DualClockScheduler(backend)
    schedule = scheduler.build_schedule(DualClockConfig(two_tone=True, probe_threshold=-0.1))
    result = run_schedule(schedule)
    print("Superconducting backend final state:", result.final_state)
    print("Probe result:", result.probe)


def build_and_run_trapped_ion() -> None:
    backend = TrappedIonBackend(raman_frequency=80.0, photon_collection_efficiency=0.35)
    scheduler = DualClockScheduler(backend)
    config = DualClockConfig(
        t_drive_1=10e-6,
        t_probe=5e-6,
        t_drive_2=12e-6,
        probe_threshold=0.05,
    )
    result = run_schedule(scheduler.build_schedule(config), SimulatorConfig(physical_correction=True))
    print("Trapped-ion backend final state:", result.final_state)
    print("Probe result:", result.probe)


if __name__ == "__main__":
    build_and_run_superconducting()
    build_and_run_trapped_ion()
