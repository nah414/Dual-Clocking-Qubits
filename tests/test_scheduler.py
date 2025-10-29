import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dual_clocking import (
    DualClockConfig,
    DualClockScheduler,
    NMRBackend,
    SuperconductingBackend,
    TrappedIonBackend,
)


def test_superconducting_metadata():
    backend = SuperconductingBackend(drive_frequency=5.0, readout_frequency=6.5)
    scheduler = DualClockScheduler(backend)
    cfg = DualClockConfig(two_tone=True, probe_threshold=-0.05)
    schedule = scheduler.build_schedule(cfg)

    drive_1 = schedule.operations[0]
    assert drive_1.metadata["backend"] == "superconducting"
    assert drive_1.metadata["envelope"] == "two-tone"
    assert drive_1.metadata["units"] == "GHz"

    probe = schedule.operations[1]
    assert probe.metadata["method"] == "dispersive"
    assert probe.metadata["threshold"] == -0.05


def test_trapped_ion_metadata():
    backend = TrappedIonBackend(raman_frequency=80.0, photon_collection_efficiency=0.4)
    scheduler = DualClockScheduler(backend)
    cfg = DualClockConfig(t_drive_1=10e-6, t_probe=5e-6, t_drive_2=12e-6)
    schedule = scheduler.build_schedule(cfg)

    drive_1 = schedule.operations[0]
    assert drive_1.metadata["backend"] == "trapped-ion"
    assert drive_1.metadata["units"] == "MHz"

    probe = schedule.operations[1]
    assert probe.metadata["method"] == "photon-counting"
    assert probe.metadata["units"] == "photons"


def test_nmr_metadata():
    backend = NMRBackend(carrier_frequency=150.0, b1_field=0.025)
    scheduler = DualClockScheduler(backend)
    cfg = DualClockConfig(two_tone=True, probe_threshold=0.2)
    schedule = scheduler.build_schedule(cfg)

    drive = schedule.operations[0]
    assert drive.metadata["backend"] == "nmr"
    assert drive.metadata["envelope"] == "composite"
    assert drive.metadata["units"] == "MHz"

    probe = schedule.operations[1]
    assert probe.metadata["method"] == "inductive"
    assert probe.metadata["units"] == "arb"
    assert probe.metadata["threshold"] == 0.2
