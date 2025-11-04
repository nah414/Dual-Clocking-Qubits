import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dual_clocking import (
    BUILTIN_BACKENDS,
    DualClockConfig,
    DualClockScheduler,
    NMRBackend,
    SuperconductingBackend,
    TeleportationConfig,
    TelecomPhotonicsBackend,
    TrappedIonBackend,
    available_backends,
    get_backend_class,
)
from dual_clocking.backends import plugins as backend_plugins
from dual_clocking.backends.base import DualClockingBackend


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


def test_telecom_backend_with_teleportation():
    backend = TelecomPhotonicsBackend(
        central_frequency_thz=193.1,
        fiber_length_km=50.0,
        entanglement_rate=1.5e6,
    )
    scheduler = DualClockScheduler(backend)
    cfg = DualClockConfig(
        two_tone=False,
        t_drive_1=5e-9,
        t_probe=8e-9,
        teleport=TeleportationConfig(
            pair_id="node-a:node-b",
            fidelity_target=0.97,
            coincidence_window=2.5e-9,
            entanglement_source="quantum-dot",
            enable_feedforward=False,
        ),
    )
    schedule = scheduler.build_schedule(cfg)

    assert schedule.operations[0].kind == "drive_1"
    teleport_op = schedule.operations[1]
    assert teleport_op.kind == "teleport"
    assert teleport_op.duration == pytest.approx(2.5e-9)
    assert teleport_op.metadata["backend"] == "telecom-photonics"
    assert teleport_op.metadata["method"] == "entanglement-swapping"
    assert teleport_op.metadata["link_pair"] == "node-a:node-b"
    assert teleport_op.metadata["entanglement_source"] == "quantum-dot"
    assert teleport_op.metadata["feedforward"] is False

    probe = schedule.operations[2]
    assert probe.kind == "probe"
    assert probe.metadata["method"] == "superconducting-nanowire"


def test_plugin_registry_includes_builtins():
    available = available_backends()
    assert all(name in available for name in BUILTIN_BACKENDS)
    assert available["telecom-photonics"] is TelecomPhotonicsBackend


def test_plugin_registry_discovers_entry_points(monkeypatch):
    class DummyBackend(DualClockingBackend):
        def __init__(self) -> None:
            super().__init__(name="dummy", dt=1e-9)

        def drive_metadata(self, *, two_tone: bool) -> dict:
            payload = {"two_tone": two_tone}
            self.annotate(payload)
            return payload

        def probe_metadata(self, *, threshold: float) -> dict:
            payload = {"threshold": threshold}
            self.annotate(payload)
            return payload

    class DummyEntryPoint:
        def __init__(self, name: str):
            self.name = name

        def load(self):
            return DummyBackend

    def fake_iter_entry_points():
        yield DummyEntryPoint("dummy")

    monkeypatch.setattr(backend_plugins, "_iter_entry_points", fake_iter_entry_points)

    cls = get_backend_class("dummy")
    assert cls is DummyBackend

    available = available_backends()
    assert "dummy" in available
    assert available["dummy"] is DummyBackend

    with pytest.raises(KeyError):
        get_backend_class("missing")
