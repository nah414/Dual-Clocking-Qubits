import math

from dual_clocking.backends import SuperconductingBackend, TrappedIonBackend, NMRBackend
from dual_clocking.scheduler import DualClockScheduler, DrivePulse, ProbePulse, Barrier, GradientPulse, Delay

def test_run_sc_backend():
    be = SuperconductingBackend()
    be.calibrate(0)
    sch = DualClockScheduler(be)
    sch.add(DrivePulse(qubit=0, duration=20e-9, amp=0.5))
    sch.add(ProbePulse(qubit=0, duration=200e-9, strength=0.2))
    sch.add(Barrier())
    res = sch.run()
    assert len(res) == 1

def test_run_ion_backend():
    be = TrappedIonBackend()
    be.calibrate(0)
    sch = DualClockScheduler(be)
    sch.add(DrivePulse(qubit=0, duration=0.6e-6, amp=0.5))
    sch.add(ProbePulse(qubit=0, duration=10e-6, strength=0.2))
    res = sch.run()
    assert len(res) == 1


def test_run_nmr_backend_with_gradient():
    be = NMRBackend()
    params = be.calibrate(0)
    gamma = params["gamma_rad_per_t"]
    B1_max = params["B1_max"]
    amp = 0.6
    tau_90 = (0.5 * math.pi) / (gamma * B1_max * amp)
    tau_180 = math.pi / (gamma * B1_max * amp)
    sch = DualClockScheduler(be)
    sch.add(DrivePulse(qubit=0, duration=tau_90, amp=amp, phase=math.pi / 2))
    sch.add(GradientPulse(qubit=0, duration=0.8e-3, strength=0.5))
    sch.add(Delay(duration=1e-3))
    sch.add(DrivePulse(qubit=0, duration=tau_180, amp=amp, phase=0.0))
    sch.add(GradientPulse(qubit=0, duration=0.8e-3, strength=-0.5))
    sch.add(Delay(duration=1e-3))
    sch.add(ProbePulse(qubit=0, duration=2.56e-3, strength=0.5))
    res = sch.run()
    assert len(res) == 1
    assert res[0].metadata is not None
    assert "fid" in res[0].metadata
