from __future__ import annotations

import math

from ..scheduler import DualClockScheduler, DrivePulse, ProbePulse, Barrier
from ..backends import Backend

def dc_ramsey(backend: Backend, qubit: int, t_wait: float, drive_amp: float = 0.5, probe_strength: float = 0.2):
    sch = DualClockScheduler(backend)
    if backend.name == "trapped_ion":
        drive_duration = 2/3 * 1e-6
    elif backend.name == "nmr":
        par = getattr(backend, "params", None) or backend.calibrate(qubit)
        gamma = par["gamma_rad_per_t"]
        B1_max = par["B1_max"]
        amp = max(min(drive_amp, 1.0), 1e-6)
        drive_duration = (0.5 * math.pi) / (gamma * B1_max * amp)
    else:
        drive_duration = 20e-9

    sch.add(DrivePulse(qubit=qubit, duration=drive_duration, amp=drive_amp, phase=0.0))
    sch.add(Barrier())
    sch.add(ProbePulse(qubit=qubit, duration=t_wait, strength=probe_strength, detuning=0.0, weak=True))
    sch.add(Barrier())
    sch.add(DrivePulse(qubit=qubit, duration=drive_duration, amp=drive_amp, phase=0.0))
    return sch.run()
