from dual_clocking.backends import SuperconductingBackend
from dual_clocking.experiments.dcdramsey import dc_ramsey

if __name__ == "__main__":
    be = SuperconductingBackend(dt=1e-9)
    be.calibrate(0)
    res = dc_ramsey(be, qubit=0, t_wait=200e-9, drive_amp=0.5, probe_strength=0.2)
    for i, r in enumerate(res):
        print(f"[SC] Probe {i}: signal={r.signal:.6g}, gamma_phi={r.gamma_phi:.6g}, meta={r.metadata}")
