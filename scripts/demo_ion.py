from dual_clocking.backends import TrappedIonBackend
from dual_clocking.experiments.dcdramsey import dc_ramsey

if __name__ == "__main__":
    be = TrappedIonBackend(dt=1e-6)
    be.calibrate(0)
    res = dc_ramsey(be, qubit=0, t_wait=10e-6, drive_amp=0.5, probe_strength=0.2)
    for i, r in enumerate(res):
        print(f"[ION] Probe {i}: counts={r.counts}, signal={r.signal:.6g}, gamma_phi={r.gamma_phi:.6g}, meta={r.metadata}")
