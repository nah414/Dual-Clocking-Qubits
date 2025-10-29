import argparse
from dual_clocking.backends import SuperconductingBackend, TrappedIonBackend, NMRBackend
from dual_clocking.experiments.dcdramsey import dc_ramsey

def parse_args():
    p = argparse.ArgumentParser(description="Run Dual-Clocking demo")
    p.add_argument("--backend", choices=["sc", "ion", "nmr"], default="sc")
    p.add_argument("--t-wait", type=float, default=2e-7, help="Probe duration / wait time (s)")
    p.add_argument("--strength", type=float, default=0.2, help="Probe strength (0..1)")
    p.add_argument("--drive-amp", type=float, default=0.5, help="Drive amplitude (0..1)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.backend == "sc":
        be = SuperconductingBackend(dt=1e-9)
    elif args.backend == "ion":
        be = TrappedIonBackend(dt=1e-6)
    else:
        be = NMRBackend(dt=1e-6)
    be.calibrate(0)
    res = dc_ramsey(be, qubit=0, t_wait=args.t_wait, drive_amp=args.drive_amp, probe_strength=args.strength)
    for i, r in enumerate(res):
        print(f"Probe {i}: {r.__dict__}")
