from dual_clocking_qubit import (
    DualClockingConfig, DualClockingController, require_sqi_edge_fraction
)

cfg = DualClockingConfig(
    omega_d = 2*math.pi*5.0e9,   # metadata for downstream pulse mappers
    Omega_amp = 2*math.pi*25e6,  # 25 MHz Rabi
    t_drive1 = 40e-9, t_probe = 20e-9, t_drive2 = 40e-9,
    epsilon_probe = 0.05, T1 = 30e-6, Tphi = 40e-6,
    prefer_virtual_Z = True, apply_corrections = True,
    meas_axis = 'z', meas_error = 0.015, enable_two_tone = False
)

require_sqi_edge_fraction(0.8, threshold=cfg.require_sqi_edge_fraction)

ctrl = DualClockingController(cfg)
ctrl.build_schedule()
res = ctrl.simulate(init_state="|0>", dt=0.25e-9, record=True, seed=7)

print(res["bloch_final"], res["meas_outcome"])
print(res.get("probe_snapshot"))  # mid-circuit readout estimate

## Example programs

```bash
# 90° pulse followed by an NMR FID
make run-nmr

# Simple two-lobe gradient echo
make run-nmr-grad
```

