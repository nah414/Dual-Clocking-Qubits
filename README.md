# Dual Clocking Qubits

This repository provides a lightweight, backend-aware drive–probe–drive
simulator.  Schedules are built through `DualClockScheduler` and may target
multiple hardware families by choosing the appropriate backend implementation.

```python
from dual_clocking import (
    DualClockConfig,
    DualClockScheduler,
    SimulatorConfig,
    SuperconductingBackend,
    TrappedIonBackend,
    run_schedule,
)

backend = SuperconductingBackend(drive_frequency=5.0, readout_frequency=6.5)
scheduler = DualClockScheduler(backend)
schedule = scheduler.build_schedule(DualClockConfig(two_tone=True, probe_threshold=-0.05))
result = run_schedule(schedule, SimulatorConfig(physical_correction=True))
print(result.probe)
```

Switching to trapped-ion control only requires swapping the backend:

```python
ion_backend = TrappedIonBackend(raman_frequency=80.0, photon_collection_efficiency=0.35)
ion_schedule = DualClockScheduler(ion_backend).build_schedule(
    DualClockConfig(t_drive_1=10e-6, t_probe=5e-6, t_drive_2=12e-6, probe_threshold=0.05)
)
ion_result = run_schedule(ion_schedule)
print(ion_result.final_state)
```

## Command-line Ramsey example

Install the package in editable mode and run the included example CLIs to execute
simulated Ramsey or NMR experiments against the desired backend:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e . -r requirements.txt
python -m examples.run_dc_ramsey --backend ion --t-wait 10e-6 --probe-strength 0.2
python -m examples.run_dc_ramsey --backend superconducting --t-wait 2e-7 --probe-strength 0.2
python -m examples.run_nmr --mode fid --amp 1.0 --strength 0.5 --acq 2.56e-3
python -m examples.run_nmr --mode echo --amp 1.0 --tau 1e-3 --strength 0.5 --acq 2.56e-3
```

For a quick FID run with sensible defaults you can also rely on the helper
target:

```bash
make run-nmr
```

The script prints the schedule metadata, simulator configuration, probe result,
and any feed-forward corrections that were applied.

See `dual_clocking/README_BACKENDS.md` for a concise overview of the available
hardware abstractions.

## Security scanning with PhantomRaven

Run the bundled PhantomRaven CLI to ensure npm manifests and lockfiles are free
from known-bad packages and suspicious remote URLs:

```bash
# run a scan
python phantomraven.py .

# fail CI on findings with JSON output
python phantomraven.py . --json

# install a pre-commit that routes through this script
python phantomraven.py --install-precommit
```
