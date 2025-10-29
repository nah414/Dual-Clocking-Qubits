# Dual-Clocking Qubits

Dual clocking uses independent **drive** and **probe** references to interrogate a
qubit or spin ensemble between two timing domains. This repository offers a small
Python package that demonstrates how a scheduler can target different hardware
models while sharing the same experiment descriptions.

The package currently ships with three simulator-style backends:

- `SuperconductingBackend` – a transmon-inspired control and readout model.
- `TrappedIonBackend` – resonant Raman drive with photon-counting probe rates.
- `NMRBackend` – a bulk NMR ensemble supporting gradient echoes and FIDs.

Each backend implements the same abstract interface so you can swap targets with
one import change.

## Installation

Create a virtual environment and install the project in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

The editable install exposes all three backends from `dual_clocking.backends`
without additional extras, so superconducting, trapped-ion, and NMR workflows
are available immediately.

## Quick start

Run the demo Ramsey experiment against any backend via the helper script:

```bash
python main.py --backend sc   # superconducting
python main.py --backend ion  # trapped ion
python main.py --backend nmr  # NMR ensemble
```

For NMR-specific experiments the repository also includes richer examples:

```bash
make run-nmr       # 90° pulse followed by an FID acquisition
make run-nmr-grad  # two-lobe gradient echo demonstration
```

All demos use the shared `DualClockScheduler` timeline abstraction. See
`examples/` and `scripts/` for reference code.

## Testing

Run the test suite with:

```bash
pytest -q
```

Tests focus on scheduler semantics to ensure drive, probe, gradient, and delay
operations behave consistently across backends.
