# Dual-Clocking Hardware Backends

Two clock domains per qubit:
- **drive**: gate control
- **probe**: weak measurement / Stark probe

Use the `DualClockScheduler` with `SuperconductingBackend`, `TrappedIonBackend`, or the new `NMRBackend` for ensemble spin control.
