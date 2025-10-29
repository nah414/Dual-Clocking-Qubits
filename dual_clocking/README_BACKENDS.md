# Backend guide

The package now exposes explicit backend classes so the same drive–probe–drive
logic can be adapted to multiple hardware stacks:

- `SuperconductingBackend` encodes GHz frame frequencies, dispersive readout
  parameters, and nanosecond time steps suitable for circuit QED platforms.
- `TrappedIonBackend` reflects Raman / AOM control with microsecond time steps
  and photon-count detection metadata.
- `NMRBackend` captures RF transmit/receive characteristics with microsecond
  sampling suitable for bulk-spin NMR experiments.

Each backend populates the schedule metadata differently, allowing experiments
and downstream exporters to branch on `op.metadata["backend"]` without custom
conditionals scattered throughout the code base.
