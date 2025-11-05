# Repository Improvement Suggestions

## Declare external plotting dependencies
- The `quantum_link_fidelity` CLI imports both `matplotlib` and `numpy`, so running the tool in a fresh environment without those packages will fail immediately. Consider moving these libraries into the project dependencies (for example via `requirements.txt` or an `extras_require` entry) so users get the right stack when they install the project.
- As part of the same change, updating the installation guidance in the README to mention the plotting requirements would prevent confusion for contributors and CI users.

## Streamline test imports
- `tests/test_scheduler.py` manipulates `sys.path` to import the package under test. Installing the package in editable mode for the test session (or using relative imports) would avoid path hacks and better reflect real-world usage, especially once CI is set up.

## Expand simulator coverage
- The Bloch-sphere simulator covers several behaviors (drive rotation, probe feedback, and helper utilities) but currently lacks direct unit tests. Adding targeted cases for `run_schedule`, `require_sqi_edge_fraction`, and `decode_majority` would guard against regressions in feed-forward logic and validation rules.
