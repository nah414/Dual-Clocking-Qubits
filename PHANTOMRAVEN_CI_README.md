# PhantomRaven CI

This repository uses **PhantomRaven** to scan for known-bad npm packages and suspicious remote URLs on every push/PR, and weekly on a schedule.

- Workflow: `.github/workflows/phantomraven-scan.yml`
- Scanner script expected at repo root: `phantomraven.py`
- Optional local blocklist: `.phantomraven_blocklist.txt`

To run locally:
```bash
python phantomraven.py . --json
```
