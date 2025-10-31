# Dual‑Clocking Qubits (DCQ) — 500‑Qubit Scale‑Up Roadmap (v0.1)

**Owners:** Adam × Nova  
**Repo:** `Dual-Clocking-Qubits`  
**Status:** Draft for review — designed to be dropped into the repo as `docs/ROADMAP-DCQ-500.md`

---

## 0) Executive Summary

We are scaling the Dual‑Clocking (DC) control method from small‑device experiments to **~500 qubits operating concurrently**. In this roadmap we define the target architecture, the timing/clock distribution strategy, control/readout multiplexing, calibration automation, and the software and verification plan required to make DC work at scale.

> **Working theory for “Dual‑Clocking” (DC)**: each qubit has **two phase‑coherent, independently programmable reference clocks** (implemented as *virtual clocks* via numerically controlled oscillators, NCOs), enabling (a) robust mitigation of frequency drift/crosstalk through fast retuning; (b) mode switching between gate and idling frames; and (c) advanced dynamical decoupling / AC‑Stark bias tricks without re‑wiring LOs. DC can be realized either with two physical LOs per bank or—preferably—**digitally** via multi‑tone direct upconversion locked to a master 10 MHz + 1 PPS reference. This roadmap assumes the **digital dual‑NCO** approach.

**North‑star KPI:** run **simultaneous randomized benchmarking (sRB)** across **≥500 qubits** with median 1‑qubit gate error ≤ **3e‑4**, 2‑qubit gate error ≤ **1e‑2**, **<50 ps** inter‑channel timing skew, and automated re‑calibration that keeps ≥90% of qubits within spec for ≥8 hours unattended.

---

## 1) System Architecture (High‑Level)

### 1.1 Timing & Clock Tree
- **Master reference:** lab‑grade 10 MHz + 1 PPS; distribute via low‑jitter fan‑out (star + daisy hybrids).  
- **DC “virtual clocks”:** per‑qubit dual NCOs (f₁,f₂) inside AWG/FPGA DUCs; both phase‑locked to master.  
- **Deterministic triggers:** global deterministic start (1 PPS) + per‑island start‑of‑sequence markers; time tags in run logs.  
- **Skew budget:** end‑to‑end <50 ps between any two channels; enforce via calibrated cable delays + per‑channel phase offsets.  
- **Phase noise budget:** L(f) consistent with single‑qubit RB <3e‑4 (track at 10 kHz–10 MHz offsets).

### 1.2 Control Fabric (XY, Flux, Readout)
- **Digital upconversion (DUC):** multi‑tone DACs (≥2.5 GS/s, ≥12 bit) with per‑tone amplitude/phase control.  
- **Vector switch matrix (optional):** for bank sharing / fast re‑routes during maintenance.  
- **Flux/ZZ control (if applicable):** slower DACs with aggressive filtering; gate‑synchronous with XY via shared triggers.  
- **Readout:** frequency‑multiplexed resonators; target **8–16:1** mux per line with JPAs/TWPAs per group.  
- **Cryo wiring:** adopt **island topology** of 64–96 qubits per island to localize crosstalk and simplify routing.

### 1.3 Software/Compile Path
- **Pulse IR:** adopt a hardware‑agnostic pulse IR (OpenPulse/QIR‑Pulse/MLIR‑dialect).  
- **Frame semantics:** each qubit exposes two frames `f1`, `f2` with commands: `set_detuning`, `set_phase`, `swap_frames`, `chirp`, `park`.  
- **Scheduler:** resource‑aware, island‑aware; emits deterministic event tables with integer sample timing.  
- **Observability:** structured logs: experiment hash, reference time, per‑channel phase/amp at start of run, firmware version.  

---

## 2) Frequency Planning & Crosstalk Strategy

### 2.1 Guard‑Bands
- **Qubit XY band:** e.g., 4.3–6.2 GHz (transmons) or as appropriate for platform.  
- **Per‑island spacing:** ≥10–15 MHz minimum separation; neighbors get **orthogonal park frames** (f₁ vs f₂) to reduce spectral crowding.  
- **Readout band:** cluster by island; **250–400 MHz** spans per readout feedline; ≥1–2 MHz spacing per resonator; avoid TLS hot‑spots.  
- **Two‑tone DC use:** reserve **Δ=5–50 MHz** between f₁ and f₂ per qubit to enable fast retune/AC‑Stark bias without LO retune.

### 2.2 Allocation Algorithm (repo deliverable)
- **Inputs:** desired frequencies, forbidden windows, coupling graph, readout mux ratio, DC Δ‑windows.  
- **Outputs:** (f₁,f₂) per qubit, readout allocation, island grouping, guard‑band slacks.  
- **Method:** graph coloring + local search; minimize pairwise spectral proximity weighted by couplings; respect island channel budgets.

> **Action:** add `tools/fplan/allocator.py` producing `frequency_plan.json` and a plot. Unit tests with randomized device graphs.

---

## 3) Calibration at Scale (Automation‑First)

### 3.1 Calibration DAG (always‑on background)
- **Level 0 (device):** reference distribution health, skew map, DAC linearity, JPA bias points, readout chains SNR.  
- **Level 1 (per‑qubit):** `(f₁,f₂)` idles, amplitude/phase calibration, DRAG/derivative coefficients, π/2 lengths, readout assignment.  
- **Level 2 (pairs/edges):** 2Q entangler tune‑ups (CZ/iSWAP), echo sequences, spectator idles on alternate frames.  
- **Level 3 (global):** sRB in tiles (island‑wise), drift monitors, crosstalk tomography snapshots.

### 3.2 Throughput Targets
- **Warm start (daily):** ≤60 min for 500 qubits via parallel islands; **continuous trickle‑recal** replaces big‑bang recal.  
- **Drift guards:** dual‑clock “park” frame reduces idling sensitivity; auto‑retune f₁ from f₂ if drift exceeds thresholds.  
- **Data store:** `caldb/` (JSON or SQLite) with versioned parameter sets; immutable runs refer to exact snapshot IDs.

### 3.3 API & Services (repo deliverables)
- `cal/runner.py` — executes DAG nodes; concurrency limits per island; retries with backoff.  
- `cal/metrics.py` — sRB aggregations, thresholding, health scores; Prometheus/CSV exporters.  
- `cal/tasks/*.py` — modular tasks (Rabi, Ramsey on f₁/f₂, phase‑walk, spectroscopies, 2Q tune‑ups).  
- **Unit tests:** determinism of schedulers; simulated plants (noise models) to validate convergence heuristics.

---

## 4) Readout @ 500 Qubits

- **Mux ratio:** start at **8:1**, stretch to **12–16:1** once SNR+linearity margins proven.  
- **Isolation plan:** stagger readout pulses across islands by a few hundred ns to avoid JPA compression.  
- **Dynamic range audits:** per‑island compression points > sum of tones + 6 dB headroom; continuous PSD monitors.  
- **Discriminators:** calibrate IQ blobs per island; DC gives option to idle on alternate frame during readout to reduce back‑action.

**Deliverables:** `readout/lineup.py`, `readout/jpa_guard.py`, and `docs/readout-multiplexing.md` with gain maps and safe operating regions.

---

## 5) Control Firmware & Timing Details

- **Event timing grid:** 1 sample = 0.4 ns (2.5 GS/s) or 0.5 ns (2.0 GS/s). All events integer‑aligned.  
- **Per‑qubit dual NCOs:** commands `nco.set_freq(frame, Hz)`, `nco.set_phase(frame, rad)`, `nco.swap_frames()` cost ≤2 cycles.  
- **Chirps & phase ramps:** linear piece‑wise ramps compiled into LUT bursts.  
- **Skew calibration:** measure with loopback fixtures or dedicated qubit pairs; store per‑channel phase offsets.  
- **Jitter budget:** ≤300 fs RMS at DAC clock; maintain via disciplined PLLs; verify by measuring inter‑channel Ramsey.

**Deliverables:** `ctrl/driver.py`, `ctrl/firmware/` (register maps), `docs/timing-and-frames.md` (with examples).

---

## 6) Scheduler & Compiler

- **Resource model:** channels, islands, readout lanes, JPAs; DC frames as first‑class schedulable resources.  
- **Conflict rules:** prevent same‑island simultaneous high‑power ops that trip compression; enforce guard‑bands.  
- **Deterministic build:** compile → canonical event table (`.evt`) → binary for AWGs/SoCs; include hash + caldb snapshot ID.  
- **Interfaces:** Python driver + CLI (`dcq run schedule.yaml`), optional QIR/OpenPulse loader.

**Deliverables:** `sched/` package, `examples/schedules/*.yaml`, conversion tools, unit tests for determinism and resource conflicts.

---

## 7) Verification & Continuous Testing

### 7.1 Bench & Sim
- **Sim models:** phase‑noise injection, amplitude drift, crosstalk kernels; validate dual‑frame robustness.  
- **Golden tests:** (i) single‑qubit RB on f₁ vs f₂; (ii) swap‑frame mid‑sequence; (iii) 2Q gate with spectators parked.  
- **Stress test:** sRB across 500 qubits in tiles; ensure scheduler and firmware stay deterministic and within power budgets.

### 7.2 CI Hooks (repo)
- `ci/run_srb_tiles.py` — runs nightly synthetic tiles, checks expected error envelopes.  
- GitHub Actions job `dcq-sim.yml`: run fast sims, lint frequency plans, and validate cal DAG edges.  
- Badges: build, sim, coverage.

---

## 8) Milestones & Deliverables

> Time is indicative; parallelize across teams. “Definition of done” (DoD) is explicit per milestone.

### M0 — Architecture Spike (Weeks 0–3)
- **Docs:** this roadmap; timing tree; frequency planning spec.  
- **Prototypes:** dual‑NCO API; tiny scheduler stub; basic cal tasks (Rabi/Ramsey).  
- **DoD:** demo swap‑frames on 2–4 qubits; event table hash round‑trip.

### M1 — 32‑Q DC Island (Weeks 4–9)
- **Hardware:** first island with full timing discipline; 8:1 readout; JPA chain.  
- **Software:** allocator v1; scheduler v1; cal DAG level‑1 automated.  
- **DoD:** sRB on 32 qubits concurrently; median 1Q error ≤5e‑4; unattended 2‑hour stability.

### M2 — 128‑Q (Weeks 10–18)
- **Hardware:** two islands; isolation patterns; telemetry.  
- **Software:** allocator v2 (coupling‑weighted), scheduler v2 (conflict rules), cal level‑2 (2Q) automated.  
- **DoD:** 128‑Q sRB tiles with spectators parked; median 2Q error ≤1.5e‑2.

### M3 — 256‑Q (Weeks 19–28)
- **Hardware:** expand readout mux to 12–16:1 where safe; power audits.  
- **Software:** cal trickle engine; drift detectors; automated retune f₁ from f₂ on threshold.  
- **DoD:** 256‑Q sRB, 8‑hour unattended; 90% within spec.

### M4 — 500‑Q (Weeks 29–42)
- **Hardware:** five islands of ~100Q (or 8×64Q); refined isolation; spare capacity for hot‑swaps.  
- **Software:** scheduler v3 (global tiling), cal level‑3 snapshots, fleet dashboards.  
- **DoD:** 500‑Q sRB concurrently; med 1Q ≤3e‑4, 2Q ≤1e‑2; <50 ps skew verified; weekly soak tests passed.

---

## 9) Risk Register & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Phase noise / spurs | Gate errors inflate | Tight PLL discipline, clock audits, spur maps, tone dithering |
| Readout compression | Nonlinear distortions | Staggered readout windows, per‑island power budgets, JPA bias auto‑trim |
| Spectral crowding | Frequency collisions | Graph‑aware allocator, alternate frame parking, guard‑band enforcement |
| Calibration drift | Frequent failures | Trickle recal, dual‑frame retune, health thresholds, snapshot rollback |
| Firmware non‑determinism | Irreproducible runs | Canonical event tables, hashes, pinned versions, replay harness |
| Cryo wiring limits | Scale stalls | Island topology, multiplexing, spare channels, switch matrices |
| Tooling sprawl | Dev friction | Pulse IR abstraction, stable APIs, CI sim harness |

---

## 10) Repo Changes (Structure & Tasks)

```
docs/
  ROADMAP-DCQ-500.md      ← this file
  timing-and-frames.md
  readout-multiplexing.md
  allocator-spec.md
ctrl/
  driver.py
  firmware/
sched/
  __init__.py
  scheduler.py
  resources.py
cal/
  runner.py
  tasks/
  metrics.py
readout/
  lineup.py
  jpa_guard.py
tools/
  fplan/allocator.py
  fplan/tests/test_allocator.py
ci/
  run_srb_tiles.py
```

- **Issues/Epics:** create GitHub Epics for M0…M4; labels: `area/cal`, `area/sched`, `area/ctrl`, `area/readout`, `island:X`, `priority:P0…P2`.
- **Coding standards:** black/ruff; mypy optional; unit tests via pytest; type hints encouraged.
- **Docs workflow:** `docs/` are source‑of‑truth; keep diagrams in `docs/img/` with source (draw.io or mermaid).

---

## 11) Measurement & KPIs

- **Core:** sRB medians (1Q/2Q), readout assignment fidelity, uptime %, #qubits in‑spec over time.  
- **Infra:** skew RMS, PLL lock stability, JPA compression margin, scheduler determinism rate.  
- **Ops:** time‑to‑recal, failure recovery time, nightly CI pass rate.

---

## 12) Next Actions (Two‑Week Sprint Backlog)

1. Land `docs/timing-and-frames.md` describing dual‑NCO frames and command set.  
2. Implement `tools/fplan/allocator.py` v0 with graph coloring + tests.  
3. Build scheduler v0 with island resources and deterministic event tables.  
4. Add cal tasks for Rabi/Ramsey on `f₁` and `f₂`; wire to tiny runner.  
5. Define telemetry schema; add minimal CLI for sRB tile execution and dumping metrics.  
6. Stand up CI job for allocator/scheduler unit tests and simulated sRB tiles.

---

### Appendix A — DC Command Examples (Pseudo‑code)

```python
# Frame setup
nco.set_freq(q, frame='f1', Hz=5_123_000_000.0)
nco.set_freq(q, frame='f2', Hz=5_147_000_000.0)
nco.set_phase(q, frame='f1', rad=0.0)
nco.set_phase(q, frame='f2', rad=0.0)

# Gate sequence on f1, spectators parked on f2
play.xy_pulse(q, frame='f1', kind='gaussian', amp=0.23, dur=32)   # X/2
play.wait(ns=20)
play.xy_pulse(q, frame='f1', kind='drag', amp=0.46, dur=32)       # X
play.swap_frames(q)  # f1 ↔ f2 (spectator parked on alternate frame)
```

---

**This roadmap is intentionally opinionated and assumes a digital dual‑NCO implementation.** If physical dual‑LO banks are required by platform constraints, we can adapt with the same logical API and more aggressive LO distribution discipline.
