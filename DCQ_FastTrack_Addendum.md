# DCQ Fast‑Track Addendum — Safely Compressing to 500 Qubits

**Owners:** Adam × Nova  
**Version:** v0.1 (Fast‑Track)  
**Intent:** Shorten M0→M4 to ~24 weeks while maintaining high‑fidelity targets (1Q ≤3e‑4, 2Q ≤1e‑2, <50 ps skew).

---

## A. Summary of the Acceleration Plan
We compress the 42‑week baseline to **~24 weeks** by: (1) parallelizing island bring‑up, (2) front‑loading simulation + CI to eliminate lab dead time, (3) locking *copy‑exact* templates (controls, wiring, readout) after the first island, (4) aggressive automation of calibration “trickle‑recal,” and (5) deferring non‑critical options (e.g., 16:1 readout) until after 256Q.

**Non‑negotiable guardrails:** deterministic event tables; CI‑gated allocator/scheduler; per‑island skew <50 ps; sRB tiles must pass before scaling.

---

## B. Accelerated Milestones (Gated)

### FT‑M0 — Architecture + Tooling Spike (Weeks 0–2)
- Deliver: timing tree spec, allocator v0, scheduler v0, cal tasks (Rabi/Ramsey f₁/f₂), CI green.  
- Gate: golden tests pass; event‑table determinism; allocator separation ≥10 MHz on random graphs.

### FT‑M1 — **Copy‑Exact Island** @ 48–64Q (Weeks 3–6)
- Deliver: first island with full timing discipline; 8:1 readout; JPA chain; trickle‑recal v1.  
- Gate: island sRB med 1Q ≤5e‑4; unattended 2 h stability; <50 ps skew verified; freeze “copy‑exact” BOM + wiring + firmware image.

### FT‑M2 — **Quad Island** @ 192–256Q (Weeks 7–12)
- Deliver: replicate M1 template x4 in parallel; scheduler v2 (island‑aware conflicts); allocator v1 (coupling‑weighted).  
- Gate: sRB tiles across 256Q; med 2Q ≤1.5e‑2; 6–8 h unattended with ≥85% qubits in spec; telemetry dashboards online.

### FT‑M3 — **Octa Island** @ 384Q (Weeks 13–18)
- Deliver: add 2–4 islands; readout stays **8:1** (defer 12–16:1); power audits pass; drift detectors auto‑retune f₁ from f₂.  
- Gate: 384Q sRB tiles; ≥88% in spec; nightly CI sim + lab metrics alignment within tolerance bands.

### FT‑M4 — **500Q Cohort** (Weeks 19–24)
- Deliver: 500Q online; scheduler v3 (global tiling); level‑3 cal snapshots; spare channels for hot‑swap.  
- Gate: 500Q concurrent sRB; med 1Q ≤3e‑4, 2Q ≤1e‑2; weekly soak test; <50 ps skew lab‑verified.

---

## C. What Enables the Speed‑Up (Concrete Moves)

1. **Parallel “Islands‑First” Bring‑up**  
   - Staff per‑island crews; central timing/firmware team; daily copy‑exact drops.  
   - Hot‑spare JPAs/readout chains to avoid blocking on repairs.

2. **Copy‑Exact Discipline**  
   - Freeze after FT‑M1; changes batched and A/B tested on a sacrificial island before fleet rollout.

3. **Simulation‑First CI**  
   - Fast physics‑lite sims for allocator/scheduler/cals; nightly stress harness with injected drift/crosstalk envelopes.  
   - CI blocks merges that would violate guard‑bands or determinism.

4. **Aggressive Automation**  
   - Trickle‑recal runs continuously; failure thresholds trigger auto retune (f₁ ← f₂) and open an incident with artifacts attached.

5. **Deferrals that Don’t Hurt Fidelity Now**  
   - Keep readout at **8:1** through FT‑M3; push 12–16:1 after 500Q.  
   - Two‑tone advanced modes optional; default to Drive–Probe–Drive on f₁ with spectators parked on f₂.  

---

## D. Risk Buy‑Down & Telemetry

- **Phase noise:** spur scans logged daily; PLL discipline audits; reject builds with out‑of‑family L(f).  
- **Compression:** per‑island PSD monitors; staggered readout windows; automatic JPA bias trim.  
- **Spectral crowding:** allocator reports slack histograms; red‑flag if any pair <10 MHz within an island.  
- **Drift:** fleet dashboard: % qubits auto‑retuned; MTBF for retunes; rollback to last‑known‑good cal snapshot on spike.

---

## E. Minimal‑Viable Feature Set (to hold the line on scope)

- Pulse IR with dual‑frame semantics only (no exotic gates yet).  
- Deterministic event tables + hash; reproducible replay.  
- sRB, Rabi, Ramsey, 2Q tune‑up primitives; no optional tomography suites until post‑FT‑M4.  
- 8:1 readout, island staggering; no dynamic remux during FT milestones.

---

## F. Success Metrics (Leading Indicators)

- **Build time to “green island”:** ≤ 3 days from rack‑ready.  
- **CI drift delta:** |lab − sim| error envelopes shrink week‑over‑week.  
- **Trickle‑recal efficacy:** ≥75% of drifts auto‑recovered without human touch.  
- **Copy‑exact divergence:** <5% param variance across islands after FT‑M2.

---

## G. Required Resourcing (Lean)

- **Timing/firmware** (2–3 FTE): PLL/DUC, skew metrology, event engine.  
- **Controls software** (2–3 FTE): scheduler, allocator, cal runner, CI harness.  
- **Readout & cryo** (2–3 FTE): JPAs/TWPAs, mux, power audits, repairs.  
- **QE/TE** (1–2 FTE): sRB, data plumbing, dashboards, incident response.

---

## H. Immediate Actions (Next 10 Days)

1. Lock copy‑exact checklist from the first island build (BOM + wiring + firmware).  
2. Land CI tests: allocator guard‑band property tests (Hypothesis), scheduler determinism seeds, trickle‑recal dry‑run.  
3. Stand up fleet dashboards: skew, sRB medians, drift retunes, red/yellow/green rollups per island.  
4. Pre‑stage hot spares (JPA, cryo cables, DACs); pre‑calibrated replacements cut mean down‑time.  
5. Freeze optional scope; raise a change‑control gate for anything beyond this addendum.

---

> This fast‑track keeps the original **fidelity and skew targets** intact by trading feature scope and maximizing parallelism + automation. Once 500Q is stable, we can resume readout 12–16:1, advanced two‑tone sequences, and deeper tomography.
