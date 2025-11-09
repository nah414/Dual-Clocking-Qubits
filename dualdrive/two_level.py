# Dual-Drive + Probe Timing Optimizer — Two-level models
# Version: v0.2 (2025-11-09, America/Chicago)

from __future__ import annotations
import numpy as np
from .core import (
    average_gate_fidelity_montecarlo,
    build_period_superop_twolevel,
    prepare_monte_carlo_batch,
)

# ---------- Simple flat envelopes ----------
def make_envelopes(T: float, Om1_val: float, Om2_val: float):
    def Omega1(t: float) -> float:
        return Om1_val
    def Omega2(t: float) -> float:
        return Om2_val
    return Omega1, Omega2

# ---------- Simulate a single period and return the superoperator ----------
def simulate_period_two_level(
    base_params: dict,
    dt: float = 2e-3,
    method: str = "rk4",
):
    return build_period_superop_twolevel(base_params, dt=dt, method=method)

# ---------- Identity target by default ----------
def target_unitary_identity():
    return np.eye(2, dtype=complex)

# ---------- Parameter sweep over (delta phase, probe start) ----------
def sweep_2level(
    base_params: dict,
    dphi_grid: np.ndarray,
    tau_grid: np.ndarray,
    dt: float = 2e-3,
    trials_mc: int = 200,
    seed: int = 0,
    method: str = "rk4",
):
    U_tgt = target_unitary_identity()
    batch = prepare_monte_carlo_batch(U_tgt, trials=trials_mc, seed=seed)
    F = np.zeros((len(tau_grid), len(dphi_grid)))
    best = {"F": -1.0, "tau": None, "dphi": None}

    for i, tau in enumerate(tau_grid):
        for j, dphi in enumerate(dphi_grid):
            params = dict(base_params)
            params["tau"] = float(tau)
            params["dphi"] = float(dphi)
            superop = simulate_period_two_level(params, dt=dt, method=method)
            Fij = average_gate_fidelity_montecarlo(U_tgt, superop, batch=batch)
            F[i, j] = Fij
            if Fij > best["F"]:
                best.update({"F": Fij, "tau": float(tau), "dphi": float(dphi)})
    return F, best

# ---------- Simple coordinate-descent refinement ----------
def refine_local(
    base_params: dict,
    tau0: float,
    dphi0: float,
    dt: float = 2e-3,
    steps: int = 20,
    eps_tau: float = 0.01,
    eps_phi: float = 0.05,
    trials_mc: int = 300,
    seed: int = 1,
    method: str = "rk4",
):
    U_tgt = target_unitary_identity()
    batch = prepare_monte_carlo_batch(U_tgt, trials=trials_mc, seed=seed)
    tau = tau0; ph = dphi0
    bestF = -1.0

    def evalF(tau_, ph_):
        p = dict(base_params)
        p["tau"] = float(np.clip(tau_, 0.0, p["T"] - p["tau_p"] - 1e-9))
        # wrap phase to (-pi, pi]
        x = ((ph_ + np.pi) % (2*np.pi)) - np.pi
        p["dphi"] = float(x)
        sop = simulate_period_two_level(p, dt=dt, method=method)
        return average_gate_fidelity_montecarlo(U_tgt, sop, batch=batch), p["tau"], p["dphi"]

    for _ in range(steps):
        improved = False
        for dta, dph in ((eps_tau, 0.0), (-eps_tau, 0.0), (0.0, eps_phi), (0.0, -eps_phi)):
            F1, tau1, ph1 = evalF(tau + dta, ph + dph)
            if F1 > bestF:
                bestF = F1; tau = tau1; ph = ph1; improved = True
        if not improved:
            break
    return {"F": bestF, "tau": float(tau), "dphi": float(ph)}
