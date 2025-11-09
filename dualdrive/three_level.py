# Dual-Drive + Probe Timing Optimizer — Three-level transmon (leakage + DRAG)
# Version: v0.2 (2025-11-09, America/Chicago)

from __future__ import annotations
import numpy as np
from .core import propagate_lindblad

# Basis: |0>, |1>, |2>
I3 = np.eye(3, dtype=complex)

# Projectors and ladder ops
P0 = np.diag([1,0,0]).astype(complex)
P1 = np.diag([0,1,0]).astype(complex)
P2 = np.diag([0,0,1]).astype(complex)
S01 = np.zeros((3,3), dtype=complex); S01[0,1]=1; S01[1,0]=1  # x-like on 0-1
A01 = np.zeros((3,3), dtype=complex); A01[0,1]=-1j; A01[1,0]=1j  # y-like on 0-1
S12 = np.zeros((3,3), dtype=complex); S12[1,2]=1; S12[2,1]=1
A12 = np.zeros((3,3), dtype=complex); A12[1,2]=-1j; A12[2,1]=1j

# Dephasing operator that distinguishes |0> and |1> primarily (readout-like)
Sz01 = np.diag([1,-1,0]).astype(complex)

def drive_hamiltonian_transmon(t: float, p: dict) -> np.ndarray:
    """
    H0 = diag(0, w01, w01+w12)
    Drive couples 0-1 and 1-2 with relative matrix elements (1, sqrt(2)) unless overridden.
    DRAG term on 0-1 quadrature: + beta * d/dt Omega01 * A01
    """
    w01 = p["w01"]
    w12 = p["w12"]
    H0 = np.diag([0.0, w01, w01 + w12]).astype(complex)

    # envelopes
    Om1 = p["Omega1"](t)  # primary amplitude (0-1 leg)
    Om2 = p["Omega2"](t)  # secondary amplitude
    D1  = p.get("Delta1", 0.0)
    D2  = p.get("Delta2", 0.0)
    phi1 = p.get("phi1", 0.0)
    phi2 = p.get("phi2", 0.0)
    dphi = p.get("dphi", 0.0)
    ratio_12 = p.get("ratio_12", np.sqrt(2.0))  # 1-2 matrix element scale

    # Simple near-RWA transverse fields on 0-1 and 1-2
    c1, s1 = np.cos(D1*t + phi1), np.sin(D1*t + phi1)
    c2, s2 = np.cos(D2*t + phi2 + dphi), np.sin(D2*t + phi2 + dphi)

    H01 = 0.5 * (Om1 * (c1 * S01 + s1 * A01))
    H12 = 0.5 * (ratio_12 * Om2 * (c2 * S12 + s2 * A12))

    # DRAG: quadrature on 0-1 proportional to d/dt Omega1
    if "Omega1_dot" in p:
        beta = p.get("beta_drag", 0.0)
        H_drag = 0.5 * beta * p["Omega1_dot"](t) * A01
    else:
        H_drag = 0.0 * I3

    return H0 + H01 + H12 + H_drag

def make_envelopes_flat(T: float, Om1: float, Om2: float, beta_drag: float = 0.0):
    def Omega1(t: float) -> float: return Om1
    def Omega2(t: float) -> float: return Om2
    def Omega1_dot(t: float) -> float: return 0.0
    env = {"Omega1": Omega1, "Omega1_dot": Omega1_dot, "Omega2": Omega2, "beta_drag": beta_drag}
    return env

def make_in_probe(tau: float, tau_p: float):
    def _f(t: float) -> bool:
        return (t >= tau) and (t < tau + tau_p)
    return _f

def simulate_period_three_level(
    base: dict,
    dt: float = 2e-3,
    method: str = "rk4",
):
    """
    Returns (final_rho, leakage_prob, superop_func_2x2_projection)
    For convenience, we also return a superoperator restricted to the {|0>,|1>} subspace
    by embedding rho(2x2) -> rho(3x3) with zeros in |2>, evolving, then projecting back.
    """
    T = base["T"]; tau = base["tau"]; tau_p = base["tau_p"]
    Gam = base.get("Gamma_m", 0.0)
    in_probe = make_in_probe(tau, tau_p)
    Ls_probe = [np.sqrt(2.0 * Gam) * Sz01] if Gam > 0 else []
    Ls_free: list[np.ndarray] = []
    steps = max(2, int(np.ceil(T / dt)))
    dt_step = T / steps
    ts = np.arange(steps) * dt_step

    def h_func(time: float) -> np.ndarray:
        return drive_hamiltonian_transmon(time, base)

    def jump_func(time: float) -> list[np.ndarray]:
        return Ls_probe if in_probe(time) else Ls_free

    def evolve_3(rho3_in: np.ndarray) -> np.ndarray:
        rho = rho3_in.astype(complex, copy=True)
        for t in ts:
            rho = propagate_lindblad(rho, t, dt_step, h_func, jump_func, method=method)
        return rho

    # leakage if we start in |+x> within {|0>,|1>}
    psi01 = (1/np.sqrt(2)) * np.array([1,1,0], dtype=complex)
    rho0 = np.outer(psi01, psi01.conj())
    rhoF = evolve_3(rho0)
    p_leak = np.real(np.trace(P2 @ rhoF))

    # Build a superop for the computational subspace via embedding/projection
    basis2 = [
        np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex),
        np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex),
        np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex),
        np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex),
    ]
    columns = []
    for mat in basis2:
        rho3 = np.zeros((3, 3), dtype=complex)
        rho3[0:2, 0:2] = mat
        columns.append(evolve_3(rho3)[0:2, 0:2].reshape(-1))
    super_matrix = np.stack(columns, axis=1)

    def superop_2x2(rho2_in: np.ndarray) -> np.ndarray:
        vec = np.asarray(rho2_in, dtype=complex).reshape(-1)
        return (super_matrix @ vec).reshape(2, 2)

    superop_2x2.super_matrix = super_matrix  # type: ignore[attr-defined]
    return rhoF, float(p_leak), superop_2x2

def sweep_3level(
    base: dict,
    dphi_grid: np.ndarray,
    tau_grid: np.ndarray,
    fidelity_func,  # callable(U_target, superop)->float
    U_target: np.ndarray,
    dt: float = 2e-3,
    method: str = "rk4",
):
    F = np.zeros((len(tau_grid), len(dphi_grid)))
    best = {"F": -1.0, "tau": None, "dphi": None, "p_leak": None}

    for i, tau in enumerate(tau_grid):
        for j, dphi in enumerate(dphi_grid):
            p = dict(base)
            p["tau"] = float(tau)
            p["dphi"] = float(dphi)
            _, p_leak, sop = simulate_period_three_level(p, dt=dt, method=method)
            Fij = fidelity_func(U_target, sop)
            F[i, j] = Fij
            if Fij > best["F"]:
                best.update({"F": Fij, "tau": float(tau), "dphi": float(dphi), "p_leak": float(p_leak)})
    return F, best
