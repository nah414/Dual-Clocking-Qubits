# Dual-Drive + Probe Timing Optimizer — Core utilities
# Version: v0.2 (2025-11-09, America/Chicago)

from __future__ import annotations
import numpy as np
from dataclasses import dataclass

# ---------- Pauli / basic operators ----------
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)


def pauli():
    return sx, sy, sz, I2


def _lindblad_rhs(rho: np.ndarray, H: np.ndarray, Ls: list[np.ndarray]) -> np.ndarray:
    """Compute the Lindblad right-hand side at a fixed Hamiltonian and jump set."""
    drho = -1j * (H @ rho - rho @ H)
    for L in Ls:
        LdL = L.conj().T @ L
        drho += L @ rho @ L.conj().T - 0.5 * (LdL @ rho + rho @ LdL)
    return drho


# ---------- Lindblad integrators ----------
def lindblad_step(rho: np.ndarray, H: np.ndarray, Ls: list[np.ndarray], dt: float) -> np.ndarray:
    """Single-step explicit Euler update (retained for backwards compatibility)."""
    return rho + dt * _lindblad_rhs(rho, H, Ls)


def propagate_lindblad(
    rho: np.ndarray,
    t: float,
    dt: float,
    h_func,
    jump_func,
    method: str = "rk4",
) -> np.ndarray:
    """Propagate ``rho`` forward by ``dt`` using the chosen Runge–Kutta method."""

    if method == "euler":
        H = h_func(t)
        Ls = jump_func(t)
        return rho + dt * _lindblad_rhs(rho, H, Ls)

    if method != "rk4":
        raise ValueError(f"Unsupported Lindblad integrator '{method}'")

    def rhs(state: np.ndarray, time: float) -> np.ndarray:
        return _lindblad_rhs(state, h_func(time), jump_func(time))

    k1 = rhs(rho, t)
    k2 = rhs(rho + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = rhs(rho + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = rhs(rho + dt * k3, t + dt)
    return rho + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

# ---------- Average gate fidelity (Monte Carlo over pure states) ----------
@dataclass(frozen=True)
class MonteCarloBatch:
    """Precomputed Bloch-sphere samples and target outputs for fidelity estimates."""

    rhos: np.ndarray
    rho_targets: np.ndarray


def _generate_bloch_samples(trials: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vecs = rng.normal(size=(trials, 3))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    theta = np.arccos(np.clip(vecs[:, 2], -1.0, 1.0))
    phi = np.arctan2(vecs[:, 1], vecs[:, 0])

    cos_half = np.cos(0.5 * theta)
    sin_half = np.sin(0.5 * theta)
    phase = np.exp(0.5j * phi)

    psi = np.empty((trials, 2), dtype=complex)
    psi[:, 0] = cos_half * phase.conj()
    psi[:, 1] = sin_half * phase

    return psi[:, :, None] * psi.conj()[:, None, :]


def prepare_monte_carlo_batch(
    U_target,
    trials: int = 200,
    seed: int = 0,
) -> MonteCarloBatch:
    """Precompute Bloch samples and their ideal outputs for repeated fidelity calls."""

    rhos = _generate_bloch_samples(trials, seed)
    U = np.asarray(U_target)
    U_dag = U.conj().T
    rho_targets = np.einsum("ab,sbc,cd->sad", U, rhos, U_dag)
    return MonteCarloBatch(rhos=rhos, rho_targets=rho_targets)


def average_gate_fidelity_montecarlo(
    U_target,
    superop,
    trials: int = 200,
    seed: int = 0,
    batch: MonteCarloBatch | None = None,
) -> float:
    """Estimate the average state fidelity via Monte Carlo sampling on the Bloch sphere."""

    if batch is None:
        batch = prepare_monte_carlo_batch(U_target, trials=trials, seed=seed)
    rhos = batch.rhos
    rho_targets = batch.rho_targets

    if hasattr(superop, "super_matrix"):
        S = superop.super_matrix
        rho_evolved = (rhos.reshape(rhos.shape[0], -1) @ S.T).reshape(-1, 2, 2)
    else:
        rho_evolved = np.stack([superop(rho) for rho in rhos], axis=0)

    overlaps = np.einsum("sij,sji->s", rho_targets, rho_evolved).real
    return float(np.mean(overlaps))

# ---------- Two-tone, near-RWA rotating-frame Hamiltonian ----------
def drive_hamiltonian_two_tone(t: float, params: dict) -> np.ndarray:
    """H(t) = 0.5*Delta_eff*sz + 0.5*(Ω1 e^{i(Δ1 t+φ1)} + Ω2 e^{i(Δ2 t+φ2+δφ)}) σ_+ + h.c."""
    sx_, sy_, sz_, _ = pauli()
    Delta_eff = params.get("Delta_eff", 0.0)
    Om1 = params["Omega1"](t)
    Om2 = params["Omega2"](t)
    D1 = params.get("Delta1", 0.0)
    D2 = params.get("Delta2", 0.0)
    phi1 = params.get("phi1", 0.0)
    phi2 = params.get("phi2", 0.0)
    dphi = params.get("dphi", 0.0)  # includes timing slip ω2 Δt + Δφ

    hx = 0.5 * (Om1 * np.cos(D1 * t + phi1) + Om2 * np.cos(D2 * t + phi2 + dphi))
    hy = 0.5 * (Om1 * np.sin(D1 * t + phi1) + Om2 * np.sin(D2 * t + phi2 + dphi))
    H = 0.5 * Delta_eff * sz_ + hx * sx_ + hy * sy_
    return H

# ---------- Piecewise measurement window check ----------
def make_in_probe(tau: float, tau_p: float):
    def _f(t: float) -> bool:
        return (t >= tau) and (t < tau + tau_p)
    return _f

# ---------- Build superoperator by linearity on density matrices ----------
def build_period_superop_twolevel(
    params: dict,
    dt: float = 2e-3,
    method: str = "rk4",
):
    T = params["T"]
    tau = params["tau"]
    tau_p = params["tau_p"]
    Gam = params.get("Gamma_m", 0.0)
    in_probe = make_in_probe(tau, tau_p)
    Ls_probe = [np.sqrt(2.0 * Gam) * sz] if Gam > 0 else []
    Ls_free: list[np.ndarray] = []

    steps = max(2, int(np.ceil(T / dt)))
    dt_step = T / steps
    ts = np.arange(steps) * dt_step

    def h_func(time: float) -> np.ndarray:
        return drive_hamiltonian_two_tone(time, params)

    def jump_func(time: float) -> list[np.ndarray]:
        return Ls_probe if in_probe(time) else Ls_free

    def evolve(rho_in: np.ndarray) -> np.ndarray:
        rho = rho_in.astype(complex, copy=True)
        for t in ts:
            rho = propagate_lindblad(rho, t, dt_step, h_func, jump_func, method=method)
        return rho

    basis = [
        np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex),
        np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex),
        np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex),
        np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex),
    ]
    columns = [evolve(mat).reshape(-1) for mat in basis]
    super_matrix = np.stack(columns, axis=1)

    def apply(rho_in: np.ndarray) -> np.ndarray:
        vec = np.asarray(rho_in, dtype=complex).reshape(-1)
        return (super_matrix @ vec).reshape(2, 2)

    apply.super_matrix = super_matrix  # type: ignore[attr-defined]
    return apply
