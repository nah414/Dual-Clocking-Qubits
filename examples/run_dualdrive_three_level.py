"""Runner: three-level (transmon) sweep with leakage + optional DRAG."""

import numpy as np
from dualdrive.core import average_gate_fidelity_montecarlo, prepare_monte_carlo_batch
from dualdrive.three_level import (
    make_envelopes_flat,
    simulate_period_three_level,
    sweep_3level,
)


def target_unitary_identity() -> np.ndarray:
    return np.eye(2, dtype=complex)


def main() -> None:
    """Execute a leakage-aware sweep using the three-level transmon model."""
    base = dict(
        T=1.0,
        tau=0.35,
        tau_p=0.10,
        Gamma_m=0.6,
        w01=0.0,
        w12=-0.25,
        Delta1=0.0,
        Delta2=0.0,
        phi1=0.0,
        phi2=0.0,
        dphi=0.0,
        ratio_12=np.sqrt(2.0),
    )

    env = make_envelopes_flat(T=base["T"], Om1=1.0, Om2=0.9, beta_drag=-1.0 / 0.25)
    base.update(env)

    rhoF, p_leak, sop = simulate_period_three_level(base, dt=2e-3)
    print(f"[Sanity] Single-period leakage p2 = {p_leak:.6f}")
    del rhoF, sop

    dphi_grid = np.linspace(-np.pi, np.pi, 41)
    tau_grid = np.linspace(0.0, base["T"] - base["tau_p"] - 1e-6, 41)
    U_tgt = target_unitary_identity()

    batch = prepare_monte_carlo_batch(U_tgt, trials=200, seed=0)

    fidelity_func = lambda U, superop: average_gate_fidelity_montecarlo(  # noqa: E731
        U, superop, batch=batch
    )
    F, best = sweep_3level(base, dphi_grid, tau_grid, fidelity_func, U_tgt, dt=2e-3)
    print(
        "[Coarse-3lvl] Best: "
        f"F={best['F']:.6f}, tau={best['tau']:.6f}, dphi={best['dphi']:.6f}, leak≈{best['p_leak']:.6f}"
    )


if __name__ == "__main__":
    main()
