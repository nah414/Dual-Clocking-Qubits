"""Runner: two-level sweep and local refinement."""

import numpy as np
from dualdrive.two_level import make_envelopes, sweep_2level, refine_local


def main() -> None:
    """Execute a coarse sweep followed by local refinement using default params."""
    # Base params
    base = dict(
        T=1.0,              # period
        tau=0.4,            # probe start (will be swept)
        tau_p=0.1,          # probe duration
        Gamma_m=0.8,        # dephasing during probe
        Delta_eff=0.0,      # effective detuning (post Stark/BS comp)
        Delta1=0.0, Delta2=0.0,
        phi1=0.0, phi2=0.0,
        dphi=0.0            # relative phase (will be swept)
    )

    # Envelopes
    Omega1, Omega2 = make_envelopes(T=base["T"], Om1_val=1.0, Om2_val=0.9)
    base["Omega1"] = Omega1
    base["Omega2"] = Omega2

    # Grids
    dphi_grid = np.linspace(-np.pi, np.pi, 49)
    tau_grid = np.linspace(0.0, base["T"] - base["tau_p"] - 1e-6, 49)

    # Sweep
    F, best = sweep_2level(base, dphi_grid, tau_grid, dt=2e-3, trials_mc=200, seed=0)
    print(f"[Coarse] Best: F={best['F']:.6f}, tau={best['tau']:.6f}, dphi={best['dphi']:.6f}")

    # Refine
    refined = refine_local(
        base,
        tau0=best["tau"],
        dphi0=best["dphi"],
        dt=2e-3,
        steps=24,
        eps_tau=0.01,
        eps_phi=0.05,
        trials_mc=400,
        seed=1,
    )
    print(
        f"[Refined] Best: F={refined['F']:.6f}, tau={refined['tau']:.6f}, dphi={refined['dphi']:.6f}"
    )


if __name__ == "__main__":
    main()
