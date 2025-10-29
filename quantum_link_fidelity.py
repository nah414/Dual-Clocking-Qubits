"""Quantum Link Fidelity Simulation utilities and CLI."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class SimulationParameters:
    """Container for BB84-style channel parameters."""

    alpha_db_per_km: float = 0.20  # Fiber attenuation (dB/km)
    det_efficiency: float = 0.8  # Detector efficiency
    dark_counts: float = 50.0  # Dark counts per second
    gate_rate: float = 1e6  # Detection windows per second
    mean_photons: float = 0.1  # Mean photons per pulse
    misalignment_error: float = 0.015
    extra_loss_db: float = 1.0  # Connector + filter loss (dB)

    @property
    def dark_count_probability(self) -> float:
        """Probability of a dark count per detection window."""

        return self.dark_counts / self.gate_rate


DEFAULT_PARAMETERS = SimulationParameters()


# --- Helper functions ---
def db_to_lin(db: float) -> float:
    """Convert decibel value to linear scale."""

    return 10 ** (-db / 10.0)


def link_eta(distance_km: float, params: SimulationParameters = DEFAULT_PARAMETERS) -> float:
    """Linear transmission through fiber for a given distance."""

    total_db = params.alpha_db_per_km * distance_km + params.extra_loss_db
    return db_to_lin(total_db)


def compute_qber(signal_prob: float, noise_prob: float, e_align: float) -> float:
    """Simple quantum bit error rate model."""

    denom = signal_prob + noise_prob
    return (0.5 * noise_prob + e_align * signal_prob) / denom if denom > 0 else 0.5


def fidelity_from_qber(qber: float) -> float:
    """Approximate fidelity as one minus the QBER."""

    return max(0.0, min(1.0, 1.0 - qber))


def signal_detection_probability(distance_km: float, params: SimulationParameters) -> float:
    """Probability that a signal photon is detected for a given distance."""

    eta = link_eta(distance_km, params)
    return 1.0 - math.exp(-params.mean_photons * eta * params.det_efficiency)


def simulate_fidelity(
    distances_km: Iterable[float],
    params: SimulationParameters = DEFAULT_PARAMETERS,
) -> List[float]:
    """Compute the fidelity proxy across a sequence of fiber distances."""

    p_dark = params.dark_count_probability
    results: List[float] = []

    for distance in distances_km:
        p_sig = signal_detection_probability(distance, params)
        qber = compute_qber(p_sig, p_dark, params.misalignment_error)
        results.append(fidelity_from_qber(qber))

    return results


def plot_fidelity(
    distances_km: Sequence[float],
    fidelity: Sequence[float],
    *,
    title: str,
    output_path: Path | None = None,
    show_plot: bool = True,
) -> None:
    """Render the fidelity curve and optionally save the figure."""

    plt.figure()
    plt.plot(distances_km, fidelity, linewidth=2)
    plt.xlabel("Fiber distance (km)")
    plt.ylabel("Fidelity (proxy)")
    plt.title(title)
    plt.grid(True)

    if output_path is not None:
        output_path = output_path.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close()


def parse_arguments() -> argparse.Namespace:
    """Construct the CLI argument parser."""

    parser = argparse.ArgumentParser(description=__doc__ or "Quantum Link Fidelity")
    parser.add_argument(
        "--max-distance",
        type=float,
        default=200.0,
        help="Maximum fiber distance (km) to simulate.",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=5.0,
        help="Distance increment (km).",
    )
    parser.add_argument(
        "--alpha-db-per-km",
        type=float,
        default=DEFAULT_PARAMETERS.alpha_db_per_km,
        help="Fiber attenuation in dB/km.",
    )
    parser.add_argument(
        "--det-efficiency",
        type=float,
        default=DEFAULT_PARAMETERS.det_efficiency,
        help="Detector efficiency.",
    )
    parser.add_argument(
        "--dark-counts",
        type=float,
        default=DEFAULT_PARAMETERS.dark_counts,
        help="Dark counts per second.",
    )
    parser.add_argument(
        "--gate-rate",
        type=float,
        default=DEFAULT_PARAMETERS.gate_rate,
        help="Detection windows per second.",
    )
    parser.add_argument(
        "--mean-photons",
        type=float,
        default=DEFAULT_PARAMETERS.mean_photons,
        help="Mean photons per pulse.",
    )
    parser.add_argument(
        "--misalignment-error",
        type=float,
        default=DEFAULT_PARAMETERS.misalignment_error,
        help="Misalignment error probability.",
    )
    parser.add_argument(
        "--extra-loss-db",
        type=float,
        default=DEFAULT_PARAMETERS.extra_loss_db,
        help="Additional connector/filter loss in dB.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the plot as an image.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the plot window.",
    )

    return parser.parse_args()


def build_parameters(args: argparse.Namespace) -> SimulationParameters:
    """Create a parameter set from parsed CLI arguments."""

    return replace(
        DEFAULT_PARAMETERS,
        alpha_db_per_km=args.alpha_db_per_km,
        det_efficiency=args.det_efficiency,
        dark_counts=args.dark_counts,
        gate_rate=args.gate_rate,
        mean_photons=args.mean_photons,
        misalignment_error=args.misalignment_error,
        extra_loss_db=args.extra_loss_db,
    )


def main() -> None:
    args = parse_arguments()
    params = build_parameters(args)

    if args.step <= 0:
        raise ValueError("Step size must be positive.")
    if args.max_distance < 0:
        raise ValueError("Maximum distance must be non-negative.")

    distances = np.arange(0.0, args.max_distance + args.step / 2.0, args.step)
    fidelity = simulate_fidelity(distances, params)

    plot_fidelity(
        distances,
        fidelity,
        title="Quantum Link Fidelity vs Distance",
        output_path=args.output,
        show_plot=not args.no_show,
    )


if __name__ == "__main__":
    main()
