"""Probe Calibration System for Quantum Readout Optimization.

This module provides a framework for calibrating quantum readout probes across
multi-dimensional parameter spaces, analyzing measurement quality metrics, and
identifying optimal configurations via Pareto optimization.

Author: Refined implementation
Date: 2025
"""

from __future__ import annotations

import json
import math
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING

try:  # pragma: no cover - optional dependency resolution
    import numpy as _np  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised when numpy missing
    _np = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - typing aid only
    import numpy as np


def _require_numpy() -> Any:
    """Return the NumPy module, raising a helpful error if unavailable."""

    if _np is None:  # pragma: no cover - environment without numpy
        raise ImportError(
            "dual_clocking.calibration requires NumPy. Install numpy to use "
            "probe calibration utilities."
        )
    return _np


# ============================================================================
# Data Structures
# ============================================================================


@dataclass(frozen=True)
class ProbePoint:
    """Immutable probe configuration point."""

    amp: float
    dur_ns: int
    detuning_MHz: float
    rel_phase_deg: float

    def __post_init__(self) -> None:  # pragma: no cover - simple validation
        """Validate parameters at construction time."""
        if self.amp < 0:
            raise ValueError(f"Amplitude must be non-negative, got {self.amp}")
        if self.dur_ns <= 0:
            raise ValueError(f"Duration must be positive, got {self.dur_ns}")
        if not (0 <= self.rel_phase_deg < 360):
            warnings.warn(f"Phase {self.rel_phase_deg} outside [0, 360), wrapping")
            object.__setattr__(self, "rel_phase_deg", self.rel_phase_deg % 360)

    def to_dict(self) -> Dict[str, Union[float, int]]:
        """Convert to dictionary for serialization."""
        return {
            "amp": float(self.amp),
            "dur_ns": int(self.dur_ns),
            "detuning_MHz": float(self.detuning_MHz),
            "rel_phase_deg": float(self.rel_phase_deg),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Union[float, int]]) -> "ProbePoint":
        """Reconstruct from dictionary."""
        return cls(
            amp=float(data["amp"]),
            dur_ns=int(data["dur_ns"]),
            detuning_MHz=float(data["detuning_MHz"]),
            rel_phase_deg=float(data["rel_phase_deg"]),
        )

    def __str__(self) -> str:  # pragma: no cover - simple formatting
        """Human-readable representation."""
        return (
            f"ProbePoint(amp={self.amp:.3f}, dur={self.dur_ns}ns, "
            f"det={self.detuning_MHz:.2f}MHz, φ={self.rel_phase_deg:.1f}°)"
        )


@dataclass
class ProbeData:
    """Container for probe measurement results."""

    IQ: "np.ndarray"
    labels: "np.ndarray"
    post_gate_scores: "np.ndarray"

    def __post_init__(self) -> None:
        """Validate data consistency."""
        np = _require_numpy()

        iq_arr = np.asarray(self.IQ)
        labels_arr = np.asarray(self.labels)
        post_arr = np.asarray(self.post_gate_scores)

        n = len(iq_arr)
        if len(labels_arr) != n or len(post_arr) != n:
            raise ValueError(
                "Array length mismatch: "
                f"IQ={len(iq_arr)}, labels={len(labels_arr)}, "
                f"post_gate={len(post_arr)}"
            )

        if not np.iscomplexobj(iq_arr):
            iq_arr = iq_arr.astype(complex)

        object.__setattr__(self, "IQ", iq_arr)
        object.__setattr__(self, "labels", labels_arr)
        object.__setattr__(self, "post_gate_scores", post_arr)

    def split_by_state(self) -> Dict[str, Tuple["np.ndarray", "np.ndarray"]]:
        """Split data by prepared state."""
        np = _require_numpy()
        result: Dict[str, Tuple["np.ndarray", "np.ndarray"]] = {}
        unique_states = np.unique(self.labels)
        for state in unique_states:
            mask = self.labels == state
            result[str(state)] = (self.IQ[mask], self.post_gate_scores[mask])
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "IQ_real": self.IQ.real.tolist(),
            "IQ_imag": self.IQ.imag.tolist(),
            "labels": self.labels.tolist(),
            "post_gate_scores": self.post_gate_scores.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProbeData":
        """Reconstruct from dictionary."""
        np = _require_numpy()
        iq = np.array(data["IQ_real"]) + 1j * np.array(data["IQ_imag"])
        return cls(
            IQ=iq,
            labels=np.array(data["labels"]),
            post_gate_scores=np.array(data["post_gate_scores"]),
        )


@dataclass
class CalibrationMetrics:
    """Aggregated metrics for a probe configuration."""

    mutual_info: float
    qnd_score: float
    mean_fidelity: float
    snr: float
    separation: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "mutual_info": float(self.mutual_info),
            "qnd_score": float(self.qnd_score),
            "mean_fidelity": float(self.mean_fidelity),
            "snr": float(self.snr),
            "separation": float(self.separation),
        }

    def score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Compute weighted composite score."""
        if weights is None:
            weights = {
                "mutual_info": 0.3,
                "qnd_score": 0.25,
                "mean_fidelity": 0.25,
                "snr": 0.1,
                "separation": 0.1,
            }

        return (
            weights.get("mutual_info", 0.0) * self.mutual_info
            + weights.get("qnd_score", 0.0) * self.qnd_score
            + weights.get("mean_fidelity", 0.0) * self.mean_fidelity
            + weights.get("snr", 0.0) * self.snr
            + weights.get("separation", 0.0) * self.separation
        )


# ============================================================================
# Probe Calibrator
# ============================================================================


class ProbeCalibrator:
    """Calibrates readout probes across multi-dimensional parameter space."""

    def __init__(
        self,
        hw: Any,
        grid: Dict[str, Sequence[Union[int, float]]],
        repeats: int,
    ) -> None:
        self.hw = hw
        self.grid = self._validate_grid(grid)
        self.repeats = repeats
        self._validate_repeats()
        self._measurement_count = 0

    def _validate_grid(self, grid: Dict[str, Sequence[Union[int, float]]]) -> Dict[str, "np.ndarray"]:
        np = _require_numpy()
        required = {"amp", "dur_ns", "detuning_MHz", "rel_phase_deg"}
        if not required.issubset(grid.keys()):
            missing = required - set(grid.keys())
            raise ValueError(f"Grid missing required keys: {missing}")

        validated: Dict[str, "np.ndarray"] = {}
        for key in required:
            arr = np.asarray(grid[key])
            if arr.size == 0:
                raise ValueError(f"Grid parameter '{key}' cannot be empty")
            validated[key] = arr
        return validated

    def _validate_repeats(self) -> None:
        if self.repeats < 1:
            raise ValueError(f"Repeats must be >= 1, got {self.repeats}")
        if self.repeats > 10_000:
            warnings.warn(f"Very large repeat count: {self.repeats}")

    def total_measurements(self, states: Sequence[str]) -> int:
        n_points = math.prod(len(values) for values in self.grid.values())
        return n_points * len(tuple(states)) * int(self.repeats)

    def grid_shape(self) -> Tuple[int, ...]:
        return tuple(len(self.grid[key]) for key in ["amp", "dur_ns", "detuning_MHz", "rel_phase_deg"])

    def acquire(
        self,
        states: Sequence[str] = ("Z0", "Z1", "X+", "Y+"),
        progress_callback: Optional[Callable[[int, int], None]] = None,
        save_intermediate: Optional[Path] = None,
    ) -> Dict[ProbePoint, ProbeData]:
        np = _require_numpy()
        data: Dict[ProbePoint, ProbeData] = {}
        states_tuple = tuple(states)
        total = self.total_measurements(states_tuple)
        self._measurement_count = 0

        for amp in self.grid["amp"]:
            for dur in self.grid["dur_ns"]:
                for detuning in self.grid["detuning_MHz"]:
                    for phase in self.grid["rel_phase_deg"]:
                        point = ProbePoint(float(amp), int(dur), float(detuning), float(phase))
                        n_shots = len(states_tuple) * self.repeats
                        iq = np.zeros(n_shots, dtype=complex)
                        labels = np.empty(n_shots, dtype=object)
                        post_scores = np.zeros(n_shots, dtype=float)

                        idx = 0
                        for state in states_tuple:
                            for _ in range(self.repeats):
                                self.hw.prepare(state)
                                self.hw.apply_probe(amp, dur, detuning, phase)

                                iq_raw = self.hw.measure_IQ()
                                if isinstance(iq_raw, (tuple, list)) and len(iq_raw) >= 2:
                                    iq[idx] = complex(iq_raw[0], iq_raw[1])
                                elif np.iscomplexobj(iq_raw):
                                    iq[idx] = complex(iq_raw)
                                else:
                                    iq[idx] = complex(float(iq_raw), 0.0)

                                labels[idx] = state
                                post_scores[idx] = float(self.hw.run_post_gate_and_benchmark())

                                idx += 1
                                self._measurement_count += 1

                                if progress_callback is not None:
                                    progress_callback(self._measurement_count, total)

                        data[point] = ProbeData(IQ=iq, labels=labels, post_gate_scores=post_scores)

                        if save_intermediate is not None and len(data) % 10 == 0:
                            self.save_data(data, save_intermediate)

        return data

    def save_data(self, data: Dict[ProbePoint, ProbeData], filepath: Path) -> None:
        filepath = Path(filepath)
        if filepath.suffix == ".pkl":
            with filepath.open("wb") as handle:
                pickle.dump(data, handle)
        elif filepath.suffix == ".json":
            json_payload = {
                json.dumps(point.to_dict(), sort_keys=True): probe_data.to_dict()
                for point, probe_data in data.items()
            }
            with filepath.open("w", encoding="utf-8") as handle:
                json.dump(json_payload, handle, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

    @staticmethod
    def load_data(filepath: Path) -> Dict[ProbePoint, ProbeData]:
        filepath = Path(filepath)
        if filepath.suffix == ".pkl":
            with filepath.open("rb") as handle:
                return pickle.load(handle)
        if filepath.suffix == ".json":
            with filepath.open("r", encoding="utf-8") as handle:
                raw: Dict[str, Any] = json.load(handle)
            out: Dict[ProbePoint, ProbeData] = {}
            for key, payload in raw.items():
                point = ProbePoint.from_dict(json.loads(key))
                out[point] = ProbeData.from_dict(payload)
            return out
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


# ============================================================================
# Analysis Functions
# ============================================================================


def mutual_info_bits(
    llr: "np.ndarray",
    labels: "np.ndarray",
    n_bins: Optional[int] = None,
    min_samples: int = 10,
) -> float:
    """Estimate mutual information I(measurement; state) in bits."""

    np = _require_numpy()
    z_mask = (labels == "Z0") | (labels == "Z1")
    if int(np.sum(z_mask)) < min_samples:
        warnings.warn(f"Insufficient data for MI estimation: {np.sum(z_mask)} < {min_samples}")
        return 0.0

    measurements = llr[z_mask]
    states = (labels[z_mask] == "Z1").astype(int)

    if len(np.unique(states)) < 2:
        warnings.warn("Only one state present, MI = 0")
        return 0.0

    if n_bins is None:
        n_bins = max(10, int(np.ceil(np.log2(len(measurements)) + 1)))

    bins = np.linspace(float(measurements.min()), float(measurements.max()), n_bins + 1)

    eps = 1e-12
    hist0, _ = np.histogram(measurements[states == 0], bins=bins)
    hist1, _ = np.histogram(measurements[states == 1], bins=bins)

    p_m_s0 = (hist0 + eps) / (hist0.sum() + eps * len(hist0))
    p_m_s1 = (hist1 + eps) / (hist1.sum() + eps * len(hist1))

    p_m = 0.5 * (p_m_s0 + p_m_s1)

    mi_s0 = np.sum(p_m_s0 * np.log2(p_m_s0 / p_m))
    mi_s1 = np.sum(p_m_s1 * np.log2(p_m_s1 / p_m))

    mi = 0.5 * (mi_s0 + mi_s1)
    return float(np.clip(mi, 0.0, 1.0))


def qnd_correlation(m1: "np.ndarray", m2: "np.ndarray", method: str = "pearson") -> float:
    """Quantify quantum non-demolition (QND) quality via measurement correlation."""

    np = _require_numpy()
    if len(m1) != len(m2):
        raise ValueError(f"Array length mismatch: {len(m1)} != {len(m2)}")
    if len(m1) < 2:
        warnings.warn("Insufficient data for correlation")
        return 0.0

    mask = np.isfinite(m1) & np.isfinite(m2)
    m1 = m1[mask]
    m2 = m2[mask]

    if len(m1) < 2:
        return 0.0

    if method == "spearman":
        try:
            from scipy.stats import spearmanr

            corr, _ = spearmanr(m1, m2)
            return float(corr) if np.isfinite(corr) else 0.0
        except ImportError:  # pragma: no cover - optional dependency
            warnings.warn("scipy not available, falling back to Pearson")
            method = "pearson"

    if method == "pearson":
        m1_centered = m1 - np.mean(m1)
        m2_centered = m2 - np.mean(m2)
        numerator = float(np.dot(m1_centered, m2_centered))
        denominator = float(np.linalg.norm(m1_centered) * np.linalg.norm(m2_centered))
        if denominator < 1e-12:
            return 0.0
        corr = numerator / denominator
        return float(np.clip(corr, -1.0, 1.0))

    raise ValueError(f"Unknown correlation method: {method}")


def compute_snr(iq_data: "np.ndarray", labels: "np.ndarray") -> float:
    """Compute signal-to-noise ratio from IQ data."""

    np = _require_numpy()
    z0_mask = labels == "Z0"
    z1_mask = labels == "Z1"

    if not (np.any(z0_mask) and np.any(z1_mask)):
        return 0.0

    z0_data = iq_data[z0_mask]
    z1_data = iq_data[z1_mask]

    mean_0 = np.mean(z0_data)
    mean_1 = np.mean(z1_data)
    signal = np.abs(mean_1 - mean_0)

    noise_0 = np.std(z0_data)
    noise_1 = np.std(z1_data)
    noise = 0.5 * (noise_0 + noise_1)

    if noise < 1e-12:
        return 0.0

    return float(signal / noise)


def compute_separation(iq_data: "np.ndarray", labels: "np.ndarray") -> float:
    """Compute normalized separation in IQ plane between Z0 and Z1."""

    np = _require_numpy()
    z0_mask = labels == "Z0"
    z1_mask = labels == "Z1"

    if not (np.any(z0_mask) and np.any(z1_mask)):
        return 0.0

    z0_data = iq_data[z0_mask]
    z1_data = iq_data[z1_mask]

    mean_diff = np.abs(np.mean(z1_data) - np.mean(z0_data))
    pooled_var = 0.5 * (np.var(z0_data) + np.var(z1_data))
    pooled_std = np.sqrt(pooled_var)

    if pooled_std < 1e-12:
        return 0.0

    return float(mean_diff / pooled_std)


def compute_metrics(probe_data: ProbeData, qnd_stride: int = 1) -> CalibrationMetrics:
    """Compute all calibration metrics for a probe configuration."""

    np = _require_numpy()
    llr = np.real(probe_data.IQ)
    mi = mutual_info_bits(llr, probe_data.labels)

    state_data = probe_data.split_by_state()
    qnd_scores: List[float] = []
    for iq, _post in state_data.values():
        if len(iq) > qnd_stride:
            llr_state = np.real(iq)
            qnd = qnd_correlation(llr_state[:-qnd_stride], llr_state[qnd_stride:])
            qnd_scores.append(qnd)
    qnd_score = float(np.mean(qnd_scores)) if qnd_scores else 0.0

    mean_fidelity = float(np.mean(probe_data.post_gate_scores))
    snr = compute_snr(probe_data.IQ, probe_data.labels)
    separation = compute_separation(probe_data.IQ, probe_data.labels)

    return CalibrationMetrics(
        mutual_info=mi,
        qnd_score=qnd_score,
        mean_fidelity=mean_fidelity,
        snr=snr,
        separation=separation,
    )


def pareto_front(points: Sequence[ProbePoint], metrics: "np.ndarray", maximize_mask: "np.ndarray") -> "np.ndarray":
    """Find Pareto-optimal (non-dominated) points in multi-objective space."""

    np = _require_numpy()
    if len(points) != metrics.shape[0]:
        raise ValueError(
            f"Number of points ({len(points)}) must match metrics rows ({metrics.shape[0]})"
        )
    if len(maximize_mask) != metrics.shape[1]:
        raise ValueError(
            f"Maximize mask length ({len(maximize_mask)}) must match metrics columns ({metrics.shape[1]})"
        )

    metric_matrix = metrics.astype(float).copy()
    for idx, should_maximize in enumerate(maximize_mask):
        if not should_maximize:
            metric_matrix[:, idx] = -metric_matrix[:, idx]

    n_points = metric_matrix.shape[0]
    is_dominated = np.zeros(n_points, dtype=bool)

    for i in range(n_points):
        if is_dominated[i]:
            continue
        weakly_dominates = np.all(metric_matrix[i] >= metric_matrix, axis=1)
        strictly_dominates = np.any(metric_matrix[i] > metric_matrix, axis=1)
        dominates = weakly_dominates & strictly_dominates
        dominates[i] = False
        is_dominated |= dominates

    return np.where(~is_dominated)[0]


def analyze_probe_sweep(
    data: Dict[ProbePoint, ProbeData],
    qnd_threshold: float = 0.8,
    mi_threshold: float = 0.5,
    fidelity_threshold: float = 0.95,
    metric_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Comprehensive analysis of probe sweep results."""

    np = _require_numpy()
    probe_points: List[ProbePoint] = []
    metrics: List[CalibrationMetrics] = []

    for point, probe_data in data.items():
        probe_points.append(point)
        metrics.append(compute_metrics(probe_data))

    if not metrics:
        return {
            "probe_points": [],
            "metrics": [],
            "pareto_indices": np.array([], dtype=int),
            "best_overall": None,
            "best_mi": None,
            "best_qnd": None,
            "best_fidelity": None,
            "best_snr": None,
            "recommendations": {
                "total_points": 0,
                "pareto_count": 0,
                "threshold_passing": 0,
            },
        }

    mi_scores = np.array([m.mutual_info for m in metrics])
    qnd_scores = np.array([m.qnd_score for m in metrics])
    fidelities = np.array([m.mean_fidelity for m in metrics])
    snr_scores = np.array([m.snr for m in metrics])
    separations = np.array([m.separation for m in metrics])

    composite_scores = np.array([m.score(metric_weights) for m in metrics])

    metric_matrix = np.column_stack((mi_scores, qnd_scores, fidelities, snr_scores, separations))
    maximize_mask = np.ones(metric_matrix.shape[1], dtype=bool)
    pareto_indices = pareto_front(probe_points, metric_matrix, maximize_mask)

    best_overall = int(np.argmax(composite_scores))
    best_mi = int(np.argmax(mi_scores))
    best_qnd = int(np.argmax(qnd_scores))
    best_fidelity = int(np.argmax(fidelities))
    best_snr = int(np.argmax(snr_scores))

    passing_indices = np.where(
        (mi_scores >= mi_threshold)
        & (qnd_scores >= qnd_threshold)
        & (fidelities >= fidelity_threshold)
    )[0]

    recommendations = {
        "total_points": len(metrics),
        "pareto_count": int(len(pareto_indices)),
        "threshold_passing": int(len(passing_indices)),
        "mean_mi": float(np.mean(mi_scores)),
        "mean_qnd": float(np.mean(qnd_scores)),
        "mean_fidelity": float(np.mean(fidelities)),
        "best_candidates": passing_indices.tolist(),
    }

    return {
        "probe_points": probe_points,
        "metrics": metrics,
        "pareto_indices": pareto_indices,
        "best_overall": best_overall,
        "best_mi": best_mi,
        "best_qnd": best_qnd,
        "best_fidelity": best_fidelity,
        "best_snr": best_snr,
        "recommendations": recommendations,
    }
