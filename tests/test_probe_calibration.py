from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from dual_clocking.calibration import (
    CalibrationMetrics,
    ProbeCalibrator,
    ProbeData,
    ProbePoint,
    analyze_probe_sweep,
    compute_metrics,
    compute_separation,
    compute_snr,
    mutual_info_bits,
    pareto_front,
    qnd_correlation,
)


class DummyHardware:
    def __init__(self) -> None:
        self.measure_count = 0
        self.prepared_states = []
        self.probe_calls = []

    def prepare(self, state: str) -> None:
        self.prepared_states.append(state)

    def apply_probe(self, amp, dur_ns, detuning_MHz, rel_phase_deg) -> None:
        self.probe_calls.append((float(amp), float(dur_ns), float(detuning_MHz), float(rel_phase_deg)))

    def measure_IQ(self):
        self.measure_count += 1
        return complex(self.measure_count, -self.measure_count / 2)

    def run_post_gate_and_benchmark(self) -> float:
        return 0.9 + 0.01 * (self.measure_count % 3)


def test_probe_point_validation_and_wrapping():
    with pytest.raises(ValueError):
        ProbePoint(amp=-0.1, dur_ns=10, detuning_MHz=0.0, rel_phase_deg=0.0)

    with pytest.raises(ValueError):
        ProbePoint(amp=0.1, dur_ns=0, detuning_MHz=0.0, rel_phase_deg=0.0)

    with pytest.warns(UserWarning):
        point = ProbePoint(amp=0.2, dur_ns=10, detuning_MHz=0.0, rel_phase_deg=400.0)
    assert pytest.approx(40.0) == point.rel_phase_deg


def test_probe_data_serialization_roundtrip():
    iq = np.array([1 + 2j, 3 + 4j])
    labels = np.array(["Z0", "Z1"])
    post_scores = np.array([0.95, 0.9])
    data = ProbeData(IQ=iq, labels=labels, post_gate_scores=post_scores)

    split = data.split_by_state()
    assert set(split.keys()) == {"Z0", "Z1"}
    np.testing.assert_allclose(split["Z0"][0], [1 + 2j])

    payload = data.to_dict()
    restored = ProbeData.from_dict(payload)
    np.testing.assert_allclose(restored.IQ, iq)
    np.testing.assert_allclose(restored.post_gate_scores, post_scores)


def test_metric_utilities_behave_reasonably():
    iq = np.array([1 + 0j, 1.1 + 0j, 2 + 0j, 2.1 + 0j])
    labels = np.array(["Z0", "Z0", "Z1", "Z1"])
    post_scores = np.array([0.95, 0.94, 0.93, 0.96])
    data = ProbeData(IQ=iq, labels=labels, post_gate_scores=post_scores)

    metrics = compute_metrics(data)
    assert isinstance(metrics, CalibrationMetrics)
    assert metrics.mutual_info >= 0
    assert metrics.qnd_score >= -1
    assert metrics.mean_fidelity == pytest.approx(np.mean(post_scores))
    assert compute_snr(iq, labels) > 0
    assert compute_separation(iq, labels) > 0

    corr = qnd_correlation(np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.25, 0.35]))
    assert -1.0 <= corr <= 1.0

    mi = mutual_info_bits(np.real(iq), labels)
    assert 0.0 <= mi <= 1.0


def test_pareto_front_identifies_non_dominated_points():
    points = [
        ProbePoint(0.1, 10, -1.0, 0.0),
        ProbePoint(0.2, 10, -1.0, 90.0),
        ProbePoint(0.3, 10, -1.0, 180.0),
    ]
    metrics = np.array(
        [
            [0.9, 0.8],
            [0.85, 0.85],
            [0.7, 0.9],
        ]
    )
    mask = np.array([True, True])
    indices = pareto_front(points, metrics, mask)
    assert set(indices.tolist()) == {0, 1, 2}


def test_probe_calibrator_acquire_and_persistence(tmp_path: Path):
    grid = {
        "amp": [0.1],
        "dur_ns": [40],
        "detuning_MHz": [0.0],
        "rel_phase_deg": [0.0, 180.0],
    }
    hw = DummyHardware()
    calibrator = ProbeCalibrator(hw=hw, grid=grid, repeats=2)

    progress_updates = []
    data = calibrator.acquire(
        states=("Z0", "Z1"),
        progress_callback=lambda cur, total: progress_updates.append((cur, total)),
    )

    assert len(data) == 2
    for probe_data in data.values():
        assert probe_data.IQ.shape == (4,)
        assert probe_data.labels.shape == (4,)
        assert probe_data.post_gate_scores.shape == (4,)

    assert progress_updates
    assert progress_updates[-1][0] == progress_updates[-1][1]

    pkl_path = tmp_path / "calibration.pkl"
    calibrator.save_data(data, pkl_path)
    loaded_pkl = ProbeCalibrator.load_data(pkl_path)
    assert set(loaded_pkl.keys()) == set(data.keys())

    json_path = tmp_path / "calibration.json"
    calibrator.save_data(data, json_path)
    loaded_json = ProbeCalibrator.load_data(json_path)
    assert set(loaded_json.keys()) == set(data.keys())


def test_analyze_probe_sweep_returns_summary():
    point_a = ProbePoint(0.1, 20, 0.0, 0.0)
    point_b = ProbePoint(0.2, 20, 0.0, 90.0)

    iq_a = np.array([1 + 0j, 1.1 + 0j, 1.2 + 0j, 1.3 + 0j])
    iq_b = np.array([2 + 0j, 2.1 + 0j, 2.2 + 0j, 2.3 + 0j])
    labels = np.array(["Z0", "Z0", "Z1", "Z1"])
    post_a = np.array([0.96, 0.95, 0.94, 0.95])
    post_b = np.array([0.97, 0.96, 0.95, 0.96])

    data = {
        point_a: ProbeData(IQ=iq_a, labels=labels, post_gate_scores=post_a),
        point_b: ProbeData(IQ=iq_b, labels=labels, post_gate_scores=post_b),
    }

    analysis = analyze_probe_sweep(data)
    assert set(analysis.keys()) == {
        "probe_points",
        "metrics",
        "pareto_indices",
        "best_overall",
        "best_mi",
        "best_qnd",
        "best_fidelity",
        "best_snr",
        "recommendations",
    }
    assert len(analysis["probe_points"]) == 2
    assert len(analysis["metrics"]) == 2
    assert isinstance(analysis["pareto_indices"], np.ndarray)
    assert analysis["recommendations"]["total_points"] == 2
