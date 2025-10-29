"""Unit tests for the quantum link fidelity helper functions."""
from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")

import quantum_link_fidelity as qlf


def test_db_to_lin_matches_known_value():
    """Ensure the decibel conversion agrees with hand-calculated value."""

    assert pytest.approx(qlf.db_to_lin(3.0)) == 0.501187


def test_link_eta_uses_extra_loss_only_at_zero_distance():
    params = qlf.DEFAULT_PARAMETERS
    expected = qlf.db_to_lin(params.extra_loss_db)
    assert pytest.approx(qlf.link_eta(0.0, params)) == expected


def test_compute_qber_handles_zero_signal_probabilities():
    """Noise-only scenario should yield 50% error rate."""

    assert qlf.compute_qber(0.0, 0.1, 0.0) == pytest.approx(0.5)


def test_simulate_fidelity_monotonic_decay():
    """Fidelity should decrease (or remain flat) with longer distances."""

    distances = np.linspace(0.0, 100.0, num=6)
    results = qlf.simulate_fidelity(distances)
    assert all(later <= earlier + 1e-9 for earlier, later in zip(results, results[1:]))


def test_signal_detection_probability_matches_closed_form():
    params = qlf.DEFAULT_PARAMETERS
    distance = 25.0
    eta = qlf.link_eta(distance, params)
    expected = 1.0 - math.exp(-params.mean_photons * eta * params.det_efficiency)
    assert qlf.signal_detection_probability(distance, params) == pytest.approx(expected)
