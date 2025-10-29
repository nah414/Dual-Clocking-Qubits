from __future__ import annotations
import math
from typing import Dict, Any, Optional, List
from .base import Backend, ProbeResult

class NMRBackend(Backend):
    """NMR backend: RF transmit (TX) and receiver (RX) frames with ensemble detection.

    - 'drive' domain: RF transmit near Larmor frequency.
    - 'probe' domain: receiver (acquisition) with its own reference (heterodyne OK).

    We keep an internal Bloch vector M = (Mx, My, Mz) for a single effective spin.
    The model is deliberately simple but captures flip angles and FID decay.
    """

    def __init__(self, dt: float = 1e-6, nucleus: str = "1H", **kwargs):
        super().__init__(dt=dt, name="nmr", **kwargs)
        self.nucleus = nucleus
        self.frames = {}  # (spin, domain) -> (freq_Hz, phase_rad)
        # Simple state for one "spin"
        self.M = {0: (0.0, 0.0, 1.0)}
        self.params: Dict[str, Any] = {}
        self._phase_spread: Dict[int, float] = {}

    def calibrate(self, qubit: int = 0, **kwargs) -> Dict[str, Any]:
        # Defaults roughly for 1H at 7.05 T (≈ 300 MHz system)
        gamma_hz_per_t = kwargs.get("gamma_hz_per_t", 42.577e6)    # Hz/T
        gamma_rad_per_t = 2 * math.pi * gamma_hz_per_t             # rad/s/T
        B0 = kwargs.get("B0", 7.05)                                # Tesla
        B1_max = kwargs.get("B1_max", 25e-6)                       # Tesla (25 µT drive ceiling)
        T1 = kwargs.get("T1", 1.0)                                  # seconds
        T2 = kwargs.get("T2", 0.2)                                  # seconds
        rx_dt = kwargs.get("rx_dt", 10e-6)                          # receiver dwell (s)
        n_samples = kwargs.get("n_samples", 256)                    # FID length
        sample_extent = kwargs.get("sample_extent", 0.02)           # metres (≈2 cm sample)
        omega_L = gamma_hz_per_t * B0                               # Hz

        self.params = dict(
            nucleus=self.nucleus, gamma_hz_per_t=gamma_hz_per_t, gamma_rad_per_t=gamma_rad_per_t,
            B0=B0, B1_max=B1_max, T1=T1, T2=T2, rx_dt=rx_dt, n_samples=n_samples,
            sample_extent=sample_extent,
            omega_L=omega_L
        )
        self.frames[(qubit, "drive")] = (omega_L, 0.0)
        self.frames[(qubit, "probe")] = (omega_L, 0.0)
        self.M[qubit] = (0.0, 0.0, 1.0)
        self._phase_spread[qubit] = 0.0
        return dict(self.params)

    # --- Frame control ---
    def set_frame(self, qubit: int, freq: float, phase: float = 0.0, domain: str = "drive") -> None:
        self.frames[(qubit, domain)] = (float(freq), float(phase))

    def shift_phase(self, qubit: int, dphi: float, domain: str = "drive") -> None:
        f, phi = self.frames.get((qubit, domain), (0.0, 0.0))
        self.frames[(qubit, domain)] = (f, (phi + dphi) % (2*math.pi))

    # --- Pulse physics ---
    def _rotate(self, M, axis, theta):
        # Rodrigues' rotation for unit axis (nx,ny,nz) by angle theta
        nx, ny, nz = axis
        mx, my, mz = M
        c = math.cos(theta); s = math.sin(theta); one_c = 1 - c
        # Rotation matrix applied to M
        rx = (c + nx*nx*one_c)*mx + (nx*ny*one_c - nz*s)*my + (nx*nz*one_c + ny*s)*mz
        ry = (ny*nx*one_c + nz*s)*mx + (c + ny*ny*one_c)*my + (ny*nz*one_c - nx*s)*mz
        rz = (nz*nx*one_c - ny*s)*mx + (nz*ny*one_c + nx*s)*my + (c + nz*nz*one_c)*mz
        return (rx, ry, rz)

    def play_pulse(self, qubit: int, duration: float, amp: float, phase: float = 0.0, freq: Optional[float] = None, domain: str = "drive", **kwargs) -> None:
        # Map amp & duration to flip angle: theta = gamma_rad * B1 * t
        par = self.params or self.calibrate(qubit)
        gamma_rad = par["gamma_rad_per_t"]
        B1_max = par["B1_max"]
        B1 = max(0.0, amp) * B1_max
        theta = gamma_rad * B1 * max(0.0, duration)
        # rotation axis in XY plane given by phase: phase=0 -> x, pi/2 -> y
        axis = (math.cos(phase), math.sin(phase), 0.0)
        self.M[qubit] = self._rotate(self.M.get(qubit, (0.0, 0.0, 1.0)), axis, theta)

    def apply_gradient(self, qubit: int, duration: float, strength: float, axis: str = "z") -> None:
        """Apply a gradient lobe, accumulating ensemble phase spread.

        The accumulated spread models a linear gradient of magnitude ``strength`` (T/m)
        over the sample extent. The overall FID amplitude is modulated by a sinc factor
        derived from the stored spread value during acquisition. A subsequent gradient
        with opposite signed area cancels the spread, reproducing a simple echo effect.
        """

        par = self.params or self.calibrate(qubit)
        extent = par.get("sample_extent", 0.0)
        gamma_rad = par["gamma_rad_per_t"]
        spread = gamma_rad * strength * extent * max(duration, 0.0)
        self._phase_spread[qubit] = self._phase_spread.get(qubit, 0.0) + spread

    def _evolve_relax(self, M, dt, par):
        Mx, My, Mz = M
        T1 = max(par["T1"], 1e-9); T2 = max(par["T2"], 1e-9)
        # simple Euler decay for short dt (sufficient for rx_dt steps)
        ex = math.exp(-dt/T2); ez = math.exp(-dt/T1)
        Mx *= ex; My *= ex; Mz = 1 - (1 - Mz) * ez
        return (Mx, My, Mz)

    def probe(self, qubit: int, duration: float, strength: float, detuning: float = 0.0, weak: bool = True, **kwargs) -> ProbeResult:
        """Acquire an FID: turn on RX for a window, return a small time series.

        - 'strength' scales receiver gain (simulated SNR).
        - 'detuning' shifts RX frequency relative to Larmor (Hz).
        """
        par = self.params or self.calibrate(qubit)
        rx_dt = par["rx_dt"]; N = min(int(max(duration, rx_dt)/rx_dt), par["n_samples"])
        if N < 1: N = 1
        omega_L = par["omega_L"]  # Hz
        rx_freq, rx_phase = self.frames.get((qubit, "probe"), (omega_L, 0.0))
        delta_f = (rx_freq - omega_L) + detuning  # Hz residual
        # Simulate FID samples (real channel) with exponential T2 decay
        fid: List[float] = []
        t = 0.0
        M = self.M.get(qubit, (0.0, 0.0, 1.0))
        spread = abs(self._phase_spread.get(qubit, 0.0))
        if spread != 0.0:
            attenuation = math.sin(0.5 * spread) / (0.5 * spread)
        else:
            attenuation = 1.0

        for i in range(N):
            # transverse magnetization magnitude (ignoring coil sensitivity)
            Mx, My, Mz = M
            # rotate observable by RX phase reference
            phase = 2*math.pi*delta_f*t + rx_phase
            signal = (Mx*math.cos(phase) + My*math.sin(phase)) * attenuation
            # apply receiver gain
            fid.append(signal * max(1e-3, strength))
            # evolve one dwell for relaxation
            M = self._evolve_relax(M, rx_dt, par)
            t += rx_dt
        # Persist updated magnetization (post-acquisition decay)
        self.M[qubit] = M
        # Gradient dephasing persists after acquisition
        gamma_phi = 1.0 / max(par["T2"], 1e-9)
        # Use RMS as scalar 'signal' summary
        rms = math.sqrt(sum(x*x for x in fid)/len(fid)) if fid else 0.0
        return ProbeResult(counts=N, signal=rms, p01=None, gamma_phi=gamma_phi,
                           metadata={"fid": fid, "rx_dt": rx_dt, "delta_f": delta_f, "weak": weak})

    def measure(self, qubit: int, duration: float, **kwargs) -> ProbeResult:
        # In NMR, "strong" just means long acquisition; state change is non-destructive
        return self.probe(qubit, duration=duration, strength=1.0, weak=False, **kwargs)
