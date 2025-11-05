"""
Example: Ramsey Interferometry
==============================

Demonstrates Ramsey interference to measure:
1. T2* dephasing time
2. Frequency precision/detuning
3. Magnetic field sensitivity

Sequence: π/2 - τ - π/2 - Measure

The phase accumulated during free evolution creates interference fringes.
"""

import numpy as np
import matplotlib.pyplot as plt
from nv_center import NVCenter, NVParameters, PulseSequence

# Initialize NV-center
params = NVParameters(
    D=2.870,  # GHz
    omega_rabi=10.0,  # MHz
    T2=5.0  # µs - T2* dephasing time
)

nv = NVCenter(params=params, include_nuclear=False)

# Magnetic field
B_field = np.array([0.0, 0.0, 5.0])  # Gauss

ps = PulseSequence(nv, B_field)

# Run Ramsey experiments with different detunings
print("Running Ramsey interferometry...")

detunings = [0.0, 1.0, 2.0]  # MHz - detuning from resonance
colors = ['blue', 'red', 'green']
tau_max = 10.0  # microseconds

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Ramsey fringes with different detunings
ax1 = axes[0, 0]
for detuning, color in zip(detunings, colors):
    taus, populations = ps.ramsey_sequence(tau_max, n_points=200,
                                           detuning=detuning)
    ax1.plot(taus, populations, color=color, linewidth=2,
            label=f'Δf = {detuning} MHz')

ax1.set_xlabel('Free Evolution Time τ (µs)')
ax1.set_ylabel('Population in |0⟩')
ax1.set_title('Ramsey Fringes: Effect of Detuning')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Zoom on first few oscillations
ax2 = axes[0, 1]
tau_zoom = 2.0  # microseconds
for detuning, color in zip(detunings, colors):
    taus, populations = ps.ramsey_sequence(tau_zoom, n_points=200,
                                           detuning=detuning)
    ax2.plot(taus, populations, color=color, linewidth=2,
            label=f'Δf = {detuning} MHz')

    # Mark oscillation period
    if detuning > 0:
        period = 1.0 / detuning  # microseconds
        ax2.axvline(period, color=color, linestyle='--', alpha=0.3)

ax2.set_xlabel('Free Evolution Time τ (µs)')
ax2.set_ylabel('Population in |0⟩')
ax2.set_title('Ramsey Fringes: First Oscillations')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Extract frequency from Ramsey
ax3 = axes[1, 0]
detuning_test = 1.5  # MHz
taus, populations = ps.ramsey_sequence(tau_max, n_points=300,
                                       detuning=detuning_test)

# Fourier transform to extract frequency
from scipy import signal
from scipy.fft import fft, fftfreq

# Remove DC component
populations_ac = populations - np.mean(populations)

# Apply window to reduce spectral leakage
window = signal.windows.hann(len(populations_ac))
populations_windowed = populations_ac * window

# FFT
dt = taus[1] - taus[0]  # sampling interval
freqs = fftfreq(len(taus), dt)
fft_vals = np.abs(fft(populations_windowed))

# Only positive frequencies
positive_freqs = freqs > 0
freqs_pos = freqs[positive_freqs]
fft_pos = fft_vals[positive_freqs]

ax3.plot(freqs_pos, fft_pos, 'b-', linewidth=2)
ax3.axvline(detuning_test, color='r', linestyle='--', linewidth=2,
           label=f'True detuning: {detuning_test} MHz')

# Find peak
peak_idx = np.argmax(fft_pos)
measured_freq = freqs_pos[peak_idx]
ax3.axvline(measured_freq, color='g', linestyle='--', linewidth=2,
           label=f'Measured: {measured_freq:.2f} MHz')

ax3.set_xlabel('Frequency (MHz)')
ax3.set_ylabel('FFT Amplitude')
ax3.set_title('Frequency Extraction from Ramsey')
ax3.set_xlim(0, 5)
ax3.legend()
ax3.grid(True, alpha=0.3)

print(f"\nFrequency extraction:")
print(f"  True detuning: {detuning_test} MHz")
print(f"  Measured: {measured_freq:.3f} MHz")
print(f"  Error: {abs(measured_freq - detuning_test)*1000:.1f} kHz")

# Plot 4: T2* measurement from decay envelope
ax4 = axes[1, 1]
detuning_t2 = 2.0  # MHz - use some detuning to see oscillations
taus, populations = ps.ramsey_sequence(tau_max, n_points=200,
                                       detuning=detuning_t2)

ax4.plot(taus, populations, 'b-', linewidth=1, alpha=0.6, label='Data')

# Extract envelope (upper and lower)
from scipy.interpolate import interp1d

# Find local maxima and minima
peaks_max = signal.find_peaks(populations)[0]
peaks_min = signal.find_peaks(-populations)[0]

if len(peaks_max) > 2 and len(peaks_min) > 2:
    # Interpolate envelopes
    env_upper = interp1d(taus[peaks_max], populations[peaks_max],
                         kind='cubic', fill_value='extrapolate')
    env_lower = interp1d(taus[peaks_min], populations[peaks_min],
                         kind='cubic', fill_value='extrapolate')

    tau_smooth = np.linspace(taus[0], taus[-1], 500)
    ax4.plot(tau_smooth, env_upper(tau_smooth), 'r--', linewidth=2,
            label='Upper envelope')
    ax4.plot(tau_smooth, env_lower(tau_smooth), 'r--', linewidth=2,
            label='Lower envelope')

    # Fit exponential decay to estimate T2*
    # For ideal case, envelope should decay as exp(-t/T2*)
    contrast = (env_upper(tau_smooth) - env_lower(tau_smooth)) / 2

    # Theoretical T2* decay
    T2_star_theory = params.T2  # microseconds
    decay_theory = 0.5 * np.exp(-tau_smooth / T2_star_theory)
    ax4.plot(tau_smooth, 0.5 + decay_theory, 'g:', linewidth=2,
            label=f'Theory: T2* = {T2_star_theory} µs')
    ax4.plot(tau_smooth, 0.5 - decay_theory, 'g:', linewidth=2)

ax4.set_xlabel('Free Evolution Time τ (µs)')
ax4.set_ylabel('Population in |0⟩')
ax4.set_title('T2* Dephasing Measurement')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ramsey_interferometry.png', dpi=150)
print("\nPlot saved as 'ramsey_interferometry.png'")

# Magnetic field sensitivity
gamma_nv = params.gamma  # MHz/G
delta_f = 0.001  # MHz - frequency precision
delta_B = delta_f / gamma_nv  # Gauss

print(f"\nMagnetic field sensitivity:")
print(f"  Gyromagnetic ratio: {gamma_nv:.4f} MHz/G")
print(f"  Frequency precision: {delta_f*1000:.1f} kHz")
print(f"  Field sensitivity: {delta_B*1e6:.2f} nanoTesla")
print(f"  At T2* = {params.T2} µs: η ~ {1/(gamma_nv*np.sqrt(params.T2)):.2f} nT/√Hz")

plt.show()
