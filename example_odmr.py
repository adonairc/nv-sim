"""
Example: ODMR Spectrum
======================

Demonstrates optically detected magnetic resonance (ODMR) spectroscopy.

ODMR is the primary readout mechanism for NV-centers:
- Scan microwave frequency while continuously driving
- Measure fluorescence (proportional to ms=0 population)
- Resonances appear at transitions: |0⟩ ↔ |±1⟩

Applications:
- Magnetic field sensing (Zeeman splitting)
- Temperature sensing (zero-field splitting D vs T)
- Strain/stress sensing (E parameter)
"""

import numpy as np
import matplotlib.pyplot as plt
from nv_center import NVCenter, NVParameters, PulseSequence

# Initialize NV-center
params = NVParameters(
    D=2.870,  # GHz - zero-field splitting at room temp
    E=0.001,  # GHz - small strain splitting
    gamma=2.8025,  # MHz/G
    omega_rabi=5.0,  # MHz
)

print("ODMR Spectroscopy Simulation")
print("=" * 50)

# Create figure with multiple ODMR scenarios
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Scenario 1: Zero magnetic field (aligned with NV axis)
# ======================================================
ax1 = axes[0, 0]
print("\n1. Zero magnetic field (B = 0)...")

nv = NVCenter(params=params, include_nuclear=False)
B_field = np.array([0.0, 0.0, 0.0])
ps = PulseSequence(nv, B_field)

# Scan around D (zero-field splitting)
freq_center = params.D
freq_range = 0.020  # GHz
freqs, contrast = ps.odmr_spectrum(freq_center - freq_range,
                                   freq_center + freq_range,
                                   n_points=300,
                                   pulse_duration=5.0)

ax1.plot(freqs, contrast, 'b-', linewidth=2)
ax1.axvline(params.D, color='r', linestyle='--', alpha=0.5,
           label=f'D = {params.D} GHz')
ax1.set_xlabel('MW Frequency (GHz)')
ax1.set_ylabel('ODMR Contrast')
ax1.set_title('ODMR: Zero Field (Degenerate ms=±1)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Scenario 2: Magnetic field along NV axis (z)
# =============================================
ax2 = axes[0, 1]
print("2. Magnetic field along NV axis (B_z)...")

B_values = [5.0, 10.0, 20.0]  # Gauss
colors = ['blue', 'red', 'green']

for B_z, color in zip(B_values, colors):
    B_field = np.array([0.0, 0.0, B_z])
    ps = PulseSequence(nv, B_field)

    # Calculate expected splitting
    gamma = params.gamma / 1000.0  # GHz/G
    delta_f = gamma * B_z  # GHz - Zeeman splitting

    # Scan wider range to see both transitions
    freq_range = 0.050
    freqs, contrast = ps.odmr_spectrum(freq_center - freq_range,
                                       freq_center + freq_range,
                                       n_points=400,
                                       pulse_duration=5.0)

    ax2.plot(freqs, contrast, color=color, linewidth=2,
            label=f'B = {B_z} G (Δf = {delta_f*1000:.1f} MHz)')

    # Mark expected transitions
    f_minus = params.D - delta_f
    f_plus = params.D + delta_f
    ax2.axvline(f_minus, color=color, linestyle=':', alpha=0.3)
    ax2.axvline(f_plus, color=color, linestyle=':', alpha=0.3)

ax2.set_xlabel('MW Frequency (GHz)')
ax2.set_ylabel('ODMR Contrast')
ax2.set_title('ODMR: Zeeman Splitting (Field Along NV Axis)')
ax2.legend()
ax2.grid(True, alpha=0.3)

print("  Zeeman splitting observed!")

# Scenario 3: Magnetic field sensitivity calibration
# ==================================================
ax3 = axes[1, 0]
print("3. Magnetic field calibration curve...")

B_range = np.linspace(0, 50, 20)  # Gauss
f_minus_measured = []
f_plus_measured = []

for B_z in B_range:
    B_field = np.array([0.0, 0.0, B_z])
    ps = PulseSequence(nv, B_field)

    # Quick scan to find peaks
    freq_range = 0.08
    freqs, contrast = ps.odmr_spectrum(freq_center - freq_range,
                                       freq_center + freq_range,
                                       n_points=200,
                                       pulse_duration=3.0)

    # Find two peaks (ms = ±1 transitions)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(contrast, height=0.1, distance=10)

    if len(peaks) >= 2:
        # Sort by frequency
        peak_freqs = freqs[peaks]
        peak_freqs_sorted = np.sort(peak_freqs)
        f_minus_measured.append(peak_freqs_sorted[0])
        f_plus_measured.append(peak_freqs_sorted[-1])
    elif len(peaks) == 1:
        # Only one peak (degenerate)
        f_minus_measured.append(freqs[peaks[0]])
        f_plus_measured.append(freqs[peaks[0]])
    else:
        f_minus_measured.append(params.D)
        f_plus_measured.append(params.D)

f_minus_measured = np.array(f_minus_measured)
f_plus_measured = np.array(f_plus_measured)

# Plot transitions vs B field
ax3.plot(B_range, f_minus_measured, 'bo-', linewidth=2, markersize=6,
        label='ms = -1 transition')
ax3.plot(B_range, f_plus_measured, 'ro-', linewidth=2, markersize=6,
        label='ms = +1 transition')

# Theoretical lines
gamma = params.gamma / 1000.0  # GHz/G
f_theory_minus = params.D - gamma * B_range
f_theory_plus = params.D + gamma * B_range
ax3.plot(B_range, f_theory_minus, 'b--', alpha=0.5, label='Theory (ms=-1)')
ax3.plot(B_range, f_theory_plus, 'r--', alpha=0.5, label='Theory (ms=+1)')

ax3.set_xlabel('Magnetic Field B_z (Gauss)')
ax3.set_ylabel('Resonance Frequency (GHz)')
ax3.set_title('ODMR Frequency vs Magnetic Field')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Scenario 4: Temperature sensing via zero-field splitting
# ========================================================
ax4 = axes[1, 1]
print("4. Temperature sensing (D parameter shift)...")

# D parameter varies with temperature: dD/dT ≈ -74 kHz/K
# Room temp (300 K) → D ≈ 2.870 GHz

temperatures = np.array([250, 300, 350, 400, 450])  # Kelvin
D_vs_T = 2.870 - 74e-6 * (temperatures - 300)  # GHz

colors_temp = plt.cm.coolwarm(np.linspace(0, 1, len(temperatures)))

B_field = np.array([0.0, 0.0, 0.0])  # Zero field for simplicity

for T, D_temp, color in zip(temperatures, D_vs_T, colors_temp):
    # Create NV with temperature-dependent D
    params_temp = NVParameters(D=D_temp, omega_rabi=5.0)
    nv_temp = NVCenter(params=params_temp, include_nuclear=False)
    ps_temp = PulseSequence(nv_temp, B_field)

    # Scan narrow range
    freq_range = 0.015
    freqs, contrast = ps_temp.odmr_spectrum(D_temp - freq_range,
                                            D_temp + freq_range,
                                            n_points=200,
                                            pulse_duration=5.0)

    # Offset for visibility
    offset = (T - temperatures[0]) * 0.0002
    ax4.plot(freqs, contrast + offset, color=color, linewidth=2,
            label=f'{T} K (D={D_temp:.4f} GHz)')

ax4.set_xlabel('MW Frequency (GHz)')
ax4.set_ylabel('ODMR Contrast (offset for clarity)')
ax4.set_title('Temperature Sensing via D Parameter')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('odmr_spectrum.png', dpi=150)
print("\nPlot saved as 'odmr_spectrum.png'")

# Summary of sensing capabilities
print("\n" + "=" * 50)
print("ODMR Sensing Summary")
print("=" * 50)

# Magnetic field sensitivity
freq_precision = 0.0001  # GHz = 100 kHz (typical with good SNR)
B_sensitivity = freq_precision / (params.gamma / 1000.0)
print(f"\nMagnetic Field Sensing:")
print(f"  Gyromagnetic ratio: {params.gamma:.4f} MHz/G")
print(f"  Frequency precision: {freq_precision*1e6:.1f} kHz")
print(f"  Field sensitivity: {B_sensitivity:.4f} G = {B_sensitivity*0.1:.2f} mT")
print(f"                     = {B_sensitivity*100:.1f} µT")

# Temperature sensitivity
dD_dT = -74e-6  # GHz/K
T_sensitivity = freq_precision / abs(dD_dT)
print(f"\nTemperature Sensing:")
print(f"  dD/dT: {dD_dT*1e6:.1f} kHz/K")
print(f"  Temperature sensitivity: {T_sensitivity:.2f} K")
print(f"  At 100 kHz precision: {0.0001/abs(dD_dT):.1f} K")

# Strain/stress sensing
print(f"\nStrain Sensing:")
print(f"  E parameter (strain): {params.E*1e6:.1f} kHz")
print(f"  Strain modifies E → frequency shifts in ODMR")

plt.show()
