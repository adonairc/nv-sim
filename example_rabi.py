"""
Example: Rabi Oscillation
=========================

Demonstrates Rabi oscillations - coherent oscillation between spin states
under resonant microwave driving.

This is a fundamental experiment to calibrate pulse amplitudes and verify
coherent control of the NV-center.
"""

import numpy as np
import matplotlib.pyplot as plt
from nv_center import NVCenter, NVParameters, PulseSequence

# Initialize NV-center with default parameters
params = NVParameters(
    D=2.870,  # GHz - zero-field splitting
    omega_rabi=5.0,  # MHz - MW Rabi frequency
    T2_rabi=2.0  # µs - coherence time during driving
)

nv = NVCenter(params=params, include_nuclear=False)

# Set up magnetic field (aligned with NV axis for simplicity)
# Small field to lift degeneracy between ms=±1
B_field = np.array([0.0, 0.0, 10.0])  # Gauss

# Create pulse sequence generator
ps = PulseSequence(nv, B_field)

# Run Rabi oscillation experiment
print("Running Rabi oscillation simulation...")
print(f"Rabi frequency: {params.omega_rabi} MHz")
print(f"Expected period: {1000.0 / params.omega_rabi:.2f} ns")

duration_max = 1.0  # microseconds
times, populations = ps.rabi_oscillation(duration_max, n_points=200,
                                         omega_rabi=params.omega_rabi)

# Calculate theoretical Rabi frequency
omega_theory = params.omega_rabi  # MHz
period_theory = 1.0 / omega_theory  # microseconds

# Visualize results
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(times * 1000, populations, 'b-', linewidth=2, label='Simulation')
plt.axvline(period_theory * 1000 / 2, color='r', linestyle='--',
            alpha=0.5, label=f'π pulse ({period_theory*500:.0f} ns)')
plt.axvline(period_theory * 1000, color='g', linestyle='--',
            alpha=0.5, label=f'2π pulse ({period_theory*1000:.0f} ns)')
plt.xlabel('Pulse Duration (ns)')
plt.ylabel('Population in |0⟩')
plt.title('Rabi Oscillation: Coherent Spin Manipulation')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
# Zoom in on first oscillation
idx_zoom = times < (2 * period_theory)
plt.plot(times[idx_zoom] * 1000, populations[idx_zoom], 'b-', linewidth=2)
plt.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='50% population')
plt.xlabel('Pulse Duration (ns)')
plt.ylabel('Population in |0⟩')
plt.title('First Rabi Oscillation (Zoomed)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rabi_oscillation.png', dpi=150)
print("\nPlot saved as 'rabi_oscillation.png'")

# Find π and π/2 pulse times
min_idx = np.argmin(populations)
pi_pulse_time = times[min_idx]

# Find first crossing at 0.5
cross_idx = np.where(populations < 0.5)[0]
if len(cross_idx) > 0:
    pi_2_pulse_time = times[cross_idx[0]]
else:
    pi_2_pulse_time = pi_pulse_time / 2

print(f"\nCalibrated pulse times:")
print(f"  π/2 pulse: {pi_2_pulse_time*1000:.2f} ns")
print(f"  π pulse:   {pi_pulse_time*1000:.2f} ns")
print(f"  Theoretical π pulse: {period_theory*500:.2f} ns")

plt.show()
