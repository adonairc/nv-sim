"""
Example: Hahn Echo (Spin Echo)
==============================

Demonstrates spin echo sequence to measure T2 coherence time.

Sequence: π/2 - τ - π - τ - π/2 - Measure

The π pulse refocuses dephasing, allowing measurement of true T2
(longer than T2* from Ramsey) by eliminating inhomogeneous broadening.
"""

import numpy as np
import matplotlib.pyplot as plt
from nv_center import NVCenter, NVParameters, PulseSequence

# Initialize NV-center
params = NVParameters(
    D=2.870,  # GHz
    omega_rabi=10.0,  # MHz
    T2=20.0,  # µs - coherence time (T2 > T2*)
    T1=1000.0  # µs - relaxation time
)

nv = NVCenter(params=params, include_nuclear=False)

# Magnetic field
B_field = np.array([0.0, 0.0, 5.0])  # Gauss

ps = PulseSequence(nv, B_field)

print("Running Hahn Echo simulation...")
print(f"Expected T2: {params.T2} µs")

tau_max = 50.0  # microseconds
taus, populations = ps.hahn_echo(tau_max, n_points=150)

# Calculate total echo time (2*tau)
echo_times = 2 * taus

# Create comprehensive visualization
fig = plt.figure(figsize=(14, 10))

# Plot 1: Echo decay vs 2τ
ax1 = plt.subplot(2, 2, 1)
ax1.plot(echo_times, populations, 'bo-', markersize=4, linewidth=1.5,
         label='Echo signal')

# Fit exponential decay: P(t) = A * exp(-t/T2) + offset
from scipy.optimize import curve_fit

def exp_decay(t, A, T2, offset):
    return A * np.exp(-t / T2) + offset

try:
    # Initial guess
    p0 = [0.5, params.T2, 0.5]
    popt, pcov = curve_fit(exp_decay, echo_times, populations, p0=p0)
    A_fit, T2_fit, offset_fit = popt

    # Plot fit
    t_fit = np.linspace(0, echo_times[-1], 200)
    ax1.plot(t_fit, exp_decay(t_fit, *popt), 'r-', linewidth=2,
            label=f'Fit: T2 = {T2_fit:.2f} µs')

    print(f"\nFitted T2: {T2_fit:.2f} ± {np.sqrt(pcov[1,1]):.2f} µs")
    print(f"Expected T2: {params.T2:.2f} µs")
except:
    print("Exponential fit failed - not enough decay visible")
    T2_fit = params.T2

ax1.set_xlabel('Total Echo Time 2τ (µs)')
ax1.set_ylabel('Population in |0⟩')
ax1.set_title('Hahn Echo: T2 Coherence Measurement')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Comparison with T2* (Ramsey)
ax2 = plt.subplot(2, 2, 2)

# Simulate Ramsey for comparison (use shorter T2*)
params_ramsey = NVParameters(
    D=params.D,
    omega_rabi=params.omega_rabi,
    T2=5.0  # T2* is shorter than T2
)
nv_ramsey = NVCenter(params=params_ramsey, include_nuclear=False)
ps_ramsey = PulseSequence(nv_ramsey, B_field)

tau_max_compare = 30.0
taus_ramsey, pops_ramsey = ps_ramsey.ramsey_sequence(tau_max_compare, n_points=100,
                                                      detuning=0.5)

ax2.plot(taus_ramsey, pops_ramsey, 'b-', linewidth=2, alpha=0.6,
        label=f'Ramsey (T2* = {params_ramsey.T2} µs)')

# Plot Hahn echo on same axis
taus_hahn, pops_hahn = ps.hahn_echo(tau_max_compare, n_points=100)
ax2.plot(2*taus_hahn, pops_hahn, 'r-', linewidth=2,
        label=f'Hahn Echo (T2 = {params.T2} µs)')

ax2.set_xlabel('Time (µs)')
ax2.set_ylabel('Population in |0⟩')
ax2.set_title('Hahn Echo vs Ramsey: T2 > T2*')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Echo signal visualization (pulse sequence diagram)
ax3 = plt.subplot(2, 2, 3)

# Create schematic pulse sequence
t_example = 5.0  # µs for tau
pulse_width = 0.1

times_scheme = [0,
                pulse_width,
                pulse_width + t_example,
                pulse_width + t_example + 2*pulse_width,
                2*pulse_width + 2*t_example,
                3*pulse_width + 2*t_example]
pulse_labels = ['π/2', '', 'π', '', 'π/2', 'Measure']

# Draw pulses
for i, (t, label) in enumerate(zip(times_scheme[:-1], pulse_labels[:-1])):
    if label:
        if 'π/2' in label:
            height = 0.5
        else:
            height = 1.0
        ax3.bar(t, height, width=pulse_width, color='blue', alpha=0.7,
               edgecolor='black', linewidth=1.5)
        ax3.text(t + pulse_width/2, height + 0.1, label, ha='center',
                fontsize=12, fontweight='bold')

# Draw free evolution periods
ax3.arrow(pulse_width + 0.05, -0.3, t_example - 0.1, 0,
         head_width=0.1, head_length=0.05, fc='green', ec='green')
ax3.text(pulse_width + t_example/2, -0.5, 'τ', ha='center',
        fontsize=14, color='green', fontweight='bold')

ax3.arrow(2*pulse_width + t_example + 0.05, -0.3, t_example - 0.1, 0,
         head_width=0.1, head_length=0.05, fc='green', ec='green')
ax3.text(2*pulse_width + 1.5*t_example, -0.5, 'τ', ha='center',
        fontsize=14, color='green', fontweight='bold')

# Measurement
ax3.text(times_scheme[-1] + 0.2, 0.5, 'Measure', ha='left',
        fontsize=11, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

ax3.set_xlim(-0.5, times_scheme[-1] + 1.5)
ax3.set_ylim(-0.8, 1.5)
ax3.set_xlabel('Time')
ax3.set_ylabel('MW Amplitude')
ax3.set_title('Hahn Echo Pulse Sequence')
ax3.set_xticks([])
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Dynamical decoupling comparison
ax4 = plt.subplot(2, 2, 4)

# Single echo (already have)
ax4.plot(2*taus, populations, 'b-', linewidth=2, marker='o',
        markersize=3, label='Hahn Echo (1 pulse)')

# For demonstration: show what CPMG (multiple π pulses) would do
# Simplified: assume each π pulse extends coherence
n_pi_pulses = [1, 2, 4]
colors_dd = ['blue', 'red', 'green']

for n_pi, color in zip(n_pi_pulses, colors_dd):
    # Simplified model: T2_eff increases with more pulses
    # In reality, needs proper CPMG simulation
    T2_eff = params.T2 * np.sqrt(n_pi)  # Simplified scaling
    signal = 0.5 * np.exp(-echo_times / T2_eff) + 0.5

    ax4.plot(echo_times, signal, '--', color=color, linewidth=1.5,
            alpha=0.7, label=f'CPMG-{n_pi} (T2,eff ~ {T2_eff:.1f} µs)')

ax4.set_xlabel('Total Time (µs)')
ax4.set_ylabel('Population in |0⟩')
ax4.set_title('Dynamical Decoupling: Extending Coherence')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hahn_echo.png', dpi=150)
print("\nPlot saved as 'hahn_echo.png'")

# Summary statistics
print(f"\n{'='*50}")
print("Summary:")
print(f"{'='*50}")
print(f"Coherence times:")
print(f"  T2* (Ramsey):     {params_ramsey.T2:.1f} µs")
print(f"  T2 (Hahn Echo):   {params.T2:.1f} µs")
print(f"  T1 (Relaxation):  {params.T1:.1f} µs")
print(f"\nRatio T2/T2*:       {params.T2/params_ramsey.T2:.2f}")
print(f"\nEcho decay fitted:  T2 = {T2_fit:.2f} µs")

# Calculate fidelity at different times
times_check = [1.0, 5.0, 10.0, 20.0]
print(f"\nEcho fidelity at different times:")
for t in times_check:
    idx = np.argmin(np.abs(echo_times - t))
    fidelity = populations[idx]
    print(f"  {t:5.1f} µs: {fidelity*100:5.1f}%")

plt.show()
