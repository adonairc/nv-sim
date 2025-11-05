"""
Example: Realistic B-Field Dynamics and Control
===============================================

Demonstrates advanced NV-center simulation with realistic time-dependent
magnetic fields including:
- Static bias fields
- AC modulation (e.g., from target signals)
- Noise (environmental fluctuations)
- Field ramping and sweeps

Applications:
- AC magnetometry
- Quantum sensing protocols
- Noise spectroscopy
- Field reconstruction
"""

import numpy as np
import matplotlib.pyplot as plt
from nv_center import NVCenter, NVParameters, PulseSequence
import qutip as qt

print("Realistic B-Field Dynamics Simulation")
print("=" * 50)

# Initialize NV-center with realistic parameters
params = NVParameters(
    D=2.870,  # GHz
    gamma=2.8025,  # MHz/G
    omega_rabi=10.0,  # MHz
    T2=30.0,  # µs
)

nv = NVCenter(params=params, include_nuclear=False)

# ============================================================================
# Example 1: AC Magnetometry - Detecting oscillating magnetic fields
# ============================================================================
print("\n1. AC Magnetometry: Detecting oscillating field...")

fig = plt.figure(figsize=(16, 12))

# Static field + AC signal
B_static = np.array([0.0, 0.0, 5.0])  # Gauss
B_ac_amplitude = 0.5  # Gauss
f_ac = 1.0  # MHz - AC field frequency

# Time-dependent Hamiltonian simulation
tau_sense = 20.0  # µs - sensing time
n_points = 500
tlist = np.linspace(0, tau_sense, n_points)

# Define time-dependent field
def B_field_ac(t, args):
    """Time-dependent B-field with AC component"""
    omega_ac = args['omega_ac']
    return B_static + np.array([0, 0, B_ac_amplitude * np.sin(omega_ac * t)])

# Build time-dependent Hamiltonian
H_static = nv.hamiltonian_static(B_static)

# AC Zeeman term: γ * B_ac(t) * Sz
gamma = params.gamma / 1000.0  # GHz/G
def H_ac_coeff(t, args):
    """Coefficient for AC field term"""
    omega_ac = args['omega_ac']
    return gamma * B_ac_amplitude * np.sin(omega_ac * t)

H_ac = nv.Sz
H_list = [H_static, [H_ac, H_ac_coeff]]

# Prepare state in superposition (π/2 pulse)
nv.reset_to_ground()
nv.apply_pulse('pi/2_x', 0.05, B_static)
psi0 = nv.state

# Evolve with AC field
args = {'omega_ac': 2 * np.pi * f_ac}  # rad/µs
e_ops = [nv.Sx, nv.Sy, nv.Sz]  # Track all spin components

result = nv.evolve_time_dependent(H_list, tlist, state_init=psi0, e_ops=e_ops)

# Plot Bloch sphere dynamics
ax1 = plt.subplot(3, 3, 1)
Sx_exp = np.real(result.expect[0])
Sy_exp = np.real(result.expect[1])
Sz_exp = np.real(result.expect[2])

ax1.plot(tlist, Sx_exp, 'r-', linewidth=2, label='⟨Sx⟩')
ax1.plot(tlist, Sy_exp, 'g-', linewidth=2, label='⟨Sy⟩')
ax1.plot(tlist, Sz_exp, 'b-', linewidth=2, label='⟨Sz⟩')
ax1.set_xlabel('Time (µs)')
ax1.set_ylabel('Expectation Value')
ax1.set_title('Spin Precession under AC Field')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Phase accumulation
ax2 = plt.subplot(3, 3, 2)
phase = np.arctan2(Sy_exp, Sx_exp)
ax2.plot(tlist, phase, 'purple', linewidth=2)
ax2.set_xlabel('Time (µs)')
ax2.set_ylabel('Phase (rad)')
ax2.set_title('Phase Accumulation from AC Field')
ax2.grid(True, alpha=0.3)

# Plot 3: Frequency analysis
ax3 = plt.subplot(3, 3, 3)
from scipy.fft import fft, fftfreq

dt = tlist[1] - tlist[0]
freqs_fft = fftfreq(len(tlist), dt)
fft_phase = np.abs(fft(phase - np.mean(phase)))

positive_mask = freqs_fft > 0
ax3.plot(freqs_fft[positive_mask], fft_phase[positive_mask], 'b-', linewidth=2)
ax3.axvline(f_ac, color='r', linestyle='--', linewidth=2,
           label=f'Signal: {f_ac} MHz')
ax3.set_xlabel('Frequency (MHz)')
ax3.set_ylabel('FFT Amplitude')
ax3.set_title('Detected AC Field Frequency')
ax3.set_xlim(0, 5)
ax3.legend()
ax3.grid(True, alpha=0.3)

# ============================================================================
# Example 2: Field Noise Spectroscopy
# ============================================================================
print("2. Magnetic field noise spectroscopy...")

# Simulate measurement with different noise levels
ax4 = plt.subplot(3, 3, 4)
ax5 = plt.subplot(3, 3, 5)

noise_levels = [0.0, 0.1, 0.5, 1.0]  # Gauss
colors = ['blue', 'green', 'orange', 'red']

for noise_level, color in zip(noise_levels, colors):
    # Run multiple trials with noise
    n_trials = 50
    final_pops = []

    for trial in range(n_trials):
        nv.reset_to_ground()
        nv.apply_pulse('pi/2_x', 0.05, B_static)

        # Evolve with noisy field
        tau_noise = 10.0
        n_steps = 100
        dt_noise = tau_noise / n_steps

        for step in range(n_steps):
            # Add random noise to field
            B_noise = B_static + np.random.normal(0, noise_level, 3)
            H_noise = nv.hamiltonian_static(B_noise)
            nv.evolve(H_noise, dt_noise)

        # Readout
        nv.apply_pulse('pi/2_x', 0.05, B_static)
        _, P0, _ = nv.measure_population()
        final_pops.append(P0)

    # Plot histogram
    ax4.hist(final_pops, bins=20, alpha=0.5, color=color,
            label=f'σ = {noise_level} G', density=True)

    # Plot mean and std
    mean_pop = np.mean(final_pops)
    std_pop = np.std(final_pops)
    ax5.errorbar(noise_level, mean_pop, yerr=std_pop, fmt='o',
                color=color, markersize=8, capsize=5)

ax4.set_xlabel('Final Population in |0⟩')
ax4.set_ylabel('Probability Density')
ax4.set_title('Effect of Field Noise on Readout')
ax4.legend()
ax4.grid(True, alpha=0.3)

ax5.set_xlabel('Noise Level σ (Gauss)')
ax5.set_ylabel('Mean Population in |0⟩')
ax5.set_title('Decoherence vs Noise Strength')
ax5.grid(True, alpha=0.3)

print(f"  Noise reduces coherence: larger fluctuations with noise")

# ============================================================================
# Example 3: Field Sweeps and Adiabatic Passage
# ============================================================================
print("3. Magnetic field sweep (adiabatic passage)...")

ax6 = plt.subplot(3, 3, 6)

# Sweep B-field from low to high
B_start = 1.0  # Gauss
B_end = 20.0  # Gauss
sweep_time = 50.0  # µs

n_sweep = 200
tlist_sweep = np.linspace(0, sweep_time, n_sweep)

# Linear sweep
B_z_sweep = np.linspace(B_start, B_end, n_sweep)

# Prepare in |0⟩ and track populations during sweep
nv.reset_to_ground()
populations_sweep = np.zeros((n_sweep, 3))  # Store all three populations

for i, (t, B_z) in enumerate(zip(tlist_sweep, B_z_sweep)):
    if i > 0:
        dt = tlist_sweep[i] - tlist_sweep[i-1]
        B_field_current = np.array([0, 0, B_z])
        H_current = nv.hamiltonian_static(B_field_current)
        nv.evolve(H_current, dt)

    populations_sweep[i] = nv.measure_population()

# Plot population evolution
ax6.plot(B_z_sweep, populations_sweep[:, 0], 'b-', linewidth=2,
        label='ms = -1')
ax6.plot(B_z_sweep, populations_sweep[:, 1], 'g-', linewidth=2,
        label='ms = 0')
ax6.plot(B_z_sweep, populations_sweep[:, 2], 'r-', linewidth=2,
        label='ms = +1')
ax6.set_xlabel('Magnetic Field B_z (Gauss)')
ax6.set_ylabel('Population')
ax6.set_title('Adiabatic Field Sweep')
ax6.legend()
ax6.grid(True, alpha=0.3)

# ============================================================================
# Example 4: Composite field (static + AC + noise)
# ============================================================================
print("4. Composite field: static + AC + noise...")

ax7 = plt.subplot(3, 3, 7)
ax8 = plt.subplot(3, 3, 8)

# Define realistic composite field
B_static_comp = np.array([0.5, 0.3, 8.0])  # Gauss
B_ac_comp = 0.3  # Gauss
f_ac_comp = 0.5  # MHz
noise_comp = 0.05  # Gauss

tau_composite = 30.0
n_comp = 300
tlist_comp = np.linspace(0, tau_composite, n_comp)

# Sample field at each time point
B_fields_sampled = []
for t in tlist_comp:
    B_ac_contrib = B_ac_comp * np.sin(2 * np.pi * f_ac_comp * t)
    B_noise_contrib = np.random.normal(0, noise_comp, 3)
    B_total = B_static_comp + np.array([0, 0, B_ac_contrib]) + B_noise_contrib
    B_fields_sampled.append(B_total)

B_fields_sampled = np.array(B_fields_sampled)

# Plot field components
ax7.plot(tlist_comp, B_fields_sampled[:, 0], 'r-', alpha=0.7, linewidth=1,
        label='Bx')
ax7.plot(tlist_comp, B_fields_sampled[:, 1], 'g-', alpha=0.7, linewidth=1,
        label='By')
ax7.plot(tlist_comp, B_fields_sampled[:, 2], 'b-', alpha=0.7, linewidth=1,
        label='Bz')
ax7.set_xlabel('Time (µs)')
ax7.set_ylabel('B-field (Gauss)')
ax7.set_title('Realistic Composite B-Field')
ax7.legend()
ax7.grid(True, alpha=0.3)

# Simulate NV response to this field
nv.reset_to_ground()
nv.apply_pulse('pi/2_x', 0.05, B_static_comp)

Sz_values = []
for i in range(len(tlist_comp) - 1):
    dt = tlist_comp[i+1] - tlist_comp[i]
    H_current = nv.hamiltonian_static(B_fields_sampled[i])
    nv.evolve(H_current, dt)
    Sz_values.append(nv.measure_expectation(nv.Sz))

ax8.plot(tlist_comp[:-1], Sz_values, 'purple', linewidth=2)
ax8.set_xlabel('Time (µs)')
ax8.set_ylabel('⟨Sz⟩')
ax8.set_title('NV Response to Composite Field')
ax8.grid(True, alpha=0.3)

# ============================================================================
# Example 5: Quantum sensing protocol - XY8 sequence
# ============================================================================
print("5. Dynamical decoupling: XY8-N sequence...")

ax9 = plt.subplot(3, 3, 9)

# XY8 sequence for AC magnetometry
# Sequence: π/2 - (X-Y-X-Y-Y-X-Y-X)^N - π/2
# where X,Y are π pulses with alternating phases

def xy8_sequence(nv, ps, N_cycles, tau_cycle, f_target):
    """
    Implement XY8-N dynamical decoupling sequence

    Parameters
    ----------
    N_cycles : int
        Number of XY8 cycles
    tau_cycle : float
        Duration of one cycle (µs)
    f_target : float
        Target frequency to sense (MHz)
    """
    nv.reset_to_ground()

    # Initial π/2 pulse
    B_field_seq = np.array([0, 0, 5.0])
    ps.nv.apply_pulse('pi/2_x', 0.05, B_field_seq)

    # Phases for XY8: X-Y-X-Y-Y-X-Y-X = 0, π/2, 0, π/2, π/2, 0, π/2, 0
    phases = [0, np.pi/2, 0, np.pi/2, np.pi/2, 0, np.pi/2, 0]
    n_pulses = len(phases)
    tau_between = tau_cycle / n_pulses  # Time between pulses

    for cycle in range(N_cycles):
        for phase in phases:
            # Free evolution
            if tau_between > 0:
                # AC field during evolution
                omega_ac_rad = 2 * np.pi * f_target
                t_start = cycle * tau_cycle + phases.index(phase) * tau_between

                # Simple AC field model
                B_ac_val = 0.5 * np.sin(omega_ac_rad * t_start)
                B_eff = B_field_seq + np.array([0, 0, B_ac_val])
                H_evo = ps.nv.hamiltonian_static(B_eff)
                ps.nv.evolve(H_evo, tau_between / 2)

            # π pulse with specific phase
            ps.nv.apply_pulse('custom', 0.05, B_field_seq,
                             amplitude=100.0, phase=phase)  # High Rabi for short pulse

            # Free evolution after pulse
            if tau_between > 0:
                t_mid = cycle * tau_cycle + phases.index(phase) * tau_between
                B_ac_val = 0.5 * np.sin(omega_ac_rad * t_mid)
                B_eff = B_field_seq + np.array([0, 0, B_ac_val])
                H_evo = ps.nv.hamiltonian_static(B_eff)
                ps.nv.evolve(H_evo, tau_between / 2)

    # Final π/2 pulse
    ps.nv.apply_pulse('pi/2_x', 0.05, B_field_seq)

    _, P0, _ = ps.nv.measure_population()
    return P0

# Scan target frequency
ps = PulseSequence(nv, np.array([0, 0, 5.0]))
freqs_xy8 = np.linspace(0.1, 3.0, 30)
signals_xy8 = []

N_cycles = 4
tau_cycle = 5.0  # µs

for f_target in freqs_xy8:
    signal = xy8_sequence(nv, ps, N_cycles, tau_cycle, f_target)
    signals_xy8.append(signal)

ax9.plot(freqs_xy8, signals_xy8, 'bo-', linewidth=2, markersize=6)
ax9.set_xlabel('Target Frequency (MHz)')
ax9.set_ylabel('Signal (Population in |0⟩)')
ax9.set_title(f'XY8-{N_cycles} AC Magnetometry')
ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bfield_dynamics.png', dpi=150)
print("\nPlot saved as 'bfield_dynamics.png'")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 50)
print("Realistic B-Field Control Summary")
print("=" * 50)

print("\nImplemented scenarios:")
print("  1. AC magnetometry - detecting oscillating fields")
print("  2. Noise spectroscopy - characterizing field noise")
print("  3. Adiabatic sweeps - controlled field ramping")
print("  4. Composite fields - static + AC + noise")
print("  5. Dynamical decoupling - XY8 quantum sensing")

print("\nKey parameters:")
print(f"  Gyromagnetic ratio: {params.gamma} MHz/G")
print(f"  T2 coherence time: {params.T2} µs")
print(f"  Rabi frequency: {params.omega_rabi} MHz")

print("\nApplications:")
print("  - Sensitive magnetometry (nT - µT range)")
print("  - AC field detection and spectroscopy")
print("  - Quantum sensing with dynamical decoupling")
print("  - Environmental noise characterization")

plt.show()
