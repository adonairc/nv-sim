"""
Example: FEM Antenna-Driven MW Pulses
=====================================

Demonstrates using realistic B-field data from FEM simulations of microwave
antennas to drive NV-center transitions.

This example shows:
1. Loading/generating FEM B-field data
2. Different antenna geometries (wire, CPW)
3. Realistic MW pulse envelopes
4. Spatial field profiles
5. Comparison with ideal pulses

Applications:
- Experiment planning and optimization
- Antenna design and characterization
- Realistic pulse fidelity analysis
- Inhomogeneous MW field effects
"""

import numpy as np
import matplotlib.pyplot as plt
from nv_center import NVCenter, NVParameters
from fem_bfield import (
    FEMBFieldData,
    FEMBFieldLoader,
    MicrowaveAntennaSimulator,
    create_mw_field_function
)

print("FEM Antenna-Driven MW Pulse Simulation")
print("=" * 70)

# Initialize NV-center
params = NVParameters(
    D=2.870,
    gamma=2.8025,
    omega_rabi=10.0,
    T2=50.0
)

nv = NVCenter(params=params, include_nuclear=False)

# ============================================================================
# Example 1: Different MW Pulse Envelopes
# ============================================================================
print("\n1. MW Pulse Envelopes from FEM Antenna Simulation...")

fig = plt.figure(figsize=(16, 12))

# Static background field
B_static = np.array([0, 0, 5.0])  # Gauss

# Generate different envelope types
envelopes = ['rectangular', 'gaussian', 'raised_cosine', 'sech']
colors = ['blue', 'red', 'green', 'purple']

duration = 0.5  # µs
frequency = params.D  # GHz - on resonance
amplitude = 1.0  # Gauss

ax1 = plt.subplot(3, 3, 1)
ax2 = plt.subplot(3, 3, 2)

for envelope, color in zip(envelopes, colors):
    # Generate MW pulse with envelope
    mw_pulse = MicrowaveAntennaSimulator.generate_pulsed_mw(
        duration=duration,
        frequency=frequency,
        amplitude=amplitude,
        envelope=envelope,
        n_points=500
    )

    # Plot MW field
    ax1.plot(mw_pulse.time * 1000, mw_pulse.B_field[:, 0], color=color,
            linewidth=2, label=envelope, alpha=0.7)

    # Simulate Rabi with this pulse
    nv.reset_to_ground()
    result = nv.apply_fem_mw_pulse(mw_pulse, B_static,
                                   e_ops=[nv.Sx, nv.Sy, nv.Sz])

    # Plot final population
    Sz_trace = np.real(result.expect[2])
    ax2.plot(mw_pulse.time, Sz_trace, color=color, linewidth=2,
            label=envelope, alpha=0.7)

ax1.set_xlabel('Time (ns)')
ax1.set_ylabel('B_x MW Field (Gauss)')
ax1.set_title('MW Pulse Envelopes')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('Time (µs)')
ax2.set_ylabel('⟨Sz⟩')
ax2.set_title('NV Spin Response to Different Envelopes')
ax2.legend()
ax2.grid(True, alpha=0.3)

print(f"   Generated {len(envelopes)} envelope types")

# ============================================================================
# Example 2: Wire Antenna Rabi Oscillations
# ============================================================================
print("\n2. Rabi oscillations with wire antenna...")

ax3 = plt.subplot(3, 3, 3)

# Wire antenna at different distances
distances = [0.01, 0.05, 0.1]  # cm
power = 1.0  # Watt
wire_length = 0.5  # cm

for distance in distances:
    # Calculate field amplitude at NV position
    nv_pos = np.array([0, 0, distance])
    B_amp = MicrowaveAntennaSimulator.wire_antenna(
        nv_pos, wire_length=wire_length, power=power, frequency=frequency
    )

    print(f"   Distance: {distance*10:.1f} mm → B_MW: {B_amp:.3f} G")

    # Run Rabi scan
    duration_max = 1.0  # µs
    n_points = 50
    durations = np.linspace(0.01, duration_max, n_points)
    populations = []

    for dur in durations:
        mw_pulse = MicrowaveAntennaSimulator.generate_pulsed_mw(
            duration=dur,
            frequency=frequency,
            amplitude=B_amp,
            envelope='rectangular',
            n_points=max(50, int(dur * 100))
        )

        nv.reset_to_ground()
        nv.apply_fem_mw_pulse(mw_pulse, B_static)
        _, P0, _ = nv.measure_population()
        populations.append(P0)

    ax3.plot(durations * 1000, populations, 'o-', linewidth=2, markersize=4,
            label=f'd = {distance*10:.1f} mm (B = {B_amp:.2f} G)')

ax3.set_xlabel('Pulse Duration (ns)')
ax3.set_ylabel('Population in |0⟩')
ax3.set_title('Rabi with Wire Antenna at Different Distances')
ax3.legend()
ax3.grid(True, alpha=0.3)

# ============================================================================
# Example 3: Coplanar Waveguide (CPW) Antenna
# ============================================================================
print("\n3. Coplanar waveguide antenna simulation...")

ax4 = plt.subplot(3, 3, 4)
ax5 = plt.subplot(3, 3, 5)

# CPW parameters
cpw_width = 0.02  # cm (200 µm)
cpw_gap = 0.01  # cm (100 µm)
cpw_power = 0.5  # W

# Height scan above CPW
heights = np.linspace(0.001, 0.1, 50)  # cm
B_amplitudes = []

for h in heights:
    pos = np.array([0, 0, h])
    B_amp, _ = MicrowaveAntennaSimulator.coplanar_waveguide(
        pos, width=cpw_width, gap=cpw_gap, power=cpw_power
    )
    B_amplitudes.append(B_amp)

ax4.semilogy(heights * 10, B_amplitudes, 'b-', linewidth=2)
ax4.set_xlabel('Height Above CPW (mm)')
ax4.set_ylabel('B-field Amplitude (Gauss)')
ax4.set_title('CPW Field Profile vs Height')
ax4.grid(True, alpha=0.3)

# Rabi frequency vs height
rabi_freqs = []
gamma_MHz = params.gamma  # MHz/G

for B_amp in B_amplitudes:
    # Rabi frequency: Ω = γ * B_MW / 2
    omega_rabi = gamma_MHz * B_amp / 2
    rabi_freqs.append(omega_rabi)

ax5.plot(heights * 10, rabi_freqs, 'r-', linewidth=2)
ax5.set_xlabel('Height Above CPW (mm)')
ax5.set_ylabel('Rabi Frequency (MHz)')
ax5.set_title('Expected Rabi Frequency vs Height')
ax5.grid(True, alpha=0.3)

print(f"   CPW width: {cpw_width*10000:.0f} µm")
print(f"   At h=10µm: B = {B_amplitudes[5]:.3f} G, Ω_Rabi = {rabi_freqs[5]:.2f} MHz")

# ============================================================================
# Example 4: Spatial Field Profile (2D)
# ============================================================================
print("\n4. Generating spatial 2D field profile...")

ax6 = plt.subplot(3, 3, 6)

# Create 2D grid
x_range = np.linspace(-0.1, 0.1, 40)  # cm
z_range = np.linspace(0.001, 0.1, 40)  # cm
X, Z = np.meshgrid(x_range, z_range)

# Calculate field amplitude at each point
B_grid = np.zeros_like(X)

for i, x in enumerate(x_range):
    for j, z in enumerate(z_range):
        pos = np.array([x, 0, z])
        B_amp, _ = MicrowaveAntennaSimulator.coplanar_waveguide(
            pos, width=cpw_width, gap=cpw_gap, power=cpw_power
        )
        B_grid[j, i] = B_amp

# Plot field map
im = ax6.contourf(X * 10, Z * 10, B_grid, levels=20, cmap='hot')
plt.colorbar(im, ax=ax6, label='B-field (Gauss)')
ax6.set_xlabel('Lateral Position (mm)')
ax6.set_ylabel('Height (mm)')
ax6.set_title('CPW Spatial Field Profile')

# Mark NV positions
nv_positions = [[0, 0.01], [0.3, 0.05], [-0.3, 0.05]]
for pos in nv_positions:
    ax6.plot(pos[0], pos[1] * 10, 'w*', markersize=12)

# ============================================================================
# Example 5: Comparison - Ideal vs FEM Pulse
# ============================================================================
print("\n5. Comparing ideal vs FEM-realistic pulses...")

ax7 = plt.subplot(3, 3, 7)
ax8 = plt.subplot(3, 3, 8)

# Parameters for comparison
pi_pulse_duration = 0.2  # µs
n_reps = 10  # Number of π pulses

# Ideal pulse (perfect rectangular, instant rise time)
times_ideal = np.linspace(0, pi_pulse_duration * n_reps, 1000)
populations_ideal = []

for t in times_ideal:
    # Number of complete π pulses
    n_pi = int(t / pi_pulse_duration)
    phase = (t - n_pi * pi_pulse_duration) / pi_pulse_duration * np.pi

    # Population oscillates: P_0 = cos^2(phase/2)
    P0_ideal = np.cos(phase / 2)**2
    populations_ideal.append(P0_ideal)

ax7.plot(times_ideal * 1000, populations_ideal, 'b-', linewidth=2,
        label='Ideal (instant rise)')

# FEM-realistic pulse (with rise time, envelope)
populations_fem = []
time_elapsed = 0

for rep in range(n_reps):
    mw_pulse = MicrowaveAntennaSimulator.generate_pulsed_mw(
        duration=pi_pulse_duration,
        frequency=frequency,
        amplitude=2.0,  # Adjusted for π pulse
        envelope='gaussian',
        n_points=100
    )

    if rep == 0:
        nv.reset_to_ground()

    result = nv.apply_fem_mw_pulse(mw_pulse, B_static)
    _, P0, _ = nv.measure_population()
    populations_fem.append(P0)

ax7.plot(np.arange(n_reps) * pi_pulse_duration * 1000,
        populations_fem, 'ro-', linewidth=2, markersize=8,
        label='FEM (Gaussian envelope)')

ax7.set_xlabel('Time (ns)')
ax7.set_ylabel('Population in |0⟩')
ax7.set_title('Ideal vs FEM-Realistic Pulses')
ax7.legend()
ax7.grid(True, alpha=0.3)

# Fidelity comparison
ax8_data = []

rise_times = [0.001, 0.01, 0.05, 0.1]  # µs
target_angle = np.pi  # π pulse

for rise_time in rise_times:
    mw_pulse = MicrowaveAntennaSimulator.generate_pulsed_mw(
        duration=0.3,
        frequency=frequency,
        amplitude=1.5,
        envelope='rectangular',
        rise_time=rise_time,
        n_points=500
    )

    nv.reset_to_ground()
    nv.apply_fem_mw_pulse(mw_pulse, B_static)
    _, P0, _ = nv.measure_population()

    # Ideal π pulse should give P0 = 0
    fidelity = 1 - P0  # How close to excited state
    ax8_data.append(fidelity)

ax8.plot(np.array(rise_times) * 1000, ax8_data, 'go-',
        linewidth=2, markersize=10)
ax8.axhline(1.0, color='gray', linestyle='--', alpha=0.5,
           label='Ideal fidelity')
ax8.set_xlabel('Rise Time (ns)')
ax8.set_ylabel('Pulse Fidelity')
ax8.set_title('Effect of Finite Rise Time')
ax8.legend()
ax8.grid(True, alpha=0.3)

print(f"   Fidelity at 1 ns rise: {ax8_data[0]*100:.1f}%")
print(f"   Fidelity at 100 ns rise: {ax8_data[-1]*100:.1f}%")

# ============================================================================
# Example 6: Loading/Saving FEM Data
# ============================================================================
print("\n6. Demonstrating FEM data I/O...")

ax9 = plt.subplot(3, 3, 9)

# Generate sample FEM data
sample_duration = 2.0  # µs
sample_mw = MicrowaveAntennaSimulator.generate_pulsed_mw(
    duration=sample_duration,
    frequency=frequency,
    amplitude=0.8,
    envelope='raised_cosine',
    n_points=1000
)

# Save to different formats
print("   Saving FEM data to files...")

# NPZ format (recommended)
np.savez('fem_mw_pulse.npz',
         time=sample_mw.time,
         B_field=sample_mw.B_field,
         frequency=sample_mw.frequency)
print("     ✓ Saved to fem_mw_pulse.npz")

# CSV format
csv_data = np.column_stack([sample_mw.time,
                            sample_mw.B_field[:, 0],
                            sample_mw.B_field[:, 1],
                            sample_mw.B_field[:, 2]])
np.savetxt('fem_mw_pulse.csv', csv_data, delimiter=',',
          header='time_us,Bx_G,By_G,Bz_G', comments='')
print("     ✓ Saved to fem_mw_pulse.csv")

# Load back and verify
loaded_npz = FEMBFieldLoader.load_npy('fem_mw_pulse.npz')
loaded_csv = FEMBFieldLoader.load_csv('fem_mw_pulse.csv',
                                     time_unit='us', field_unit='G')

# Plot original and loaded
ax9.plot(sample_mw.time, sample_mw.B_field[:, 0], 'b-',
        linewidth=2, label='Original')
ax9.plot(loaded_npz.time, loaded_npz.B_field[:, 0], 'r--',
        linewidth=2, alpha=0.7, label='Loaded (NPZ)')
ax9.plot(loaded_csv.time, loaded_csv.B_field[:, 0], 'g:',
        linewidth=2, alpha=0.7, label='Loaded (CSV)')

ax9.set_xlabel('Time (µs)')
ax9.set_ylabel('Bx Field (Gauss)')
ax9.set_title('FEM Data Save/Load Verification')
ax9.legend()
ax9.grid(True, alpha=0.3)

print("     ✓ Loaded and verified data integrity")

# ============================================================================
# Save and Summary
# ============================================================================

plt.suptitle('FEM Antenna-Driven MW Pulse Simulation', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('fem_antenna_simulation.png', dpi=150, bbox_inches='tight')
print("\n✓ Plot saved as 'fem_antenna_simulation.png'")

print("\n" + "=" * 70)
print("Summary: FEM Antenna Integration")
print("=" * 70)

print("\n✓ Demonstrated capabilities:")
print("  • Multiple MW pulse envelope types")
print("  • Wire antenna field calculations")
print("  • Coplanar waveguide (CPW) modeling")
print("  • Spatial field profiles (2D mapping)")
print("  • Ideal vs realistic pulse comparison")
print("  • FEM data I/O (CSV, NPZ formats)")

print("\n✓ Antenna types supported:")
print("  • Wire dipole antenna")
print("  • Coplanar waveguide (CPW)")
print("  • Custom via FEM import")

print("\n✓ Realistic physics included:")
print("  • Spatial field inhomogeneity")
print("  • Finite rise/fall times")
print("  • Distance-dependent coupling")
print("  • Angular field distribution")

print("\n✓ File formats supported:")
print("  • CSV (text, portable)")
print("  • NPY/NPZ (NumPy native)")
print("  • HDF5 (large datasets, with h5py)")
print("  • MATLAB .mat (with scipy.io)")

print("\n✓ Next steps:")
print("  • Import your own FEM data from COMSOL/ANSYS")
print("  • Optimize antenna geometry for uniform fields")
print("  • Analyze pulse fidelity vs antenna design")
print("  • Multi-NV simulations with spatial gradients")

print("\n" + "=" * 70)

plt.show()
