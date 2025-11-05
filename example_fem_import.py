"""
Example: Importing FEM Data from Commercial Software
====================================================

Demonstrates how to import and use B-field data from FEM software:
- COMSOL Multiphysics
- ANSYS HFSS
- CST Microwave Studio
- Custom formats

Shows:
1. Data format conversion
2. Spatial interpolation
3. Time-domain transformation
4. Multi-NV simulation with field gradients
"""

import numpy as np
import matplotlib.pyplot as plt
from nv_center import NVCenter, NVParameters
from fem_bfield import (
    FEMBFieldData,
    FEMBFieldLoader,
    MicrowaveAntennaSimulator
)

print("FEM Data Import and Processing Tutorial")
print("=" * 70)

# ============================================================================
# Example 1: Creating FEM-like data (simulating COMSOL export)
# ============================================================================
print("\n1. Simulating FEM software export format...")

# Generate sample spatial-temporal data like FEM would produce
# Typical FEM gives: (x, y, z, time, Bx, By, Bz)

print("   Creating sample dataset (representing COMSOL export)...")

# Spatial grid (around a CPW antenna)
nx, ny, nz = 10, 10, 15
x = np.linspace(-0.1, 0.1, nx)  # cm
y = np.linspace(-0.1, 0.1, ny)  # cm
z = np.linspace(0.001, 0.05, nz)  # cm (height above antenna)

# Time points (one MW cycle)
freq = 2.87  # GHz
period = 1.0 / freq  # µs
n_time = 50
time_points = np.linspace(0, period, n_time)

# Build full spatial-temporal array
# In real FEM: this would be exported from COMSOL/ANSYS
fem_data_full = np.zeros((nx, ny, nz, n_time, 7))

print(f"   Grid size: {nx}×{ny}×{nz} = {nx*ny*nz} spatial points")
print(f"   Time steps: {n_time}")
print(f"   Total data points: {nx*ny*nz*n_time}")

# Populate with synthetic antenna field
for i, xi in enumerate(x):
    for j, yj in enumerate(y):
        for k, zk in enumerate(z):
            pos = np.array([xi, yj, zk])

            # Calculate field amplitude from CPW model
            B_amp, angle = MicrowaveAntennaSimulator.coplanar_waveguide(
                pos, width=0.02, gap=0.01, power=1.0
            )

            # Store position
            fem_data_full[i, j, k, :, 0] = xi
            fem_data_full[i, j, k, :, 1] = yj
            fem_data_full[i, j, k, :, 2] = zk

            # Time-varying field
            omega = 2 * np.pi * freq  # GHz → rad/µs × 1000
            for t_idx, t in enumerate(time_points):
                fem_data_full[i, j, k, t_idx, 3] = t
                fem_data_full[i, j, k, t_idx, 4] = B_amp * np.cos(omega * 1000 * t) * np.cos(angle)
                fem_data_full[i, j, k, t_idx, 5] = B_amp * np.cos(omega * 1000 * t) * np.sin(angle)
                fem_data_full[i, j, k, t_idx, 6] = 0.0  # Bz (transverse field)

# Save in COMSOL-like CSV format
print("   Saving to COMSOL-style CSV...")
# Flatten to 2D array
data_2d = fem_data_full.reshape(-1, 7)
header = "x_cm,y_cm,z_cm,time_us,Bx_G,By_G,Bz_G"
np.savetxt('fem_comsol_export.csv', data_2d, delimiter=',',
          header=header, comments='', fmt='%.6e')
print("   ✓ Saved to fem_comsol_export.csv")

# ============================================================================
# Example 2: Loading and Processing FEM Data
# ============================================================================
print("\n2. Loading FEM data and extracting time series at NV position...")

# Read back the CSV
print("   Loading CSV data...")
data_loaded = np.loadtxt('fem_comsol_export.csv', delimiter=',', skiprows=1)
print(f"   ✓ Loaded {len(data_loaded)} rows")

# Extract data for specific NV position
nv_position = np.array([0.0, 0.0, 0.01])  # cm - 100 µm above CPW center

print(f"   Extracting field at NV position: {nv_position} cm")

# Find nearest grid point
distances = np.sqrt(
    (data_loaded[:, 0] - nv_position[0])**2 +
    (data_loaded[:, 1] - nv_position[1])**2 +
    (data_loaded[:, 2] - nv_position[2])**2
)

# Get unique time points at this location
nearest_idx = np.argsort(distances)[:n_time]
time_local = data_loaded[nearest_idx, 3]
B_local = data_loaded[nearest_idx, 4:7]

# Sort by time
sort_idx = np.argsort(time_local)
time_local = time_local[sort_idx]
B_local = B_local[sort_idx, :]

# Create FEMBFieldData object
fem_data_nv = FEMBFieldData(
    time=time_local,
    B_field=B_local,
    position=nv_position,
    frequency=freq,
    metadata={'source': 'fem_comsol_export.csv', 'nv_position': nv_position}
)

print(f"   ✓ Extracted {len(time_local)} time points")
print(f"   Field amplitude: {np.max(np.linalg.norm(B_local, axis=1)):.4f} G")

# ============================================================================
# Example 3: Multi-NV Simulation with Spatial Field Gradients
# ============================================================================
print("\n3. Simulating multiple NV-centers with different coupling strengths...")

fig = plt.figure(figsize=(16, 10))

# Define multiple NV positions
nv_positions = [
    np.array([0.0, 0.0, 0.005]),    # 50 µm - close
    np.array([0.0, 0.0, 0.01]),     # 100 µm - medium
    np.array([0.0, 0.0, 0.02]),     # 200 µm - far
]

colors = ['red', 'green', 'blue']
labels = ['50 µm', '100 µm', '200 µm']

# Create NV center
params = NVParameters(D=2.87, gamma=2.8025, T2=50.0)
nv = NVCenter(params=params)

# Run Rabi scan for each position
ax1 = plt.subplot(2, 3, 1)
ax2 = plt.subplot(2, 3, 2)

duration_max = 1.0  # µs
n_scan = 40
durations = np.linspace(0.01, duration_max, n_scan)

for nv_pos, color, label in zip(nv_positions, colors, labels):
    # Calculate field at this position
    B_amp, _ = MicrowaveAntennaSimulator.coplanar_waveguide(
        nv_pos, width=0.02, gap=0.01, power=1.0
    )

    print(f"   NV at {nv_pos[2]*10000:.0f} µm: B_MW = {B_amp:.4f} G")

    # Expected Rabi frequency
    gamma_MHz = params.gamma  # MHz/G
    omega_rabi = gamma_MHz * B_amp / 2

    # Run Rabi scan
    populations = []
    for dur in durations:
        mw_pulse = MicrowaveAntennaSimulator.generate_pulsed_mw(
            duration=dur,
            frequency=freq,
            amplitude=B_amp,
            envelope='rectangular',
            n_points=max(50, int(dur * 100))
        )

        nv.reset_to_ground()
        B_static = np.array([0, 0, 5.0])
        nv.apply_fem_mw_pulse(mw_pulse, B_static)
        _, P0, _ = nv.measure_population()
        populations.append(P0)

    # Plot
    ax1.plot(durations * 1000, populations, 'o-', color=color,
            linewidth=2, markersize=4, label=f'{label} (Ω={omega_rabi:.1f} MHz)')

    # Plot in frequency domain
    ax2.plot([nv_pos[2]*10000], [omega_rabi], 'o', color=color,
            markersize=12, label=label)

ax1.set_xlabel('Pulse Duration (ns)')
ax1.set_ylabel('Population in |0⟩')
ax1.set_title('Multi-NV Rabi: Distance-Dependent Coupling')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('Distance from CPW (µm)')
ax2.set_ylabel('Rabi Frequency (MHz)')
ax2.set_title('Field-Induced Inhomogeneous Broadening')
ax2.set_yscale('log')
ax2.legend()
ax2.grid(True, alpha=0.3)

# ============================================================================
# Example 4: Frequency-Domain FEM Data
# ============================================================================
print("\n4. Converting frequency-domain FEM data to time-domain...")

ax3 = plt.subplot(2, 3, 3)
ax4 = plt.subplot(2, 3, 4)

# Simulate frequency-domain data (e.g., from HFSS S-parameters)
# In real case: load from HFSS export

frequencies = np.linspace(2.8, 2.95, 100)  # GHz
# S21 parameter (transmission) - resonant response
f0 = 2.87  # Resonance
Q = 50  # Quality factor
S21 = 1.0 / (1 + 1j * Q * (frequencies - f0) / f0)

# Plot frequency response
ax3.plot(frequencies, np.abs(S21), 'b-', linewidth=2)
ax3.axvline(f0, color='r', linestyle='--', alpha=0.5, label=f'f₀ = {f0} GHz')
ax3.set_xlabel('Frequency (GHz)')
ax3.set_ylabel('|S₂₁|')
ax3.set_title('Frequency-Domain FEM Response (HFSS)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Convert to time domain via IFFT
from scipy.fft import ifft, fftfreq

# Create full spectrum (positive and negative frequencies)
S21_full = np.concatenate([S21, np.conj(S21[::-1])])
time_td = np.linspace(0, 10, len(S21_full))  # µs

# IFFT to time domain
h_time = ifft(S21_full)

# Generate time-domain pulse by convolving with input pulse
input_pulse = np.exp(-((time_td - 5)**2) / (2 * 0.5**2))  # Gaussian input
B_field_td = np.convolve(input_pulse, np.real(h_time), mode='same')

# Normalize
B_field_td = B_field_td / np.max(np.abs(B_field_td)) * 0.5  # 0.5 G amplitude

ax4.plot(time_td, input_pulse, 'g--', linewidth=2, alpha=0.7, label='Input pulse')
ax4.plot(time_td, B_field_td, 'r-', linewidth=2, label='Output (after antenna)')
ax4.set_xlabel('Time (µs)')
ax4.set_ylabel('Normalized B-field')
ax4.set_title('Time-Domain Response from Frequency Data')
ax4.legend()
ax4.grid(True, alpha=0.3)

print("   ✓ Converted S-parameter data to time-domain pulse")

# ============================================================================
# Example 5: Field Homogeneity Analysis
# ============================================================================
print("\n5. Analyzing field homogeneity across NV ensemble...")

ax5 = plt.subplot(2, 3, 5)
ax6 = plt.subplot(2, 3, 6)

# Create ensemble of NVs in a volume
n_nv = 100
# Random positions in a 100×100×50 µm³ volume centered at (0,0,100µm)
nv_ensemble_pos = np.random.rand(n_nv, 3) * np.array([0.01, 0.01, 0.005]) + \
                  np.array([-0.005, -0.005, 0.0075])

# Calculate field at each position
B_ensemble = []
for pos in nv_ensemble_pos:
    B_amp, _ = MicrowaveAntennaSimulator.coplanar_waveguide(
        pos, width=0.02, gap=0.01, power=1.0
    )
    B_ensemble.append(B_amp)

B_ensemble = np.array(B_ensemble)

# Plot field distribution
ax5.scatter(nv_ensemble_pos[:, 0] * 10000, nv_ensemble_pos[:, 2] * 10000,
           c=B_ensemble, cmap='viridis', s=50, edgecolors='black', linewidth=0.5)
ax5.set_xlabel('X Position (µm)')
ax5.set_ylabel('Z Position (µm)')
ax5.set_title('Field Distribution Across NV Ensemble')
cbar = plt.colorbar(ax5.collections[0], ax=ax5, label='B-field (G)')

# Histogram of field strengths
ax6.hist(B_ensemble, bins=20, color='blue', alpha=0.7, edgecolor='black')
ax6.axvline(np.mean(B_ensemble), color='r', linestyle='--',
           linewidth=2, label=f'Mean: {np.mean(B_ensemble):.4f} G')
ax6.axvline(np.median(B_ensemble), color='g', linestyle='--',
           linewidth=2, label=f'Median: {np.median(B_ensemble):.4f} G')
ax6.set_xlabel('B-field Amplitude (G)')
ax6.set_ylabel('Number of NVs')
ax6.set_title('Field Homogeneity Distribution')
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

# Calculate homogeneity metrics
std_B = np.std(B_ensemble)
mean_B = np.mean(B_ensemble)
homogeneity = (1 - std_B / mean_B) * 100

print(f"   Mean field: {mean_B:.4f} G")
print(f"   Std dev: {std_B:.4f} G")
print(f"   Homogeneity: {homogeneity:.1f}%")
print(f"   Min/Max ratio: {np.min(B_ensemble)/np.max(B_ensemble):.3f}")

# ============================================================================
# Summary and Tips
# ============================================================================

plt.tight_layout()
plt.savefig('fem_import_tutorial.png', dpi=150, bbox_inches='tight')
print("\n✓ Plot saved as 'fem_import_tutorial.png'")

print("\n" + "=" * 70)
print("FEM Import Tutorial - Summary and Best Practices")
print("=" * 70)

print("\n📊 Data Format Guidelines:")
print("  • COMSOL: Export as CSV with columns [x, y, z, t, Bx, By, Bz]")
print("  • ANSYS HFSS: Export field data as text, convert to NPZ")
print("  • CST: Use ASCII export, specify coordinates and field components")
print("  • Always include: spatial coords, time, 3D B-field vector")

print("\n⚙️ Processing Tips:")
print("  • Use NPZ format for Python (fast, compressed)")
print("  • Use HDF5 for very large datasets (>100 MB)")
print("  • Interpolate spatial data to NV positions")
print("  • Verify units: time in µs, field in Gauss")

print("\n🎯 Optimization Strategies:")
print("  • Design antennas for field homogeneity")
print("  • Place NVs in high-field regions")
print("  • Use frequency-domain for resonator analysis")
print("  • Check inhomogeneous broadening for ensembles")

print("\n✅ Quality Checks:")
print("  • Verify field amplitude is realistic (0.1-10 G typical)")
print("  • Check time resolution (>10 points per Rabi period)")
print("  • Ensure spatial sampling covers NV ensemble")
print("  • Validate with analytical models when possible")

print("\n🔧 Common FEM Software Workflows:")
print("\n  COMSOL Multiphysics:")
print("    1. Run RF module simulation")
print("    2. Export → Data → Cut Point/Line/Plane")
print("    3. Select B-field (emw.normH for normalized)")
print("    4. Export as CSV or Excel")
print("\n  ANSYS HFSS:")
print("    1. Analyze → Solution Data")
print("    2. Field Calculator → B-field")
print("    3. Export Field Data")
print("    4. Convert to NPZ with provided script")
print("\n  CST Microwave Studio:")
print("    1. Post-processing → Field Monitor")
print("    2. Export → ASCII format")
print("    3. Use import helper function")

print("\n📚 Example import code snippets available in documentation")
print("=" * 70)

plt.show()

# Clean up generated files
import os
print("\nCleaning up example files...")
for f in ['fem_comsol_export.csv', 'fem_mw_pulse.npz', 'fem_mw_pulse.csv']:
    if os.path.exists(f):
        os.remove(f)
        print(f"  Removed: {f}")
