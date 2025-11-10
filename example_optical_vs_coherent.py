"""
Example: Optical Model vs Coherent Model Comparison
===================================================

Compares the full optical dynamics model (8-level + master equation)
with the simplified coherent model (3-level + Schrödinger equation).

Shows when each model is appropriate:

Full Optical Model (optical_dynamics.py):
- Use when: Optical pumping, PL readout, ODMR, realistic dynamics
- Includes: Excited state, singlets, ISC, dissipation
- Method: Master equation (Lindblad)
- Slower but realistic

Coherent Model (nv_center.py):
- Use when: Pure MW manipulation, pulse sequences, coherent control
- Includes: Ground state only, no dissipation
- Method: Schrödinger equation
- Faster, ideal pulses

This example helps users choose the right model for their application.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from nv_center import NVCenter, NVParameters, PulseSequence
from optical_dynamics import (
    ExtendedNVCenter,
    OpticalParameters,
    simulate_ODMR_with_PL_contrast
)

print("Optical vs Coherent Model Comparison")
print("=" * 70)

# Common parameters
nv_params = NVParameters(D=2.870, gamma=2.8025, omega_rabi=10.0, T2=50.0)
optical_params = OpticalParameters()
B_field = np.array([0, 0, 10.0])

fig = plt.figure(figsize=(16, 10))

# ============================================================================
# Example 1: Rabi Oscillations - Coherent Model
# ============================================================================
print("\n1. Rabi oscillations with coherent model (Schrödinger)...")

ax1 = plt.subplot(2, 3, 1)

nv_coherent = NVCenter(params=nv_params, include_nuclear=False)
ps = PulseSequence(nv_coherent, B_field)

t_start = time.time()
duration_max = 1.0
times_rabi, pops_rabi = ps.rabi_oscillation(duration_max, n_points=100)
t_coherent = time.time() - t_start

ax1.plot(times_rabi * 1000, pops_rabi, 'b-', linewidth=2, label='Coherent model')
ax1.set_xlabel('Pulse Duration (ns)')
ax1.set_ylabel('Population in |0⟩')
ax1.set_title(f'Rabi Oscillations - Coherent Model ({t_coherent:.3f}s)')
ax1.legend()
ax1.grid(True, alpha=0.3)

print(f"   Coherent model time: {t_coherent:.3f} s")

# ============================================================================
# Example 2: Rabi with Optical Model (No Optical Pumping)
# ============================================================================
print("\n2. Rabi oscillations with optical model (master equation)...")

ax2 = plt.subplot(2, 3, 2)

nv_optical = ExtendedNVCenter(nv_params, optical_params)

durations = np.linspace(0.01, duration_max, 50)  # Fewer points for speed
pops_optical = []

t_start = time.time()

for dur in durations:
    nv_optical.reset_to_ground(ms=0)

    # MW driving without laser
    tlist = np.linspace(0, dur, 30)
    mw_params = {
        'omega': nv_params.D,
        'amplitude': nv_params.omega_rabi,
        'phase': 0
    }

    # Track ground state population
    e_ops = [nv_optical.proj['g,0']]

    result = nv_optical.evolve_master_equation(
        tlist, B_field, laser_on=False, mw_params=mw_params, e_ops=e_ops
    )

    pops_optical.append(result.expect[0][-1])

t_optical = time.time() - t_start

ax2.plot(times_rabi * 1000, pops_rabi, 'b-', linewidth=2, alpha=0.5, label='Coherent')
ax2.plot(durations * 1000, pops_optical, 'ro-', linewidth=2, markersize=4, label='Optical (no laser)')
ax2.set_xlabel('Pulse Duration (ns)')
ax2.set_ylabel('Population in |0⟩')
ax2.set_title(f'Rabi Oscillations - Optical Model ({t_optical:.3f}s)')
ax2.legend()
ax2.grid(True, alpha=0.3)

print(f"   Optical model time: {t_optical:.3f} s")
print(f"   Speed ratio: {t_optical/t_coherent:.1f}x slower")
print(f"   → Use coherent model for pure MW control")

# ============================================================================
# Example 3: ODMR - Coherent Model (Approximation)
# ============================================================================
print("\n3. ODMR with coherent model (approximation)...")

ax3 = plt.subplot(2, 3, 3)

# Coherent model approximation: assume perfect contrast
freq_min = nv_params.D - 0.05
freq_max = nv_params.D + 0.05
freqs_odmr = np.linspace(freq_min, freq_max, 100)

t_start = time.time()

# Simulate by running quick Rabi at each frequency
odmr_coherent = []
for freq in freqs_odmr:
    nv_coherent.reset_to_ground()

    # Quick pulse
    dur = 0.5
    nv_coherent.apply_pulse('custom', dur, B_field,
                           omega_mw=freq, amplitude=10.0)

    _, P0, _ = nv_coherent.measure_population()
    odmr_coherent.append(P0)

t_odmr_coherent = time.time() - t_start

# Normalize (coherent model gives populations, not PL)
odmr_coherent = np.array(odmr_coherent)

ax3.plot(freqs_odmr, odmr_coherent, 'b-', linewidth=2, label='Coherent (populations)')
ax3.set_xlabel('MW Frequency (GHz)')
ax3.set_ylabel('Population / PL')
ax3.set_title(f'ODMR - Coherent Model ({t_odmr_coherent:.3f}s)')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.invert_yaxis()

print(f"   ODMR coherent model: {t_odmr_coherent:.3f} s")

# ============================================================================
# Example 4: ODMR - Optical Model (Realistic)
# ============================================================================
print("\n4. ODMR with optical model (realistic PL contrast)...")

ax4 = plt.subplot(2, 3, 4)

t_start = time.time()

freqs_optical, PL_optical = simulate_ODMR_with_PL_contrast(
    nv_params, optical_params, B_field,
    (freq_min, freq_max),
    n_points=80,
    mw_duration=5.0,
    mw_power=10.0
)

t_odmr_optical = time.time() - t_start

ax4.plot(freqs_odmr, 1 - (odmr_coherent - np.min(odmr_coherent))/(np.max(odmr_coherent) - np.min(odmr_coherent)),
        'b--', linewidth=2, alpha=0.5, label='Coherent (scaled)')
ax4.plot(freqs_optical, PL_optical, 'r-', linewidth=2, label='Optical (realistic PL)')
ax4.set_xlabel('MW Frequency (GHz)')
ax4.set_ylabel('Normalized PL')
ax4.set_title(f'ODMR - Optical Model ({t_odmr_optical:.3f}s)')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.invert_yaxis()

print(f"   ODMR optical model: {t_odmr_optical:.3f} s")
print(f"   Speed ratio: {t_odmr_optical/t_odmr_coherent:.1f}x slower")
print(f"   → Optical model gives realistic PL contrast")

# Calculate contrast difference
contrast_coherent = (np.max(odmr_coherent) - np.min(odmr_coherent)) / np.max(odmr_coherent) * 100
contrast_optical = (np.max(PL_optical) - np.min(PL_optical)) / np.max(PL_optical) * 100

print(f"   Coherent contrast: {contrast_coherent:.1f}% (ideal)")
print(f"   Optical contrast: {contrast_optical:.1f}% (realistic)")

# ============================================================================
# Example 5: Ramsey with Optical Pumping vs Without
# ============================================================================
print("\n5. Ramsey sequence with and without optical pumping...")

ax5 = plt.subplot(2, 3, 5)

# Coherent model (no optical effects)
tau_max = 5.0
taus_ramsey = np.linspace(0, tau_max, 50)
ramsey_coherent = []

for tau in taus_ramsey:
    nv_coherent.reset_to_ground()
    nv_coherent.apply_pulse('pi/2_x', 0.05, B_field)

    if tau > 0:
        H_static = nv_coherent.hamiltonian_static(B_field)
        nv_coherent.evolve(H_static, tau)

    nv_coherent.apply_pulse('pi/2_x', 0.05, B_field)
    _, P0, _ = nv_coherent.measure_population()
    ramsey_coherent.append(P0)

ax5.plot(taus_ramsey, ramsey_coherent, 'b-', linewidth=2, label='Coherent (ideal)')

# Optical model with realistic initialization imperfection
ramsey_optical = []

for tau in taus_ramsey[::2]:  # Every other point for speed
    # Optical pumping (not perfect - leaves some in ±1)
    nv_optical.reset_to_ground(ms=0)
    nv_optical.optical_pumping_cycle(0.5, B_field, n_points=20)

    # Get actual populations after pumping
    pops_init = nv_optical.get_ground_state_populations()

    # π/2 pulse
    tlist_pi2 = np.linspace(0, 0.05, 20)
    mw_params = {'omega': nv_params.D, 'amplitude': 100.0, 'phase': 0}
    nv_optical.evolve_master_equation(tlist_pi2, B_field, laser_on=False, mw_params=mw_params)

    # Free evolution
    if tau > 0:
        tlist_free = np.linspace(0, tau, 30)
        nv_optical.evolve_master_equation(tlist_free, B_field, laser_on=False)

    # Second π/2 pulse
    nv_optical.evolve_master_equation(tlist_pi2, B_field, laser_on=False, mw_params=mw_params)

    # Readout
    pops_final = nv_optical.get_ground_state_populations()
    ramsey_optical.append(pops_final[1])

ax5.plot(taus_ramsey[::2], ramsey_optical, 'ro-', linewidth=2, markersize=6,
        label='Optical (with pumping imperfection)')
ax5.set_xlabel('Free Evolution Time (µs)')
ax5.set_ylabel('Population in |0⟩')
ax5.set_title('Ramsey: Coherent vs Optical Model')
ax5.legend()
ax5.grid(True, alpha=0.3)

print(f"   Optical pumping fidelity affects Ramsey contrast")

# ============================================================================
# Example 6: Decision Tree for Model Selection
# ============================================================================
print("\n6. Creating model selection guide...")

ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

decision_text = """
MODEL SELECTION GUIDE
=====================

Use COHERENT Model (nv_center.py):
───────────────────────────────────
✓ Pure MW pulse sequences
✓ Coherent control optimization
✓ Ideal pulse simulations
✓ Fast prototyping
✓ Long time evolution (>10 µs)
✓ No optical readout needed

Speed: Fast (~1-10ms)
Accuracy: Ideal (no dissipation)

Use OPTICAL Model (optical_dynamics.py):
────────────────────────────────────────
✓ ODMR simulations
✓ Optical pumping / initialization
✓ PL contrast calculations
✓ Realistic readout
✓ Spin-dependent fluorescence
✓ Short optical cycles (<5 µs)

Speed: Slower (~0.1-1s per point)
Accuracy: Realistic (includes dissipation)

HYBRID APPROACH:
────────────────
1. Use coherent model for MW control
2. Model optical pumping with optical model
3. Assume realistic contrast reduction
4. Fastest + reasonably realistic
"""

ax6.text(0.1, 0.95, decision_text, transform=ax6.transAxes,
        fontsize=9, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# ============================================================================
# Performance Comparison Table
# ============================================================================

print("\n" + "=" * 70)
print("Performance Comparison")
print("=" * 70)

print(f"\nComputation times:")
print(f"  {'Task':<30} {'Coherent':<15} {'Optical':<15} {'Ratio':<10}")
print(f"  {'-'*70}")
print(f"  {'Rabi oscillation (100 pts)':<30} {t_coherent:<15.4f} {t_optical:<15.4f} {t_optical/t_coherent:<10.1f}x")
print(f"  {'ODMR spectrum (100 pts)':<30} {t_odmr_coherent:<15.4f} {t_odmr_optical:<15.4f} {t_odmr_optical/t_odmr_coherent:<10.1f}x")

print(f"\nModel characteristics:")
print(f"  {'Property':<30} {'Coherent':<20} {'Optical':<20}")
print(f"  {'-'*70}")
print(f"  {'Hilbert space dimension':<30} {'3':<20} {'8':<20}")
print(f"  {'Evolution equation':<30} {'Schrödinger':<20} {'Lindblad master':<20}")
print(f"  {'Dissipation':<30} {'No':<20} {'Yes (ISC, decay)':<20}")
print(f"  {'PL contrast':<30} {'Ideal (100%)':<20} {f'Realistic ({contrast_optical:.0f}%)':<20}")
print(f"  {'Optical pumping':<30} {'N/A':<20} {'Included':<20}")

print(f"\nRecommendations:")
print(f"  • Coherent model: Use for MW pulse optimization and long sequences")
print(f"  • Optical model: Use for ODMR, initialization, and realistic readout")
print(f"  • For most accuracy: Combine both models in workflow")

# ============================================================================
# Save and Summary
# ============================================================================

plt.tight_layout()
plt.savefig('optical_vs_coherent.png', dpi=150, bbox_inches='tight')
print("\n✓ Plot saved as 'optical_vs_coherent.png'")

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)

print("\n✓ Key insights:")
print("  • Coherent model: 10-100x faster, ideal for MW control")
print("  • Optical model: Realistic PL contrast and dissipation")
print("  • ODMR requires optical model for accurate contrast")
print("  • Rabi/Ramsey can use either (coherent faster)")
print("  • Optical pumping only in optical model")

print("\n✓ Typical workflow:")
print("  1. Design pulse sequence with coherent model")
print("  2. Validate with optical model for realism")
print("  3. Account for ~30% PL contrast in experiments")
print("  4. Use optical pumping time ~1-2 µs")

print("\n" + "=" * 70)

plt.show()
