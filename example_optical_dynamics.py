"""
Example: Optical Dynamics and PL Contrast
=========================================

Demonstrates the full NV-center level structure including:
- Optical pumping dynamics
- Spin-dependent intersystem crossing (ISC)
- Realistic photoluminescence (PL) contrast
- ODMR with proper optical readout
- Comparison with 3-level model

Physics:
--------
The NV-center exhibits spin-dependent fluorescence due to:
1. Optical excitation: ³A₂ → ³E (spin-preserving)
2. Radiative decay: ³E → ³A₂ (produces fluorescence)
3. ISC pathway: ³E → ¹A₁ → ¹E → ³A₂ (spin-dependent, non-radiative)
   - ms=0: Low ISC rate → High fluorescence
   - ms=±1: High ISC rate → Low fluorescence (PL contrast ~30%)

Applications:
- Optical spin initialization to ms=0
- Spin-dependent fluorescence readout
- Realistic ODMR simulation
- Understanding optical dynamics
"""

import numpy as np
import matplotlib.pyplot as plt
from nv_center import NVParameters
from optical_dynamics import (
    ExtendedNVCenter,
    OpticalParameters,
    simulate_ODMR_with_PL_contrast
)

print("Optical Dynamics and PL Contrast Simulation")
print("=" * 70)

# Initialize parameters
nv_params = NVParameters(
    D=2.870,  # GHz
    gamma=2.8025,  # MHz/G
    T2=50.0  # µs
)

optical_params = OpticalParameters(
    gamma_laser=50.0,  # MHz (moderate laser power)
    gamma_radiative=83.0,  # MHz (12 ns lifetime)
    gamma_ISC_0=10.0,  # MHz (slow ISC for ms=0)
    gamma_ISC_pm1=100.0,  # MHz (fast ISC for ms=±1)
    gamma_singlet_A=3.0,  # MHz
    gamma_singlet_E=5.0,  # MHz
    singlet_branching=(0.1, 0.8, 0.1)  # Preferential decay to ms=0
)

# ============================================================================
# Example 1: Optical Pumping Dynamics
# ============================================================================
print("\n1. Simulating optical pumping dynamics...")

fig = plt.figure(figsize=(16, 12))

nv = ExtendedNVCenter(nv_params, optical_params)

# Start in thermal mixture (equal populations)
initial_state = (
    0.33 * nv.proj['g,-1'] +
    0.34 * nv.proj['g,0'] +
    0.33 * nv.proj['g,+1']
).unit()
nv.rho = initial_state

# Apply optical pumping
B_field = np.array([0, 0, 5.0])  # Gauss
pump_duration = 10.0  # µs
n_points = 200

tlist = np.linspace(0, pump_duration, n_points)

# Track populations of all levels
e_ops = [nv.proj[name] for name in nv.level_names]

result = nv.evolve_master_equation(tlist, B_field, laser_on=True, e_ops=e_ops)

# Plot ground state populations
ax1 = plt.subplot(3, 3, 1)
ax1.plot(tlist, result.expect[nv.g_minus], 'b-', linewidth=2, label='|g,-1⟩')
ax1.plot(tlist, result.expect[nv.g_zero], 'g-', linewidth=2, label='|g,0⟩')
ax1.plot(tlist, result.expect[nv.g_plus], 'r-', linewidth=2, label='|g,+1⟩')
ax1.set_xlabel('Time (µs)')
ax1.set_ylabel('Population')
ax1.set_title('Optical Pumping: Ground State')
ax1.legend()
ax1.grid(True, alpha=0.3)

print(f"   Initial: P(g,0) = {result.expect[nv.g_zero][0]:.3f}")
print(f"   Final: P(g,0) = {result.expect[nv.g_zero][-1]:.3f}")
print(f"   Pumping efficiency: {result.expect[nv.g_zero][-1]/0.34:.1f}x")

# Plot excited state populations
ax2 = plt.subplot(3, 3, 2)
ax2.plot(tlist, result.expect[nv.e_minus], 'b-', linewidth=2, label='|e,-1⟩')
ax2.plot(tlist, result.expect[nv.e_zero], 'g-', linewidth=2, label='|e,0⟩')
ax2.plot(tlist, result.expect[nv.e_plus], 'r-', linewidth=2, label='|e,+1⟩')
ax2.set_xlabel('Time (µs)')
ax2.set_ylabel('Population')
ax2.set_title('Optical Pumping: Excited State')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot singlet populations
ax3 = plt.subplot(3, 3, 3)
ax3.plot(tlist, result.expect[nv.s_A], 'purple', linewidth=2, label='|sA⟩')
ax3.plot(tlist, result.expect[nv.s_E], 'orange', linewidth=2, label='|sE⟩')
ax3.set_xlabel('Time (µs)')
ax3.set_ylabel('Population')
ax3.set_title('Optical Pumping: Singlet States')
ax3.legend()
ax3.grid(True, alpha=0.3)

# ============================================================================
# Example 2: Spin-Dependent Fluorescence
# ============================================================================
print("\n2. Measuring spin-dependent fluorescence (PL contrast)...")

ax4 = plt.subplot(3, 3, 4)
ax5 = plt.subplot(3, 3, 5)

# Measure PL for each spin state
spin_states = [-1, 0, 1]
PL_values = []
steady_state_excited = []

for ms in spin_states:
    nv.reset_to_ground(ms=ms)

    # Optical pumping to steady state
    tlist_ss = np.linspace(0, 5.0, 100)
    e_ops_ss = [nv.proj['e,-1'], nv.proj['e,0'], nv.proj['e,+1']]

    result_ss = nv.evolve_master_equation(tlist_ss, B_field, laser_on=True, e_ops=e_ops_ss)

    # Total excited state population at steady state
    excited_ss = result_ss.expect[0][-1] + result_ss.expect[1][-1] + result_ss.expect[2][-1]
    steady_state_excited.append(excited_ss)

    # Measure PL
    PL = nv.measure_PL(integration_time=1.0, B_field=B_field)
    PL_values.append(PL)

    print(f"   ms={ms:+d}: PL = {PL:.3f}, Excited pop = {excited_ss:.4f}")

# Plot PL vs spin state
ax4.bar(spin_states, PL_values, color=['blue', 'green', 'red'], alpha=0.7, edgecolor='black')
ax4.set_xlabel('Spin State ms')
ax4.set_ylabel('PL Signal (arb. units)')
ax4.set_title('Spin-Dependent Fluorescence')
ax4.set_xticks(spin_states)
ax4.grid(True, alpha=0.3, axis='y')

# Calculate contrast
PL_bright = PL_values[1]  # ms=0
PL_dark = (PL_values[0] + PL_values[2]) / 2  # Average of ms=±1
contrast = (PL_bright - PL_dark) / PL_bright * 100

ax4.text(0, max(PL_values) * 0.9, f'Contrast: {contrast:.1f}%',
        ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

print(f"   PL contrast: {contrast:.1f}%")

# Plot steady-state excited populations
ax5.bar(spin_states, steady_state_excited, color=['blue', 'green', 'red'], alpha=0.7, edgecolor='black')
ax5.set_xlabel('Spin State ms')
ax5.set_ylabel('Excited State Population')
ax5.set_title('Steady-State Excited Population')
ax5.set_xticks(spin_states)
ax5.grid(True, alpha=0.3, axis='y')

# ============================================================================
# Example 3: Realistic ODMR with PL Contrast
# ============================================================================
print("\n3. Simulating realistic ODMR spectrum...")

ax6 = plt.subplot(3, 3, 6)

B_field_odmr = np.array([0, 0, 10.0])  # Gauss

# Calculate expected transition frequencies
gamma_MHz_G = nv_params.gamma
B_z = B_field_odmr[2]
splitting = gamma_MHz_G * B_z / 1000.0  # GHz

f_minus = nv_params.D - splitting
f_plus = nv_params.D + splitting

print(f"   Expected transitions:")
print(f"     |0⟩ ↔ |-1⟩: {f_minus:.4f} GHz")
print(f"     |0⟩ ↔ |+1⟩: {f_plus:.4f} GHz")

# Simulate ODMR
freq_min = nv_params.D - 0.05
freq_max = nv_params.D + 0.05

freqs, PL_signal = simulate_ODMR_with_PL_contrast(
    nv_params, optical_params, B_field_odmr,
    (freq_min, freq_max),
    n_points=150,
    mw_duration=5.0,
    mw_power=10.0
)

# Plot ODMR
ax6.plot(freqs, PL_signal, 'b-', linewidth=2)
ax6.axvline(f_minus, color='r', linestyle='--', alpha=0.5, label=f'f₋={f_minus:.4f} GHz')
ax6.axvline(f_plus, color='g', linestyle='--', alpha=0.5, label=f'f₊={f_plus:.4f} GHz')
ax6.set_xlabel('MW Frequency (GHz)')
ax6.set_ylabel('Normalized PL')
ax6.set_title('Realistic ODMR Spectrum')
ax6.legend()
ax6.grid(True, alpha=0.3)
ax6.invert_yaxis()  # Convention: dips point down

# Find dip depth
min_PL = np.min(PL_signal)
odmr_contrast = (1 - min_PL) * 100
print(f"   ODMR contrast: {odmr_contrast:.1f}%")

# ============================================================================
# Example 4: ISC Rate Dependence on Spin
# ============================================================================
print("\n4. Analyzing ISC rate dependence...")

ax7 = plt.subplot(3, 3, 7)

# Vary ISC rate ratio
ISC_ratios = np.linspace(1, 20, 20)
contrasts = []

for ratio in ISC_ratios:
    opt_params_temp = OpticalParameters(
        gamma_laser=50.0,
        gamma_radiative=83.0,
        gamma_ISC_0=10.0,
        gamma_ISC_pm1=10.0 * ratio,  # Scale ISC for ms=±1
        gamma_singlet_A=3.0,
        gamma_singlet_E=5.0,
        singlet_branching=(0.1, 0.8, 0.1)
    )

    nv_temp = ExtendedNVCenter(nv_params, opt_params_temp)

    # Measure PL for ms=0 and ms=±1
    nv_temp.reset_to_ground(ms=0)
    PL_0 = nv_temp.measure_PL(integration_time=1.0, B_field=B_field)

    nv_temp.reset_to_ground(ms=1)
    PL_1 = nv_temp.measure_PL(integration_time=1.0, B_field=B_field)

    contrast_temp = (PL_0 - PL_1) / PL_0 * 100
    contrasts.append(contrast_temp)

ax7.plot(ISC_ratios, contrasts, 'bo-', linewidth=2, markersize=6)
ax7.axhline(contrast, color='r', linestyle='--', alpha=0.5,
           label=f'Default (ratio={optical_params.gamma_ISC_pm1/optical_params.gamma_ISC_0:.0f})')
ax7.set_xlabel('ISC Rate Ratio (γ±₁ / γ₀)')
ax7.set_ylabel('PL Contrast (%)')
ax7.set_title('Contrast vs ISC Rate Ratio')
ax7.legend()
ax7.grid(True, alpha=0.3)

print(f"   Contrast saturates at ~{contrasts[-1]:.1f}% for high ISC ratio")

# ============================================================================
# Example 5: Laser Power Dependence
# ============================================================================
print("\n5. Studying laser power dependence...")

ax8 = plt.subplot(3, 3, 8)

laser_powers = np.logspace(0, 3, 20)  # 1 to 1000 MHz
PL_vs_power_0 = []
PL_vs_power_1 = []

for power in laser_powers:
    opt_params_power = OpticalParameters(
        gamma_laser=power,
        gamma_radiative=optical_params.gamma_radiative,
        gamma_ISC_0=optical_params.gamma_ISC_0,
        gamma_ISC_pm1=optical_params.gamma_ISC_pm1,
        gamma_singlet_A=optical_params.gamma_singlet_A,
        gamma_singlet_E=optical_params.gamma_singlet_E,
        singlet_branching=optical_params.singlet_branching
    )

    nv_power = ExtendedNVCenter(nv_params, opt_params_power)

    # ms=0
    nv_power.reset_to_ground(ms=0)
    PL_0 = nv_power.measure_PL(integration_time=0.5, B_field=B_field)
    PL_vs_power_0.append(PL_0)

    # ms=1
    nv_power.reset_to_ground(ms=1)
    PL_1 = nv_power.measure_PL(integration_time=0.5, B_field=B_field)
    PL_vs_power_1.append(PL_1)

ax8.loglog(laser_powers, PL_vs_power_0, 'g-', linewidth=2, label='ms=0 (bright)')
ax8.loglog(laser_powers, PL_vs_power_1, 'b-', linewidth=2, label='ms=1 (dark)')
ax8.set_xlabel('Laser Power (MHz excitation rate)')
ax8.set_ylabel('PL Signal (arb. units)')
ax8.set_title('PL Saturation vs Laser Power')
ax8.legend()
ax8.grid(True, alpha=0.3, which='both')

# ============================================================================
# Example 6: Time-Resolved Fluorescence
# ============================================================================
print("\n6. Time-resolved fluorescence decay...")

ax9 = plt.subplot(3, 3, 9)

# Turn off laser and watch excited state decay
nv.reset_to_ground(ms=0)

# First, pump to steady state
tlist_pump = np.linspace(0, 5.0, 50)
nv.evolve_master_equation(tlist_pump, B_field, laser_on=True)

# Then turn off laser and watch decay
decay_time = 0.5  # µs
tlist_decay = np.linspace(0, decay_time, 200)

e_ops_decay = [
    nv.proj['e,-1'] + nv.proj['e,0'] + nv.proj['e,+1']  # Total excited
]

result_decay = nv.evolve_master_equation(tlist_decay, B_field, laser_on=False, e_ops=e_ops_decay)

# Plot decay
excited_pop_decay = result_decay.expect[0]
ax9.semilogy(tlist_decay * 1000, excited_pop_decay, 'r-', linewidth=2, label='Data')

# Fit exponential
from scipy.optimize import curve_fit

def exp_decay(t, A, tau):
    return A * np.exp(-t / tau)

try:
    popt, _ = curve_fit(exp_decay, tlist_decay, excited_pop_decay, p0=[excited_pop_decay[0], 0.01])
    A_fit, tau_fit = popt

    ax9.semilogy(tlist_decay * 1000, exp_decay(tlist_decay, *popt), 'b--',
                linewidth=2, label=f'Fit: τ={tau_fit*1000:.1f} ns')

    print(f"   Excited state lifetime: {tau_fit*1000:.1f} ns")
    print(f"   Expected: {1000.0/optical_params.gamma_radiative:.1f} ns")
except:
    print("   Fit failed")

ax9.set_xlabel('Time (ns)')
ax9.set_ylabel('Excited State Population')
ax9.set_title('Time-Resolved Fluorescence Decay')
ax9.legend()
ax9.grid(True, alpha=0.3, which='both')

# ============================================================================
# Summary and Save
# ============================================================================

plt.suptitle('Optical Dynamics and PL Contrast', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('optical_dynamics.png', dpi=150, bbox_inches='tight')
print("\n✓ Plot saved as 'optical_dynamics.png'")

print("\n" + "=" * 70)
print("Summary: Optical Dynamics")
print("=" * 70)

print("\n✓ Demonstrated phenomena:")
print("  • Optical pumping to ms=0 (spin initialization)")
print("  • Spin-dependent fluorescence (PL contrast)")
print("  • Realistic ODMR with proper optical readout")
print("  • ISC rate ratio determines contrast")
print("  • PL saturation with laser power")
print("  • Time-resolved fluorescence decay")

print("\n✓ Key physics:")
print("  • Optical cycle: ³A₂ → ³E → ³A₂ (radiative)")
print("  • ISC pathway: ³E → ¹A₁ → ¹E → ³A₂ (non-radiative)")
print("  • Spin-selective ISC: γ(±1) >> γ(0)")
print("  • Singlet preferentially decays to ms=0")

print("\n✓ Measured parameters:")
print(f"  • PL contrast: {contrast:.1f}% (typical: 20-30%)")
print(f"  • ODMR contrast: {odmr_contrast:.1f}%")
print(f"  • Optical pumping: {result.expect[nv.g_zero][-1]*100:.1f}% in ms=0")
print(f"  • Excited lifetime: ~{1000.0/optical_params.gamma_radiative:.0f} ns")

print("\n✓ Applications:")
print("  • Optical spin initialization for quantum sensing")
print("  • Spin-dependent readout for quantum computing")
print("  • ODMR magnetometry with realistic SNR")
print("  • Understanding optical cycle for pulse optimization")

print("\n" + "=" * 70)

plt.show()
