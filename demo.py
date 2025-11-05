#!/usr/bin/env python3
"""
Quick Demo of NV-Center Simulator
==================================

Run this script for a quick demonstration of the simulator capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from nv_center import NVCenter, NVParameters, PulseSequence

def main():
    print("=" * 70)
    print("NV-Center in Diamond Simulator - Quick Demo")
    print("=" * 70)

    # Initialize NV-center
    print("\n1. Initializing NV-center with realistic parameters...")
    params = NVParameters(
        D=2.870,          # GHz
        omega_rabi=10.0,  # MHz
        T2=20.0,          # µs
        gamma=2.8025      # MHz/G
    )
    nv = NVCenter(params=params, include_nuclear=False)
    print(f"   Zero-field splitting D = {params.D} GHz")
    print(f"   Coherence time T2 = {params.T2} µs")
    print(f"   Rabi frequency Ω = {params.omega_rabi} MHz")

    # Set magnetic field
    B_field = np.array([0.0, 0.0, 10.0])  # Gauss
    print(f"\n2. Setting magnetic field: B = {B_field} G")

    # Calculate Zeeman splitting
    gamma_GHz = params.gamma / 1000.0  # Convert to GHz/G
    splitting = gamma_GHz * B_field[2]
    print(f"   Expected Zeeman splitting: {splitting*1000:.2f} MHz")

    # Create pulse sequence
    ps = PulseSequence(nv, B_field)

    # Run quick simulations
    print("\n3. Running simulations...")

    # Rabi oscillation
    print("   a) Rabi oscillation (π pulse calibration)...")
    duration_max = 0.5  # µs
    times_rabi, pops_rabi = ps.rabi_oscillation(duration_max, n_points=100)

    # Find π pulse time
    min_idx = np.argmin(pops_rabi)
    pi_pulse_time = times_rabi[min_idx]
    print(f"      → π pulse time: {pi_pulse_time*1000:.2f} ns")

    # Ramsey
    print("   b) Ramsey sequence (T2* measurement)...")
    tau_max_ramsey = 10.0  # µs
    taus_ramsey, pops_ramsey = ps.ramsey_sequence(tau_max_ramsey, n_points=100)
    print(f"      → Oscillations due to decoherence")

    # ODMR
    print("   c) ODMR spectrum (magnetic field measurement)...")
    freq_center = params.D
    freq_range = 0.05  # GHz
    freqs_odmr, contrast_odmr = ps.odmr_spectrum(
        freq_center - freq_range,
        freq_center + freq_range,
        n_points=150,
        pulse_duration=3.0
    )
    print(f"      → Scanned {len(freqs_odmr)} frequency points")

    # Create visualization
    print("\n4. Creating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Rabi
    ax1 = axes[0, 0]
    ax1.plot(times_rabi * 1000, pops_rabi, 'b-', linewidth=2)
    ax1.axvline(pi_pulse_time * 1000, color='r', linestyle='--',
               label=f'π pulse: {pi_pulse_time*1000:.1f} ns')
    ax1.set_xlabel('Pulse Duration (ns)')
    ax1.set_ylabel('Population in |0⟩')
    ax1.set_title('Rabi Oscillation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Ramsey
    ax2 = axes[0, 1]
    ax2.plot(taus_ramsey, pops_ramsey, 'g-', linewidth=2)
    ax2.set_xlabel('Free Evolution Time τ (µs)')
    ax2.set_ylabel('Population in |0⟩')
    ax2.set_title('Ramsey Interferometry')
    ax2.grid(True, alpha=0.3)

    # Plot 3: ODMR
    ax3 = axes[1, 0]
    ax3.plot(freqs_odmr, contrast_odmr, 'r-', linewidth=2)
    ax3.axvline(params.D, color='blue', linestyle='--', alpha=0.5,
               label=f'D = {params.D} GHz')
    ax3.set_xlabel('MW Frequency (GHz)')
    ax3.set_ylabel('ODMR Contrast')
    ax3.set_title('ODMR Spectrum')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: State visualization
    ax4 = axes[1, 1]
    nv.reset_to_ground()
    nv.apply_pulse('pi/2_x', 0.05, B_field)

    # Track state during π/2 - π - π/2 sequence
    states_demo = []
    times_demo = []

    nv.reset_to_ground()
    states_demo.append(nv.measure_expectation(nv.Sz))
    times_demo.append(0)

    nv.apply_pulse('pi/2_x', 0.05, B_field)
    states_demo.append(nv.measure_expectation(nv.Sz))
    times_demo.append(1)

    for i in range(5):
        H = nv.hamiltonian_static(B_field)
        nv.evolve(H, 0.5)
        states_demo.append(nv.measure_expectation(nv.Sz))
        times_demo.append(2 + i)

    nv.apply_pulse('pi_x', 0.1, B_field)
    states_demo.append(nv.measure_expectation(nv.Sz))
    times_demo.append(7)

    ax4.plot(times_demo, states_demo, 'mo-', linewidth=2, markersize=8)
    ax4.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax4.set_xlabel('Sequence Step')
    ax4.set_ylabel('⟨Sz⟩')
    ax4.set_title('State Evolution: π/2 - Evolution - π')
    ax4.grid(True, alpha=0.3)

    plt.suptitle('NV-Center Simulator Demo', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save figure
    output_file = 'demo_output.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   Plot saved as '{output_file}'")

    # Summary
    print("\n" + "=" * 70)
    print("Demo Summary")
    print("=" * 70)
    print("\n✓ Successfully demonstrated:")
    print("  • NV-center initialization with realistic parameters")
    print("  • Magnetic field configuration and Zeeman splitting")
    print("  • Rabi oscillations for pulse calibration")
    print("  • Ramsey interferometry for coherence measurement")
    print("  • ODMR spectroscopy for magnetic field sensing")
    print("  • Quantum state manipulation and tracking")
    print("\n✓ Generated plots showing:")
    print("  • Coherent spin oscillations")
    print("  • Quantum interference effects")
    print("  • Resonance spectroscopy")
    print("  • State evolution under control")

    print("\n" + "=" * 70)
    print("Next Steps:")
    print("=" * 70)
    print("\n1. Run individual example scripts:")
    print("   python example_rabi.py")
    print("   python example_ramsey.py")
    print("   python example_hahn_echo.py")
    print("   python example_odmr.py")
    print("   python example_bfield_dynamics.py")
    print("\n2. Explore the API:")
    print("   from nv_center import NVCenter, PulseSequence")
    print("   help(NVCenter)")
    print("\n3. Read the documentation:")
    print("   See README.md for detailed API reference")
    print("\n" + "=" * 70)

    plt.show()

if __name__ == "__main__":
    main()
