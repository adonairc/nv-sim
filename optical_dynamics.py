"""
Extended NV-Center Level Structure with Optical Dynamics
========================================================

This module implements the full NV-center level structure including:
- Ground state triplet ³A₂ (ms = -1, 0, +1)
- Excited state triplet ³E (ms = -1, 0, +1)
- Singlet states ¹A₁ and ¹E
- Intersystem crossing (ISC) pathways
- Optical excitation and emission
- Quantum master equation (Lindblad form)
- Realistic photoluminescence (PL) contrast

Physics:
--------
The NV-center has a complex level structure that enables optical spin readout:

1. Optical excitation: ³A₂ → ³E (532 nm laser, spin-preserving)
2. Radiative decay: ³E → ³A₂ (637-800 nm, produces fluorescence)
3. ISC pathway: ³E → ¹A₁ → ¹E → ³A₂ (non-radiative, spin-dependent)
   - ISC rate is ~10× higher for ms=±1 than ms=0
   - This creates spin-dependent PL contrast

PL Contrast:
-----------
- ms=0: High fluorescence (low ISC rate → more radiative decay)
- ms=±1: Low fluorescence (high ISC rate → less radiative decay)
- Typical contrast: 20-30% at room temperature

Author: Claude AI
Date: 2025-11-05
"""

import numpy as np
import qutip as qt
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import matplotlib.pyplot as plt

# Import base NV parameters
from nv_center import NVParameters


@dataclass
class OpticalParameters:
    """Parameters for optical transitions and level structure"""

    # Optical excitation rate (MHz) - proportional to laser power
    gamma_laser: float = 50.0  # MHz at ~1 mW focused laser

    # Excited state radiative decay rate (MHz)
    # Lifetime ~12 ns → rate = 1/(12 ns) ≈ 83 MHz
    gamma_radiative: float = 83.0  # MHz

    # Intersystem crossing rates (MHz)
    # ms=0: slower ISC
    gamma_ISC_0: float = 10.0  # MHz
    # ms=±1: faster ISC (spin-selective)
    gamma_ISC_pm1: float = 100.0  # MHz

    # Singlet state decay rates (MHz)
    # ¹A₁ lifetime ~300 ns → rate ≈ 3 MHz
    gamma_singlet_A: float = 3.0  # MHz
    # ¹E lifetime ~200 ns → rate ≈ 5 MHz
    gamma_singlet_E: float = 5.0  # MHz

    # Singlet ¹E → ³A₂ branching ratios (ms = -1, 0, +1)
    # Decay from singlet preferentially populates ms=0
    singlet_branching: Tuple[float, float, float] = (0.1, 0.8, 0.1)

    # Zero-field splitting in excited state (GHz)
    # Slightly different from ground state
    D_excited: float = 1.42  # GHz (approximately half of ground state)

    # Singlet state energies (GHz, relative to ground state)
    E_singlet_A: float = 1.19  # GHz
    E_singlet_E: float = 1.40  # GHz

    # Dephasing rates (MHz) - pure dephasing in excited state
    gamma_dephasing_excited: float = 100.0  # MHz
    gamma_dephasing_ground: float = 1.0  # MHz (much slower)


class ExtendedNVCenter:
    """
    Full NV-center level structure with optical dynamics.

    Level structure:
    ----------------
    Ground state ³A₂: |g,-1⟩, |g,0⟩, |g,+1⟩
    Excited state ³E: |e,-1⟩, |e,0⟩, |e,+1⟩
    Singlet ¹A₁: |sA⟩ (single level)
    Singlet ¹E: |sE,-⟩, |sE,+⟩ (doublet, but often treated as single)

    Total: 9 levels (3 ground + 3 excited + 1 + 2 singlet)

    For computational efficiency, singlets often combined: 8 levels
    """

    def __init__(self,
                 nv_params: Optional[NVParameters] = None,
                 optical_params: Optional[OpticalParameters] = None,
                 include_singlet_structure: bool = False):
        """
        Initialize extended NV-center with optical levels.

        Parameters
        ----------
        nv_params : NVParameters
            Ground state parameters
        optical_params : OpticalParameters
            Optical transition parameters
        include_singlet_structure : bool
            If True, use detailed singlet structure (9 levels)
            If False, use simplified singlet (8 levels) - faster
        """
        self.nv_params = nv_params if nv_params else NVParameters()
        self.optical_params = optical_params if optical_params else OpticalParameters()
        self.include_singlet_structure = include_singlet_structure

        # Build level structure
        self._build_level_structure()

        # Build operators
        self._build_operators()

        # Initialize in ground state ms=0
        self.reset_to_ground()

    def _build_level_structure(self):
        """Construct Hilbert space for all levels"""

        if self.include_singlet_structure:
            # Full structure: 3 ground + 3 excited + 1 singlet A + 2 singlet E = 9
            self.n_levels = 9
            self.level_names = [
                'g,-1', 'g,0', 'g,+1',      # Ground triplet
                'e,-1', 'e,0', 'e,+1',      # Excited triplet
                'sA',                        # Singlet A
                'sE,-', 'sE,+'               # Singlet E (doublet)
            ]
        else:
            # Simplified: 3 ground + 3 excited + 2 singlet combined = 8
            self.n_levels = 8
            self.level_names = [
                'g,-1', 'g,0', 'g,+1',      # Ground triplet
                'e,-1', 'e,0', 'e,+1',      # Excited triplet
                'sA',                        # Singlet A
                'sE'                         # Singlet E (combined)
            ]

        # Define level indices
        self.g_minus = 0
        self.g_zero = 1
        self.g_plus = 2
        self.e_minus = 3
        self.e_zero = 4
        self.e_plus = 5
        self.s_A = 6
        self.s_E = 7
        if self.include_singlet_structure:
            self.s_E_plus = 8

    def _build_operators(self):
        """Build operators for the extended level structure"""

        n = self.n_levels

        # Identity
        self.I = qt.qeye(n)

        # Projection operators for each level
        self.proj = {}
        for i, name in enumerate(self.level_names):
            self.proj[name] = qt.basis(n, i) * qt.basis(n, i).dag()

        # Ground state spin operators (acting on ground manifold)
        # S_z in ground state subspace
        Sz_ground = qt.Qobj(np.zeros((n, n)))
        Sz_ground[self.g_minus, self.g_minus] = -1
        Sz_ground[self.g_zero, self.g_zero] = 0
        Sz_ground[self.g_plus, self.g_plus] = 1
        self.Sz_ground = qt.Qobj(Sz_ground)

        # S_x, S_y for ground state
        Sx_ground = qt.Qobj(np.zeros((n, n), dtype=complex))
        Sx_ground[self.g_minus, self.g_zero] = 1/np.sqrt(2)
        Sx_ground[self.g_zero, self.g_minus] = 1/np.sqrt(2)
        Sx_ground[self.g_zero, self.g_plus] = 1/np.sqrt(2)
        Sx_ground[self.g_plus, self.g_zero] = 1/np.sqrt(2)
        self.Sx_ground = qt.Qobj(Sx_ground)

        Sy_ground = qt.Qobj(np.zeros((n, n), dtype=complex))
        Sy_ground[self.g_minus, self.g_zero] = -1j/np.sqrt(2)
        Sy_ground[self.g_zero, self.g_minus] = 1j/np.sqrt(2)
        Sy_ground[self.g_zero, self.g_plus] = -1j/np.sqrt(2)
        Sy_ground[self.g_plus, self.g_zero] = 1j/np.sqrt(2)
        self.Sy_ground = qt.Qobj(Sy_ground)

        # Excited state spin operators
        Sz_excited = qt.Qobj(np.zeros((n, n)))
        Sz_excited[self.e_minus, self.e_minus] = -1
        Sz_excited[self.e_zero, self.e_zero] = 0
        Sz_excited[self.e_plus, self.e_plus] = 1
        self.Sz_excited = qt.Qobj(Sz_excited)

        # Optical transition operators (spin-preserving)
        # |g,ms⟩ ↔ |e,ms⟩
        self.sigma_minus_1 = qt.basis(n, self.g_minus) * qt.basis(n, self.e_minus).dag()
        self.sigma_0 = qt.basis(n, self.g_zero) * qt.basis(n, self.e_zero).dag()
        self.sigma_plus_1 = qt.basis(n, self.g_plus) * qt.basis(n, self.e_plus).dag()

        # Total optical transition operator
        self.sigma_optical = self.sigma_minus_1 + self.sigma_0 + self.sigma_plus_1

        # ISC operators: |e,ms⟩ → |sA⟩
        self.ISC_minus = qt.basis(n, self.s_A) * qt.basis(n, self.e_minus).dag()
        self.ISC_0 = qt.basis(n, self.s_A) * qt.basis(n, self.e_zero).dag()
        self.ISC_plus = qt.basis(n, self.s_A) * qt.basis(n, self.e_plus).dag()

        # Singlet cascade: |sA⟩ → |sE⟩
        self.singlet_decay_AE = qt.basis(n, self.s_E) * qt.basis(n, self.s_A).dag()

        # Singlet to ground: |sE⟩ → |g,ms⟩ (with branching)
        br = self.optical_params.singlet_branching
        self.singlet_to_ground = (
            np.sqrt(br[0]) * qt.basis(n, self.g_minus) * qt.basis(n, self.s_E).dag() +
            np.sqrt(br[1]) * qt.basis(n, self.g_zero) * qt.basis(n, self.s_E).dag() +
            np.sqrt(br[2]) * qt.basis(n, self.g_plus) * qt.basis(n, self.s_E).dag()
        )

    def hamiltonian_ground(self, B_field: np.ndarray) -> qt.Qobj:
        """
        Hamiltonian for ground state manifold.

        H_ground = D * Sz² + γ * B · S (on ground state)
        """
        D = self.nv_params.D  # GHz
        gamma = self.nv_params.gamma / 1000.0  # GHz/G

        # Zero-field splitting: D * Sz²
        H_zfs = D * self.Sz_ground * self.Sz_ground

        # Zeeman term
        Bx, By, Bz = B_field
        H_zeeman = gamma * (Bx * self.Sx_ground + By * self.Sy_ground + Bz * self.Sz_ground)

        return H_zfs + H_zeeman

    def hamiltonian_excited(self, B_field: np.ndarray) -> qt.Qobj:
        """
        Hamiltonian for excited state manifold.

        Similar to ground but with different D parameter.
        """
        D_ex = self.optical_params.D_excited  # GHz
        gamma = self.nv_params.gamma / 1000.0  # GHz/G

        H_zfs = D_ex * self.Sz_excited * self.Sz_excited

        Bx, By, Bz = B_field
        H_zeeman = gamma * (Bx * self.Sz_excited + By * self.Sz_excited + Bz * self.Sz_excited)

        return H_zfs + H_zeeman

    def hamiltonian_full(self, B_field: np.ndarray, omega_mw: float = 0,
                        mw_phase: float = 0, mw_amplitude: float = 0) -> qt.Qobj:
        """
        Full Hamiltonian including ground, excited, singlets, and MW drive.

        Parameters
        ----------
        B_field : array
            Static magnetic field [Bx, By, Bz] in Gauss
        omega_mw : float
            MW frequency in GHz
        mw_phase : float
            MW phase in radians
        mw_amplitude : float
            MW Rabi frequency in MHz
        """
        H = self.hamiltonian_ground(B_field) + self.hamiltonian_excited(B_field)

        # MW driving on ground state (in rotating frame)
        if mw_amplitude > 0:
            omega_rabi_GHz = mw_amplitude / 1000.0  # MHz to GHz
            H_mw = (omega_rabi_GHz / 2.0) * (
                np.cos(mw_phase) * self.Sx_ground +
                np.sin(mw_phase) * self.Sy_ground
            )
            H = H - omega_mw * self.Sz_ground + H_mw

        return H

    def collapse_operators(self, laser_on: bool = True) -> List[qt.Qobj]:
        """
        Build collapse operators for Lindblad master equation.

        Returns list of collapse operators for:
        - Optical excitation (if laser on)
        - Radiative decay
        - ISC (spin-dependent)
        - Singlet cascades
        - Dephasing
        """
        c_ops = []

        # Laser excitation: |g⟩ → |e⟩
        if laser_on:
            gamma_laser = self.optical_params.gamma_laser  # MHz
            rate_GHz = gamma_laser / 1000.0  # Convert to GHz
            c_ops.append(np.sqrt(rate_GHz) * self.sigma_optical.dag())

        # Radiative decay: |e⟩ → |g⟩ (spin-preserving)
        gamma_rad = self.optical_params.gamma_radiative / 1000.0  # GHz
        c_ops.append(np.sqrt(gamma_rad) * self.sigma_minus_1.dag())
        c_ops.append(np.sqrt(gamma_rad) * self.sigma_0.dag())
        c_ops.append(np.sqrt(gamma_rad) * self.sigma_plus_1.dag())

        # ISC: |e⟩ → |sA⟩ (spin-dependent rates)
        gamma_ISC_0 = self.optical_params.gamma_ISC_0 / 1000.0  # GHz
        gamma_ISC_pm1 = self.optical_params.gamma_ISC_pm1 / 1000.0  # GHz

        c_ops.append(np.sqrt(gamma_ISC_pm1) * self.ISC_minus)
        c_ops.append(np.sqrt(gamma_ISC_0) * self.ISC_0)
        c_ops.append(np.sqrt(gamma_ISC_pm1) * self.ISC_plus)

        # Singlet cascade: |sA⟩ → |sE⟩
        gamma_sA = self.optical_params.gamma_singlet_A / 1000.0  # GHz
        c_ops.append(np.sqrt(gamma_sA) * self.singlet_decay_AE)

        # Singlet to ground: |sE⟩ → |g⟩ (with branching)
        gamma_sE = self.optical_params.gamma_singlet_E / 1000.0  # GHz
        c_ops.append(np.sqrt(gamma_sE) * self.singlet_to_ground)

        # Pure dephasing in excited state
        gamma_deph_ex = self.optical_params.gamma_dephasing_excited / 1000.0  # GHz
        for i in [self.e_minus, self.e_zero, self.e_plus]:
            c_ops.append(np.sqrt(gamma_deph_ex) * self.proj[self.level_names[i]])

        return c_ops

    def evolve_master_equation(self, tlist: np.ndarray,
                              B_field: np.ndarray,
                              laser_on: bool = True,
                              mw_params: Optional[Dict] = None,
                              e_ops: Optional[List] = None) -> qt.solver.Result:
        """
        Evolve state under master equation (Lindblad form).

        Parameters
        ----------
        tlist : array
            Time points in microseconds
        B_field : array
            Magnetic field [Bx, By, Bz] in Gauss
        laser_on : bool
            Whether laser excitation is on
        mw_params : dict, optional
            MW parameters: {'omega': GHz, 'amplitude': MHz, 'phase': rad}
        e_ops : list, optional
            Expectation value operators

        Returns
        -------
        result : Result
            QuTiP master equation result
        """
        # Build Hamiltonian
        if mw_params:
            H = self.hamiltonian_full(B_field, **mw_params)
        else:
            H = self.hamiltonian_full(B_field)

        # Convert to proper units for QuTiP (2π * GHz)
        H = H * 2 * np.pi

        # Build collapse operators
        c_ops = self.collapse_operators(laser_on=laser_on)

        # Convert time to ns (since H is in GHz)
        tlist_ns = tlist * 1000.0

        # Solve master equation
        result = qt.mesolve(H, self.rho, tlist_ns, c_ops, e_ops=e_ops)

        # Update current state
        self.rho = result.states[-1]

        return result

    def optical_pumping_cycle(self, duration: float, B_field: np.ndarray,
                             n_points: int = 100) -> qt.solver.Result:
        """
        Simulate optical pumping cycle.

        Optical pumping preferentially depletes ms=±1 via spin-selective ISC,
        initializing the NV into ms=0.

        Parameters
        ----------
        duration : float
            Pumping duration in microseconds
        B_field : array
            Magnetic field in Gauss
        n_points : int
            Number of time points

        Returns
        -------
        result : Result
            Evolution result
        """
        tlist = np.linspace(0, duration, n_points)

        # Expectation operators: populations of each level
        e_ops = [self.proj[name] for name in self.level_names]

        result = self.evolve_master_equation(tlist, B_field, laser_on=True, e_ops=e_ops)

        return result

    def measure_PL(self, integration_time: float = 1.0,
                   B_field: np.ndarray = None) -> float:
        """
        Measure photoluminescence (fluorescence) from current state.

        PL is proportional to population in excited state after optical pumping
        reaches steady state, or equivalently to radiative decay rate.

        Parameters
        ----------
        integration_time : float
            Measurement integration time in microseconds
        B_field : array, optional
            Magnetic field (if not specified, uses zero field)

        Returns
        -------
        PL_signal : float
            Fluorescence signal (arbitrary units)
        """
        if B_field is None:
            B_field = np.array([0, 0, 0])

        # Run optical pumping to steady state
        tlist = np.linspace(0, integration_time, 50)

        # Track excited state populations
        e_ops = [
            self.proj['e,-1'],
            self.proj['e,0'],
            self.proj['e,+1']
        ]

        result = self.evolve_master_equation(tlist, B_field, laser_on=True, e_ops=e_ops)

        # PL is proportional to excited state population × radiative rate
        # Integrate over time
        excited_pop = result.expect[0] + result.expect[1] + result.expect[2]
        PL_signal = np.trapz(excited_pop, tlist) * self.optical_params.gamma_radiative

        return PL_signal

    def reset_to_ground(self, ms: int = 0):
        """Reset to ground state with specified ms"""
        if ms == 0:
            idx = self.g_zero
        elif ms == -1:
            idx = self.g_minus
        elif ms == 1:
            idx = self.g_plus
        else:
            raise ValueError("ms must be -1, 0, or 1")

        self.rho = qt.basis(self.n_levels, idx) * qt.basis(self.n_levels, idx).dag()

    def get_ground_state_populations(self) -> Tuple[float, float, float]:
        """Get populations in ground state ms = -1, 0, +1"""
        P_minus = qt.expect(self.proj['g,-1'], self.rho)
        P_0 = qt.expect(self.proj['g,0'], self.rho)
        P_plus = qt.expect(self.proj['g,+1'], self.rho)

        return (P_minus, P_0, P_plus)

    def get_all_populations(self) -> Dict[str, float]:
        """Get populations in all levels"""
        pops = {}
        for name in self.level_names:
            pops[name] = qt.expect(self.proj[name], self.rho)
        return pops


def simulate_ODMR_with_PL_contrast(nv_params: NVParameters,
                                   optical_params: OpticalParameters,
                                   B_field: np.ndarray,
                                   freq_range: Tuple[float, float],
                                   n_points: int = 200,
                                   mw_duration: float = 5.0,
                                   mw_power: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate realistic ODMR with PL contrast.

    Protocol:
    1. Optical pumping → ms=0
    2. MW driving at frequency f
    3. Optical readout → measure PL
    4. PL dip when MW is resonant with |0⟩ ↔ |±1⟩

    Parameters
    ----------
    nv_params : NVParameters
        NV-center parameters
    optical_params : OpticalParameters
        Optical parameters
    B_field : array
        Magnetic field in Gauss
    freq_range : tuple
        (f_min, f_max) frequency range in GHz
    n_points : int
        Number of frequency points
    mw_duration : float
        MW pulse duration in µs
    mw_power : float
        MW Rabi frequency in MHz

    Returns
    -------
    frequencies : array
        MW frequencies in GHz
    PL_signal : array
        Normalized PL signal (1 = no MW, <1 = MW on resonance)
    """
    nv = ExtendedNVCenter(nv_params, optical_params)

    frequencies = np.linspace(freq_range[0], freq_range[1], n_points)
    PL_signals = np.zeros(n_points)

    # Reference: PL without MW
    nv.reset_to_ground(ms=0)
    PL_ref = nv.measure_PL(integration_time=1.0, B_field=B_field)

    for i, freq in enumerate(frequencies):
        # Optical pumping
        nv.reset_to_ground(ms=0)
        nv.optical_pumping_cycle(duration=1.0, B_field=B_field, n_points=20)

        # MW driving
        tlist_mw = np.linspace(0, mw_duration, 50)
        mw_params = {'omega': freq, 'amplitude': mw_power, 'phase': 0}
        nv.evolve_master_equation(tlist_mw, B_field, laser_on=False, mw_params=mw_params)

        # PL readout
        PL = nv.measure_PL(integration_time=1.0, B_field=B_field)
        PL_signals[i] = PL

    # Normalize
    PL_signals = PL_signals / PL_ref

    return frequencies, PL_signals


if __name__ == "__main__":
    print("Extended NV-Center Level Structure Module")
    print("=" * 50)
    print("\nLevel structure:")
    print("  Ground ³A₂: |g,-1⟩, |g,0⟩, |g,+1⟩")
    print("  Excited ³E: |e,-1⟩, |e,0⟩, |e,+1⟩")
    print("  Singlet ¹A₁: |sA⟩")
    print("  Singlet ¹E: |sE⟩")
    print("\nKey features:")
    print("  • Quantum master equation (Lindblad form)")
    print("  • Spin-dependent ISC")
    print("  • Optical pumping to ms=0")
    print("  • Realistic PL contrast")
    print("  • ODMR simulation with full dynamics")
