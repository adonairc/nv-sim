"""
NV-Center in Diamond Simulator
==============================

This module provides a comprehensive simulation of nitrogen-vacancy (NV) centers
in diamond using the QuTiP library. It includes realistic physical parameters
and B-field control for quantum sensing and computing applications.

Physical Model:
- Ground state electronic spin S=1 (triplet)
- Zero-field splitting D ≈ 2.87 GHz
- Magnetic field sensitivity via Zeeman effect
- Microwave control for spin manipulation

Author: Claude AI
Date: 2025-11-05
"""

import numpy as np
import qutip as qt
from typing import List, Tuple, Optional, Callable
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class NVParameters:
    """Physical parameters for NV-center simulation"""

    # Zero-field splitting (GHz) - Ground state spin-spin interaction
    D: float = 2.870  # GHz

    # Strain-induced splitting (GHz) - typically much smaller than D
    E: float = 0.0  # GHz, usually < 10 MHz

    # Gyromagnetic ratio for NV center (MHz/G)
    gamma: float = 2.8025  # MHz/Gauss

    # Rabi frequency (MHz) - MW driving strength
    omega_rabi: float = 10.0  # MHz

    # Hyperfine coupling to 14N nuclear spin (MHz)
    A_parallel: float = -2.14  # MHz (parallel to NV axis)
    A_perp: float = -2.70  # MHz (perpendicular to NV axis)

    # Nuclear quadrupole coupling for 14N (MHz)
    P: float = -4.95  # MHz

    # T1 and T2 relaxation times (microseconds)
    T1: float = 1000.0  # µs
    T2: float = 100.0  # µs
    T2_rabi: float = 2.0  # µs (during Rabi driving)


class NVCenter:
    """
    Simulate NV-center in diamond with B-field control and pulse sequences.

    The NV-center is modeled as a spin-1 system with zero-field splitting.
    Supports static and dynamic magnetic fields, MW pulses, and nuclear spin.
    """

    def __init__(self, params: Optional[NVParameters] = None,
                 include_nuclear: bool = False):
        """
        Initialize NV-center simulator.

        Parameters
        ----------
        params : NVParameters, optional
            Physical parameters. Uses defaults if None.
        include_nuclear : bool
            Whether to include 14N nuclear spin (I=1) in simulation
        """
        self.params = params if params is not None else NVParameters()
        self.include_nuclear = include_nuclear

        # Build basis operators
        self._build_operators()

        # Store current state
        self.state = None
        self.reset_to_ground()

    def _build_operators(self):
        """Construct spin operators for the quantum system"""

        # Electronic spin S=1 operators
        self.Sx = qt.jmat(1, 'x')
        self.Sy = qt.jmat(1, 'y')
        self.Sz = qt.jmat(1, 'z')
        self.S_plus = qt.jmat(1, '+')
        self.S_minus = qt.jmat(1, '-')

        # Identity operator for electron spin
        self.I_electron = qt.qeye(3)

        if self.include_nuclear:
            # Nuclear spin I=1 operators for 14N
            self.Ix = qt.jmat(1, 'x')
            self.Iy = qt.jmat(1, 'y')
            self.Iz = qt.jmat(1, 'z')
            self.I_nuclear = qt.qeye(3)

            # Tensor product structure: electron ⊗ nuclear
            self.Sx = qt.tensor(self.Sx, self.I_nuclear)
            self.Sy = qt.tensor(self.Sy, self.I_nuclear)
            self.Sz = qt.tensor(self.Sz, self.I_nuclear)
            self.S_plus = qt.tensor(self.S_plus, self.I_nuclear)
            self.S_minus = qt.tensor(self.S_minus, self.I_nuclear)

            self.Ix = qt.tensor(self.I_electron, self.Ix)
            self.Iy = qt.tensor(self.I_electron, self.Iy)
            self.Iz = qt.tensor(self.I_electron, self.Iz)

            self.dim = 9  # 3x3 for S=1 ⊗ I=1
        else:
            self.dim = 3  # Just S=1

    def hamiltonian_static(self, B_field: np.ndarray) -> qt.Qobj:
        """
        Construct static Hamiltonian for NV-center.

        H = D*Sz^2 + E*(Sx^2 - Sy^2) + γ*B·S + hyperfine terms

        Parameters
        ----------
        B_field : array-like [Bx, By, Bz]
            Magnetic field vector in Gauss

        Returns
        -------
        H : Qobj
            Static Hamiltonian in GHz
        """
        D = self.params.D  # GHz
        E = self.params.E  # GHz
        gamma = self.params.gamma / 1000.0  # Convert MHz/G to GHz/G

        # Zero-field splitting term
        H_zfs = D * self.Sz * self.Sz
        if E != 0:
            H_zfs += E * (self.Sx * self.Sx - self.Sy * self.Sy)

        # Zeeman term (magnetic field)
        Bx, By, Bz = B_field
        H_zeeman = gamma * (Bx * self.Sx + By * self.Sy + Bz * self.Sz)

        H = H_zfs + H_zeeman

        # Hyperfine coupling to 14N nuclear spin
        if self.include_nuclear:
            A_par = self.params.A_parallel / 1000.0  # Convert to GHz
            A_perp = self.params.A_perp / 1000.0  # Convert to GHz
            P = self.params.P / 1000.0  # Convert to GHz

            # Hyperfine tensor (simplified, assuming NV axis along z)
            H_hf = A_par * self.Sz * self.Iz
            H_hf += A_perp * (self.Sx * self.Ix + self.Sy * self.Iy)

            # Nuclear quadrupole term
            H_quad = P * self.Iz * self.Iz

            H = H + H_hf + H_quad

        return H

    def hamiltonian_mw(self, omega_mw: float, phase: float = 0.0,
                       amplitude: float = None) -> qt.Qobj:
        """
        Microwave driving Hamiltonian in rotating frame.

        H_MW = Ω/2 * (cos(φ)*Sx + sin(φ)*Sy)

        Parameters
        ----------
        omega_mw : float
            MW frequency in GHz (should be near transition frequency)
        phase : float
            MW phase in radians
        amplitude : float, optional
            Rabi frequency in MHz. Uses default if None.

        Returns
        -------
        H_mw : Qobj
            MW driving Hamiltonian in GHz
        """
        if amplitude is None:
            amplitude = self.params.omega_rabi

        omega_rabi = amplitude / 1000.0  # Convert MHz to GHz

        # Rotating wave approximation: drive transitions
        H_mw = (omega_rabi / 2.0) * (np.cos(phase) * self.Sx +
                                      np.sin(phase) * self.Sy)

        return H_mw

    def evolve(self, H: qt.Qobj, time: float,
               state_init: Optional[qt.Qobj] = None) -> qt.Qobj:
        """
        Evolve quantum state under Hamiltonian.

        Parameters
        ----------
        H : Qobj
            Hamiltonian (time-independent)
        time : float
            Evolution time in microseconds
        state_init : Qobj, optional
            Initial state. Uses current state if None.

        Returns
        -------
        state_final : Qobj
            Evolved quantum state
        """
        if state_init is None:
            state_init = self.state

        # Convert time to nanoseconds for consistency with GHz
        time_ns = time * 1000.0  # µs to ns

        # Time evolution operator: U = exp(-i H t)
        # Note: QuTiP uses ħ=1 convention
        U = (-1j * H * time_ns * 2 * np.pi).expm()

        state_final = U * state_init
        self.state = state_final

        return state_final

    def evolve_time_dependent(self, H_list: List, tlist: np.ndarray,
                             state_init: Optional[qt.Qobj] = None,
                             e_ops: Optional[List] = None) -> qt.solver.Result:
        """
        Evolve under time-dependent Hamiltonian.

        Parameters
        ----------
        H_list : list
            QuTiP time-dependent Hamiltonian format:
            [[H0, coeff0], [H1, coeff1], ...] or single H
        tlist : array
            Time points in microseconds
        state_init : Qobj, optional
            Initial state
        e_ops : list, optional
            Expectation value operators to track

        Returns
        -------
        result : Result
            QuTiP solver result object
        """
        if state_init is None:
            state_init = self.state

        # Convert time to nanoseconds
        tlist_ns = tlist * 1000.0

        # Scale time in coefficient functions for GHz Hamiltonian
        if isinstance(H_list, list) and len(H_list) > 0:
            H_scaled = []
            for item in H_list:
                if isinstance(item, list) and len(item) == 2:
                    H, coeff = item
                    if callable(coeff):
                        # Wrap coefficient function to handle time scaling
                        def scaled_coeff(t, args, original_coeff=coeff):
                            return original_coeff(t, args)
                        H_scaled.append([H * 2 * np.pi, scaled_coeff])
                    else:
                        H_scaled.append([H * 2 * np.pi, coeff])
                else:
                    H_scaled.append(item * 2 * np.pi)
        else:
            H_scaled = H_list * 2 * np.pi

        # Solve Schrödinger equation
        result = qt.sesolve(H_scaled, state_init, tlist_ns, e_ops=e_ops)

        self.state = result.states[-1]
        return result

    def apply_pulse(self, pulse_type: str, duration: float,
                    B_field: np.ndarray, **kwargs) -> qt.Qobj:
        """
        Apply a pulse to the NV-center.

        Parameters
        ----------
        pulse_type : str
            Type of pulse: 'pi', 'pi/2', 'pi_x', 'pi_y', 'custom'
        duration : float
            Pulse duration in microseconds
        B_field : array-like
            Background magnetic field [Bx, By, Bz] in Gauss
        **kwargs : dict
            Additional parameters (omega_mw, phase, amplitude, etc.)

        Returns
        -------
        state : Qobj
            State after pulse application
        """
        # Get MW parameters
        omega_mw = kwargs.get('omega_mw', self.params.D)  # Default to D
        phase = kwargs.get('phase', 0.0)
        amplitude = kwargs.get('amplitude', None)

        # Set phase and amplitude based on pulse type
        if pulse_type == 'pi_x':
            phase = 0.0
            # Calculate amplitude for π pulse: Ω*t = π
            if amplitude is None:
                amplitude = (np.pi / duration) * 1000.0  # MHz
        elif pulse_type == 'pi_y':
            phase = np.pi / 2
            if amplitude is None:
                amplitude = (np.pi / duration) * 1000.0  # MHz
        elif pulse_type == 'pi/2_x':
            phase = 0.0
            if amplitude is None:
                amplitude = (np.pi / 2 / duration) * 1000.0  # MHz
        elif pulse_type == 'pi/2_y':
            phase = np.pi / 2
            if amplitude is None:
                amplitude = (np.pi / 2 / duration) * 1000.0  # MHz

        # Construct Hamiltonian in rotating frame
        # Static part (in rotating frame, Sz term modified by detuning)
        H_static = self.hamiltonian_static(B_field)
        detuning = kwargs.get('detuning', 0.0)  # GHz
        H_rotating = H_static - omega_mw * self.Sz + detuning * self.Sz

        # MW driving term
        H_mw = self.hamiltonian_mw(omega_mw, phase, amplitude)

        # Total Hamiltonian
        H_total = H_rotating + H_mw

        # Evolve
        return self.evolve(H_total, duration)

    def reset_to_ground(self):
        """Reset state to |0⟩ (ms=0) ground state"""
        if self.include_nuclear:
            # |0⟩_electron ⊗ |0⟩_nuclear
            self.state = qt.tensor(qt.basis(3, 1), qt.basis(3, 1))
        else:
            self.state = qt.basis(3, 1)  # |0⟩ (ms=0)

    def measure_population(self, state: Optional[qt.Qobj] = None) -> Tuple:
        """
        Measure populations in ms = -1, 0, +1 basis.

        Returns
        -------
        populations : tuple
            (P_minus1, P_0, P_plus1)
        """
        if state is None:
            state = self.state

        if self.include_nuclear:
            # Trace out nuclear spin
            rho = state * state.dag()
            rho_reduced = rho.ptrace(0)  # Trace out second subsystem
            populations = np.real(rho_reduced.diag())
        else:
            populations = np.abs(state.full().flatten())**2

        return tuple(populations)

    def measure_expectation(self, operator: qt.Qobj,
                           state: Optional[qt.Qobj] = None) -> float:
        """Calculate expectation value of an operator"""
        if state is None:
            state = self.state

        return qt.expect(operator, state)

    def apply_fem_mw_pulse(self, fem_data, B_static: np.ndarray,
                          nv_position: Optional[np.ndarray] = None,
                          e_ops: Optional[List] = None) -> qt.solver.Result:
        """
        Apply microwave pulse from FEM B-field data.

        This method uses realistic time-dependent MW fields from FEM simulations
        of antennas, providing accurate modeling of experimental conditions.

        Parameters
        ----------
        fem_data : FEMBFieldData
            FEM magnetic field data (from fem_bfield module)
        B_static : array-like
            Static background field [Bx, By, Bz] in Gauss
        nv_position : array-like, optional
            NV-center position [x, y, z] in cm for spatial field data
        e_ops : list, optional
            Expectation value operators to track

        Returns
        -------
        result : Result
            QuTiP solver result with final state

        Examples
        --------
        >>> from fem_bfield import MicrowaveAntennaSimulator
        >>> mw_pulse = MicrowaveAntennaSimulator.generate_pulsed_mw(
        ...     duration=1.0, frequency=2.87, amplitude=0.5)
        >>> result = nv.apply_fem_mw_pulse(mw_pulse, B_static=[0, 0, 10])
        """
        from fem_bfield import FEMBFieldData

        if not isinstance(fem_data, FEMBFieldData):
            raise TypeError("fem_data must be FEMBFieldData instance")

        # Extract field at NV position if spatial
        if fem_data.is_spatial:
            if nv_position is None:
                raise ValueError("nv_position required for spatial FEM data")
            B_mw_local = fem_data.get_field_at_position(nv_position)
        else:
            B_mw_local = fem_data.B_field

        # Build time-dependent Hamiltonian
        # H = H_static + H_MW(t)

        # Static part (zero-field splitting + static B-field)
        H_static = self.hamiltonian_static(B_static)

        # Move to rotating frame at MW frequency if provided
        if fem_data.frequency is not None:
            omega_mw = fem_data.frequency  # GHz
            H_static = H_static - omega_mw * self.Sz

        # Time-dependent MW Zeeman term: γ * B_MW(t) · S
        gamma = self.params.gamma / 1000.0  # Convert MHz/G to GHz/G

        # Create interpolators for MW field components
        from scipy.interpolate import interp1d

        interp_Bx = interp1d(fem_data.time, B_mw_local[:, 0], kind='linear',
                            bounds_error=False, fill_value=0.0)
        interp_By = interp1d(fem_data.time, B_mw_local[:, 1], kind='linear',
                            bounds_error=False, fill_value=0.0)
        interp_Bz = interp1d(fem_data.time, B_mw_local[:, 2], kind='linear',
                            bounds_error=False, fill_value=0.0)

        # Coefficient functions for time-dependent Hamiltonian
        def coeff_x(t, args):
            return gamma * float(interp_Bx(t))

        def coeff_y(t, args):
            return gamma * float(interp_By(t))

        def coeff_z(t, args):
            return gamma * float(interp_Bz(t))

        # Build Hamiltonian list
        H_list = [
            H_static,
            [self.Sx, coeff_x],
            [self.Sy, coeff_y],
            [self.Sz, coeff_z]
        ]

        # Evolve
        result = self.evolve_time_dependent(H_list, fem_data.time,
                                           state_init=self.state,
                                           e_ops=e_ops)

        return result

    def rabi_with_fem_antenna(self, antenna_type: str, distance: float,
                             duration_max: float, n_points: int = 100,
                             power: float = 1.0, frequency: float = None,
                             **antenna_kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate Rabi oscillations with realistic antenna-generated MW field.

        Parameters
        ----------
        antenna_type : str
            'wire' or 'cpw'
        distance : float
            Distance from antenna to NV (cm)
        duration_max : float
            Maximum pulse duration (µs)
        n_points : int
            Number of duration points
        power : float
            Microwave power (W)
        frequency : float, optional
            MW frequency (GHz). Uses D if None.
        **antenna_kwargs : dict
            Additional antenna parameters

        Returns
        -------
        durations : array
            Pulse durations
        populations : array
            Population in ms=0 state
        """
        from fem_bfield import MicrowaveAntennaSimulator

        if frequency is None:
            frequency = self.params.D

        durations = np.linspace(0, duration_max, n_points)
        populations = np.zeros(n_points)

        # Calculate MW field amplitude at NV position
        nv_pos = np.array([0, 0, distance])  # NV above antenna

        if antenna_type == 'wire':
            B_amp = MicrowaveAntennaSimulator.wire_antenna(
                nv_pos, power=power, frequency=frequency, **antenna_kwargs)
        elif antenna_type == 'cpw':
            B_amp, _ = MicrowaveAntennaSimulator.coplanar_waveguide(
                nv_pos, power=power, **antenna_kwargs)
        else:
            raise ValueError(f"Unknown antenna type: {antenna_type}")

        # Run Rabi with this amplitude
        for i, duration in enumerate(durations):
            if duration == 0:
                self.reset_to_ground()
                _, P0, _ = self.measure_population()
                populations[i] = P0
                continue

            # Generate MW pulse
            mw_pulse = MicrowaveAntennaSimulator.generate_pulsed_mw(
                duration=duration,
                frequency=frequency,
                amplitude=B_amp,
                envelope='rectangular',
                n_points=max(50, int(duration * 100))
            )

            # Apply pulse
            self.reset_to_ground()
            B_static = np.array([0, 0, 0])  # Can add static field if needed
            result = self.apply_fem_mw_pulse(mw_pulse, B_static)

            # Measure
            _, P0, _ = self.measure_population()
            populations[i] = P0

        return durations, populations


class PulseSequence:
    """
    Builder for common pulse sequences on NV-center.
    """

    def __init__(self, nv: NVCenter, B_field: np.ndarray):
        """
        Parameters
        ----------
        nv : NVCenter
            NV-center simulator instance
        B_field : array-like
            Background magnetic field [Bx, By, Bz] in Gauss
        """
        self.nv = nv
        self.B_field = np.array(B_field)

    def rabi_oscillation(self, duration_max: float, n_points: int = 100,
                        omega_rabi: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate Rabi oscillation experiment.

        Sequence: Initialize → MW pulse (variable duration) → Measure

        Parameters
        ----------
        duration_max : float
            Maximum pulse duration in microseconds
        n_points : int
            Number of time points
        omega_rabi : float, optional
            Rabi frequency in MHz

        Returns
        -------
        times : array
            Pulse durations
        populations : array
            Population in ms=0 state vs time
        """
        times = np.linspace(0, duration_max, n_points)
        populations = np.zeros(n_points)

        for i, t in enumerate(times):
            self.nv.reset_to_ground()
            if t > 0:
                self.nv.apply_pulse('custom', t, self.B_field,
                                   amplitude=omega_rabi)
            _, P0, _ = self.nv.measure_population()
            populations[i] = P0

        return times, populations

    def ramsey_sequence(self, tau_max: float, n_points: int = 100,
                       detuning: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate Ramsey interferometry.

        Sequence: π/2 → free evolution (τ) → π/2 → Measure

        Parameters
        ----------
        tau_max : float
            Maximum free evolution time in microseconds
        n_points : int
            Number of points
        detuning : float
            MW detuning from resonance in MHz

        Returns
        -------
        taus : array
            Free evolution times
        populations : array
            Population in ms=0 state
        """
        taus = np.linspace(0, tau_max, n_points)
        populations = np.zeros(n_points)

        pulse_duration = 0.1  # µs

        for i, tau in enumerate(taus):
            self.nv.reset_to_ground()

            # First π/2 pulse
            self.nv.apply_pulse('pi/2_x', pulse_duration, self.B_field)

            # Free evolution
            if tau > 0:
                H_static = self.nv.hamiltonian_static(self.B_field)
                H_detuning = H_static + (detuning / 1000.0) * self.nv.Sz
                self.nv.evolve(H_detuning, tau)

            # Second π/2 pulse
            self.nv.apply_pulse('pi/2_x', pulse_duration, self.B_field)

            _, P0, _ = self.nv.measure_population()
            populations[i] = P0

        return taus, populations

    def hahn_echo(self, tau_max: float, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate Hahn echo (spin echo) sequence.

        Sequence: π/2 → τ → π → τ → π/2 → Measure

        Parameters
        ----------
        tau_max : float
            Maximum half-echo time in microseconds
        n_points : int
            Number of points

        Returns
        -------
        taus : array
            Half-echo times
        populations : array
            Population in ms=0 state
        """
        taus = np.linspace(0, tau_max, n_points)
        populations = np.zeros(n_points)

        pulse_duration = 0.1  # µs

        for i, tau in enumerate(taus):
            self.nv.reset_to_ground()

            # π/2 pulse
            self.nv.apply_pulse('pi/2_x', pulse_duration, self.B_field)

            # First τ
            if tau > 0:
                H = self.nv.hamiltonian_static(self.B_field)
                self.nv.evolve(H, tau)

            # π pulse (refocusing)
            self.nv.apply_pulse('pi_y', pulse_duration * 2, self.B_field)

            # Second τ
            if tau > 0:
                self.nv.evolve(H, tau)

            # Final π/2 pulse
            self.nv.apply_pulse('pi/2_x', pulse_duration, self.B_field)

            _, P0, _ = self.nv.measure_population()
            populations[i] = P0

        return taus, populations

    def odmr_spectrum(self, freq_min: float, freq_max: float,
                     n_points: int = 200, pulse_duration: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate optically detected magnetic resonance (ODMR) spectrum.

        Scan MW frequency and measure population after continuous driving.

        Parameters
        ----------
        freq_min, freq_max : float
            Frequency range in GHz
        n_points : int
            Number of frequency points
        pulse_duration : float
            Duration of MW pulse in microseconds

        Returns
        -------
        frequencies : array
            MW frequencies in GHz
        contrast : array
            ODMR contrast (1 - P0)
        """
        frequencies = np.linspace(freq_min, freq_max, n_points)
        contrast = np.zeros(n_points)

        for i, freq in enumerate(frequencies):
            self.nv.reset_to_ground()

            # Apply MW at this frequency
            self.nv.apply_pulse('custom', pulse_duration, self.B_field,
                              omega_mw=freq)

            _, P0, _ = self.nv.measure_population()
            contrast[i] = 1.0 - P0  # ODMR measures decrease in fluorescence

        return frequencies, contrast


def create_realistic_b_field(t: float, B_static: np.ndarray,
                             B_ac_amplitude: float = 0.0,
                             B_ac_freq: float = 0.0,
                             noise_level: float = 0.0) -> np.ndarray:
    """
    Generate realistic time-dependent magnetic field.

    Parameters
    ----------
    t : float
        Time in microseconds
    B_static : array-like
        Static field component [Bx, By, Bz] in Gauss
    B_ac_amplitude : float
        AC field amplitude in Gauss
    B_ac_freq : float
        AC field frequency in MHz
    noise_level : float
        Gaussian noise level in Gauss

    Returns
    -------
    B_field : array
        Total magnetic field [Bx, By, Bz]
    """
    B_static = np.array(B_static)

    # AC component (along z for simplicity)
    B_ac = np.array([0, 0, B_ac_amplitude * np.sin(2 * np.pi * B_ac_freq * t)])

    # Noise
    B_noise = np.random.normal(0, noise_level, 3) if noise_level > 0 else 0

    return B_static + B_ac + B_noise


if __name__ == "__main__":
    print("NV-Center Diamond Simulator")
    print("=" * 50)
    print("\nThis module provides tools for simulating NV-centers.")
    print("Import this module to use the NVCenter and PulseSequence classes.")
    print("\nExample usage:")
    print("  from nv_center import NVCenter, PulseSequence")
    print("  nv = NVCenter()")
    print("  ps = PulseSequence(nv, B_field=[0, 0, 10])")
    print("  times, pops = ps.rabi_oscillation(1.0)")
