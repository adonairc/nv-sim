"""
Visualization Utilities for NV-Center Simulation
================================================

Helper functions for plotting and analyzing NV-center simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d import Axes3D
import qutip as qt


class Arrow3D(FancyArrowPatch):
    """3D arrow for Bloch sphere visualization"""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def plot_bloch_sphere(state, title="State on Bloch Sphere", ax=None):
    """
    Plot quantum state on Bloch sphere.

    Parameters
    ----------
    state : Qobj
        Quantum state (must be S=1 system)
    title : str
        Plot title
    ax : Axes3D, optional
        Matplotlib 3D axis

    Returns
    -------
    ax : Axes3D
        The 3D axis
    """
    # Calculate expectation values
    if state.type == 'ket':
        rho = state * state.dag()
    else:
        rho = state

    # For S=1 system, need to use S=1 operators
    Sx = qt.jmat(1, 'x')
    Sy = qt.jmat(1, 'y')
    Sz = qt.jmat(1, 'z')

    x = qt.expect(Sx, rho)
    y = qt.expect(Sy, rho)
    z = qt.expect(Sz, rho)

    # Create plot
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

    # Draw sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1,
                   color='lightblue', edgecolor='none')

    # Draw axes
    axis_length = 1.3
    ax.plot([0, axis_length], [0, 0], [0, 0], 'k-', linewidth=1, alpha=0.5)
    ax.plot([0, 0], [0, axis_length], [0, 0], 'k-', linewidth=1, alpha=0.5)
    ax.plot([0, 0], [0, 0], [0, axis_length], 'k-', linewidth=1, alpha=0.5)

    ax.text(axis_length, 0, 0, 'X', fontsize=12)
    ax.text(0, axis_length, 0, 'Y', fontsize=12)
    ax.text(0, 0, axis_length, 'Z (ms=0)', fontsize=12)

    # Draw state vector
    arrow = Arrow3D([0, x], [0, y], [0, z],
                   mutation_scale=20, lw=3, arrowstyle='-|>',
                   color='red')
    ax.add_artist(arrow)

    # Add point at state location
    ax.scatter([x], [y], [z], color='red', s=100, marker='o')

    # Labels
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_xlabel('Sx')
    ax.set_ylabel('Sy')
    ax.set_zlabel('Sz')
    ax.set_title(title)

    return ax


def plot_energy_levels(B_field, params, ax=None):
    """
    Plot NV-center energy level diagram vs magnetic field.

    Parameters
    ----------
    B_field : float or array
        Magnetic field strength(s) in Gauss (along NV axis)
    params : NVParameters
        NV-center parameters
    ax : Axes, optional
        Matplotlib axis

    Returns
    -------
    ax : Axes
        The axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Handle single value or array
    if np.isscalar(B_field):
        B_field = np.array([B_field])
    else:
        B_field = np.asarray(B_field)

    # Calculate energy levels
    from nv_center import NVCenter
    nv = NVCenter(params=params, include_nuclear=False)

    energies = np.zeros((len(B_field), 3))  # 3 levels for S=1

    for i, B_z in enumerate(B_field):
        B_vec = np.array([0, 0, B_z])
        H = nv.hamiltonian_static(B_vec)

        # Get eigenvalues
        eigvals = H.eigenenergies()
        energies[i] = np.sort(eigvals)

    # Plot energy levels
    colors = ['blue', 'green', 'red']
    labels = ['ms = -1', 'ms = 0', 'ms = +1']

    for j in range(3):
        ax.plot(B_field, energies[:, j], color=colors[j],
               linewidth=2, label=labels[j])

    ax.set_xlabel('Magnetic Field B_z (Gauss)')
    ax.set_ylabel('Energy (GHz)')
    ax.set_title('NV-Center Energy Levels')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_pulse_sequence_diagram(sequence_name, ax=None):
    """
    Draw schematic pulse sequence diagrams.

    Parameters
    ----------
    sequence_name : str
        Name of sequence: 'rabi', 'ramsey', 'hahn', 'xy8', 'cpmg'
    ax : Axes, optional
        Matplotlib axis

    Returns
    -------
    ax : Axes
        The axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    ax.set_ylim(-0.5, 1.5)
    ax.set_xlim(-0.5, None)
    ax.axis('off')

    pulse_width = 0.3
    spacing = 1.0

    if sequence_name.lower() == 'rabi':
        # Just a single pulse with variable duration
        ax.add_patch(plt.Rectangle((0, 0), 2.0, 1.0, facecolor='blue',
                                   edgecolor='black', linewidth=2, alpha=0.7))
        ax.text(1.0, 0.5, 'MW Pulse\n(variable)', ha='center', va='center',
               fontsize=12, fontweight='bold')
        ax.text(1.0, -0.3, 'τ', ha='center', fontsize=14, color='red',
               fontweight='bold')
        ax.set_xlim(-0.5, 3)

    elif sequence_name.lower() == 'ramsey':
        # π/2 - τ - π/2
        positions = [0, 0 + pulse_width + spacing]
        labels = ['π/2', 'π/2']

        for pos, label in zip(positions, labels):
            ax.add_patch(plt.Rectangle((pos, 0), pulse_width, 0.8,
                                      facecolor='blue', edgecolor='black',
                                      linewidth=2, alpha=0.7))
            ax.text(pos + pulse_width/2, 0.4, label, ha='center', va='center',
                   fontsize=12, fontweight='bold', color='white')

        # Free evolution
        tau_start = pulse_width + 0.1
        tau_end = spacing - 0.1
        ax.annotate('', xy=(tau_end, -0.2), xytext=(tau_start, -0.2),
                   arrowprops=dict(arrowstyle='<->', color='green', lw=2))
        ax.text((tau_start + tau_end)/2, -0.35, 'τ (free)', ha='center',
               fontsize=12, color='green', fontweight='bold')

        ax.set_xlim(-0.5, positions[-1] + pulse_width + 0.5)

    elif sequence_name.lower() == 'hahn':
        # π/2 - τ - π - τ - π/2
        tau = 1.5
        positions = [0, pulse_width + tau, 2*pulse_width + 2*tau]
        widths = [pulse_width, 2*pulse_width, pulse_width]
        labels = ['π/2', 'π', 'π/2']

        for pos, width, label in zip(positions, widths, labels):
            ax.add_patch(plt.Rectangle((pos, 0), width, 0.8,
                                      facecolor='blue', edgecolor='black',
                                      linewidth=2, alpha=0.7))
            ax.text(pos + width/2, 0.4, label, ha='center', va='center',
                   fontsize=12, fontweight='bold', color='white')

        # Free evolution periods
        for i, (start, end) in enumerate([(pulse_width + 0.1, pulse_width + tau - 0.1),
                                          (2*pulse_width + tau + 0.1,
                                           2*pulse_width + 2*tau - 0.1)]):
            ax.annotate('', xy=(end, -0.2), xytext=(start, -0.2),
                       arrowprops=dict(arrowstyle='<->', color='green', lw=2))
            ax.text((start + end)/2, -0.35, 'τ', ha='center',
                   fontsize=12, color='green', fontweight='bold')

        ax.set_xlim(-0.5, positions[-1] + widths[-1] + 0.5)

    elif sequence_name.lower() == 'xy8':
        # π/2 - (X-Y-X-Y-Y-X-Y-X) - π/2
        phases = ['X', 'Y', 'X', 'Y', 'Y', 'X', 'Y', 'X']
        n_pulses = len(phases)
        tau_small = 0.4

        # Initial π/2
        ax.add_patch(plt.Rectangle((0, 0), pulse_width, 0.8,
                                  facecolor='blue', edgecolor='black',
                                  linewidth=2, alpha=0.7))
        ax.text(pulse_width/2, 0.4, 'π/2', ha='center', va='center',
               fontsize=10, fontweight='bold', color='white')

        # XY8 pulses
        colors_xy = {'X': 'red', 'Y': 'green'}
        current_pos = pulse_width + tau_small

        for phase in phases:
            ax.add_patch(plt.Rectangle((current_pos, 0), pulse_width, 0.8,
                                      facecolor=colors_xy[phase],
                                      edgecolor='black', linewidth=2, alpha=0.7))
            ax.text(current_pos + pulse_width/2, 0.4, phase, ha='center',
                   va='center', fontsize=10, fontweight='bold', color='white')
            current_pos += pulse_width + tau_small

        # Final π/2
        ax.add_patch(plt.Rectangle((current_pos, 0), pulse_width, 0.8,
                                  facecolor='blue', edgecolor='black',
                                  linewidth=2, alpha=0.7))
        ax.text(current_pos + pulse_width/2, 0.4, 'π/2', ha='center',
               va='center', fontsize=10, fontweight='bold', color='white')

        ax.set_xlim(-0.5, current_pos + pulse_width + 0.5)

    ax.set_title(f'{sequence_name.upper()} Pulse Sequence', fontsize=14,
                fontweight='bold')

    return ax


def plot_density_matrix(rho, title="Density Matrix", ax=None):
    """
    Visualize density matrix.

    Parameters
    ----------
    rho : Qobj
        Density matrix
    title : str
        Plot title
    ax : Axes, optional
        Matplotlib axis

    Returns
    -------
    ax : Axes
        The axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Get matrix elements
    rho_mat = rho.full()

    # Plot absolute values
    im = ax.imshow(np.abs(rho_mat), cmap='viridis', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='|ρ_ij|')

    # Add text annotations
    for i in range(rho_mat.shape[0]):
        for j in range(rho_mat.shape[1]):
            val = rho_mat[i, j]
            text = f'{np.abs(val):.2f}\n∠{np.angle(val)*180/np.pi:.0f}°'
            ax.text(j, i, text, ha='center', va='center',
                   fontsize=8, color='white' if np.abs(val) > 0.3 else 'black')

    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_title(title)
    ax.set_xticks(range(rho_mat.shape[1]))
    ax.set_yticks(range(rho_mat.shape[0]))

    if rho_mat.shape[0] == 3:
        labels = ['ms=-1', 'ms=0', 'ms=+1']
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

    return ax


def plot_trajectory(states, nv, title="State Trajectory", fig=None):
    """
    Plot time evolution of state on Bloch sphere.

    Parameters
    ----------
    states : list of Qobj
        List of quantum states at different times
    nv : NVCenter
        NV-center instance (for operators)
    title : str
        Plot title
    fig : Figure, optional
        Matplotlib figure

    Returns
    -------
    fig : Figure
        The figure
    """
    if fig is None:
        fig = plt.figure(figsize=(12, 5))

    # Extract expectation values
    Sx_vals = [qt.expect(nv.Sx, state) for state in states]
    Sy_vals = [qt.expect(nv.Sy, state) for state in states]
    Sz_vals = [qt.expect(nv.Sz, state) for state in states]

    # 3D trajectory on Bloch sphere
    ax1 = fig.add_subplot(121, projection='3d')

    # Draw sphere
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))

    ax1.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1,
                    color='lightblue', edgecolor='none')

    # Plot trajectory
    ax1.plot(Sx_vals, Sy_vals, Sz_vals, 'r-', linewidth=2, alpha=0.7)
    ax1.scatter(Sx_vals[0], Sy_vals[0], Sz_vals[0], color='green',
               s=100, marker='o', label='Start')
    ax1.scatter(Sx_vals[-1], Sy_vals[-1], Sz_vals[-1], color='red',
               s=100, marker='s', label='End')

    ax1.set_xlabel('Sx')
    ax1.set_ylabel('Sy')
    ax1.set_zlabel('Sz')
    ax1.set_title('Bloch Sphere Trajectory')
    ax1.legend()

    # Time series of components
    ax2 = fig.add_subplot(122)
    times = np.arange(len(states))

    ax2.plot(times, Sx_vals, 'r-', linewidth=2, label='⟨Sx⟩')
    ax2.plot(times, Sy_vals, 'g-', linewidth=2, label='⟨Sy⟩')
    ax2.plot(times, Sz_vals, 'b-', linewidth=2, label='⟨Sz⟩')

    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Expectation Value')
    ax2.set_title('Spin Components vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def create_summary_plot(nv, B_field, params):
    """
    Create comprehensive summary visualization of NV-center system.

    Parameters
    ----------
    nv : NVCenter
        NV-center instance
    B_field : array
        Magnetic field vector
    params : NVParameters
        NV-center parameters

    Returns
    -------
    fig : Figure
        Summary figure
    """
    fig = plt.figure(figsize=(16, 10))

    # Energy levels
    ax1 = plt.subplot(2, 3, 1)
    B_range = np.linspace(0, 50, 100)
    plot_energy_levels(B_range, params, ax=ax1)

    # Current state on Bloch sphere
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    plot_bloch_sphere(nv.state, "Current State", ax=ax2)

    # Density matrix
    ax3 = plt.subplot(2, 3, 3)
    rho = nv.state * nv.state.dag()
    plot_density_matrix(rho, "Density Matrix", ax=ax3)

    # Pulse sequence diagrams
    for i, seq_name in enumerate(['ramsey', 'hahn', 'xy8']):
        ax = plt.subplot(2, 3, 4 + i)
        plot_pulse_sequence_diagram(seq_name, ax=ax)

    fig.suptitle(f'NV-Center System Summary (B = {B_field})',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    return fig


if __name__ == "__main__":
    print("Visualization utilities for NV-center simulation")
    print("=" * 50)
    print("\nAvailable functions:")
    print("  - plot_bloch_sphere(state)")
    print("  - plot_energy_levels(B_field, params)")
    print("  - plot_pulse_sequence_diagram(sequence_name)")
    print("  - plot_density_matrix(rho)")
    print("  - plot_trajectory(states, nv)")
    print("  - create_summary_plot(nv, B_field, params)")
    print("\nImport this module to use visualization tools.")
