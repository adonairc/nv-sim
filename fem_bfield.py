"""
FEM B-Field Handler for NV-Center Simulation
=============================================

Module for loading, processing, and applying time-series magnetic field data
from FEM simulations of microwave antennas.

Supports:
- Common FEM output formats (CSV, HDF5, NPY, MAT)
- Time-domain and frequency-domain data
- Spatial field profiles
- Interpolation and resampling
- Antenna pattern synthesis

Author: Claude AI
Date: 2025-11-05
"""

import numpy as np
from typing import Tuple, Optional, Callable, Union, List
from dataclasses import dataclass
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.signal import resample, hilbert
import warnings

try:
    import h5py
    HAS_HDF5 = True
except ImportError:
    HAS_HDF5 = False
    warnings.warn("h5py not available. HDF5 support disabled.")

try:
    from scipy.io import loadmat, savemat
    HAS_MATLAB = True
except ImportError:
    HAS_MATLAB = False
    warnings.warn("scipy.io not available. MATLAB support limited.")


@dataclass
class FEMBFieldData:
    """
    Container for FEM magnetic field simulation data.

    Attributes
    ----------
    time : array
        Time points (microseconds)
    B_field : array
        Magnetic field [Bx, By, Bz] at each time point (Gauss)
        Shape: (n_times, 3) for single point or (n_times, nx, ny, nz, 3) for spatial
    position : array, optional
        Spatial coordinates [x, y, z] if spatially resolved (cm)
    frequency : float, optional
        Carrier frequency for MW field (GHz)
    metadata : dict, optional
        Additional information (antenna type, power, etc.)
    """
    time: np.ndarray
    B_field: np.ndarray
    position: Optional[np.ndarray] = None
    frequency: Optional[float] = None
    metadata: Optional[dict] = None

    @property
    def is_spatial(self) -> bool:
        """Check if field data has spatial dependence"""
        return len(self.B_field.shape) > 2

    @property
    def n_times(self) -> int:
        """Number of time points"""
        return len(self.time)

    @property
    def dt(self) -> float:
        """Time step (microseconds)"""
        return self.time[1] - self.time[0] if len(self.time) > 1 else 0.0

    def get_field_at_position(self, position: np.ndarray) -> np.ndarray:
        """
        Extract field time series at specific position.

        Parameters
        ----------
        position : array [x, y, z]
            Position in cm

        Returns
        -------
        B_field_local : array
            Field time series at position, shape (n_times, 3)
        """
        if not self.is_spatial:
            # Already point data
            return self.B_field

        # Interpolate spatial field
        if self.position is None:
            raise ValueError("Spatial coordinates not provided")

        # Build interpolator for each component and time
        x, y, z = position
        B_local = np.zeros((self.n_times, 3))

        for t_idx in range(self.n_times):
            for comp in range(3):
                # Create interpolator for this component at this time
                interp = RegularGridInterpolator(
                    self.position,
                    self.B_field[t_idx, ..., comp],
                    method='linear',
                    bounds_error=False,
                    fill_value=0.0
                )
                B_local[t_idx, comp] = interp([x, y, z])[0]

        return B_local


class FEMBFieldLoader:
    """Load FEM B-field data from various file formats"""

    @staticmethod
    def load_csv(filename: str, time_col: int = 0,
                 Bx_col: int = 1, By_col: int = 2, Bz_col: int = 3,
                 time_unit: str = 'us', field_unit: str = 'G',
                 delimiter: str = ',', skip_header: int = 0) -> FEMBFieldData:
        """
        Load B-field data from CSV file.

        Parameters
        ----------
        filename : str
            Path to CSV file
        time_col, Bx_col, By_col, Bz_col : int
            Column indices for time and field components
        time_unit : str
            'us' (microseconds), 'ns' (nanoseconds), or 's' (seconds)
        field_unit : str
            'G' (Gauss), 'T' (Tesla), 'mT' (milliTesla)
        delimiter : str
            CSV delimiter
        skip_header : int
            Number of header lines to skip

        Returns
        -------
        data : FEMBFieldData
            Loaded field data
        """
        # Load CSV
        raw_data = np.loadtxt(filename, delimiter=delimiter, skiprows=skip_header)

        # Extract time
        time = raw_data[:, time_col]

        # Convert time to microseconds
        if time_unit == 'ns':
            time = time / 1000.0
        elif time_unit == 's':
            time = time * 1e6
        elif time_unit != 'us':
            raise ValueError(f"Unknown time unit: {time_unit}")

        # Extract field components
        Bx = raw_data[:, Bx_col]
        By = raw_data[:, By_col]
        Bz = raw_data[:, Bz_col]

        # Convert field to Gauss
        if field_unit == 'T':
            Bx, By, Bz = Bx * 1e4, By * 1e4, Bz * 1e4
        elif field_unit == 'mT':
            Bx, By, Bz = Bx * 10, By * 10, Bz * 10
        elif field_unit != 'G':
            raise ValueError(f"Unknown field unit: {field_unit}")

        B_field = np.column_stack([Bx, By, Bz])

        metadata = {
            'source': filename,
            'format': 'csv',
            'time_unit_original': time_unit,
            'field_unit_original': field_unit
        }

        return FEMBFieldData(time=time, B_field=B_field, metadata=metadata)

    @staticmethod
    def load_npy(filename: str) -> FEMBFieldData:
        """
        Load B-field data from NumPy .npy or .npz file.

        Expected format:
        - .npy: array of shape (n_times, 4) with columns [time, Bx, By, Bz]
        - .npz: dict with keys 'time', 'B_field', and optionally 'position', 'frequency'

        Parameters
        ----------
        filename : str
            Path to NumPy file

        Returns
        -------
        data : FEMBFieldData
            Loaded field data
        """
        if filename.endswith('.npz'):
            data_dict = np.load(filename)
            time = data_dict['time']
            B_field = data_dict['B_field']
            position = data_dict.get('position', None)
            frequency = float(data_dict['frequency']) if 'frequency' in data_dict else None

            metadata = {
                'source': filename,
                'format': 'npz',
                'keys': list(data_dict.keys())
            }

            return FEMBFieldData(time=time, B_field=B_field,
                               position=position, frequency=frequency,
                               metadata=metadata)
        else:
            # Plain .npy file
            data = np.load(filename)
            if data.shape[1] == 4:
                time = data[:, 0]
                B_field = data[:, 1:4]
            else:
                raise ValueError(f"Unexpected data shape: {data.shape}")

            metadata = {'source': filename, 'format': 'npy'}
            return FEMBFieldData(time=time, B_field=B_field, metadata=metadata)

    @staticmethod
    def load_hdf5(filename: str, time_key: str = 'time',
                  field_key: str = 'B_field') -> FEMBFieldData:
        """
        Load B-field data from HDF5 file.

        Parameters
        ----------
        filename : str
            Path to HDF5 file
        time_key : str
            Key for time dataset
        field_key : str
            Key for B-field dataset

        Returns
        -------
        data : FEMBFieldData
            Loaded field data
        """
        if not HAS_HDF5:
            raise ImportError("h5py required for HDF5 support. Install: pip install h5py")

        with h5py.File(filename, 'r') as f:
            time = f[time_key][:]
            B_field = f[field_key][:]

            # Load optional data
            position = f['position'][:] if 'position' in f else None
            frequency = float(f['frequency'][()]) if 'frequency' in f else None

            # Load metadata
            metadata = dict(f.attrs) if hasattr(f, 'attrs') else {}
            metadata['source'] = filename
            metadata['format'] = 'hdf5'

        return FEMBFieldData(time=time, B_field=B_field,
                           position=position, frequency=frequency,
                           metadata=metadata)

    @staticmethod
    def save_hdf5(filename: str, data: FEMBFieldData):
        """
        Save B-field data to HDF5 file.

        Parameters
        ----------
        filename : str
            Output HDF5 file path
        data : FEMBFieldData
            Data to save
        """
        if not HAS_HDF5:
            raise ImportError("h5py required for HDF5 support")

        with h5py.File(filename, 'w') as f:
            f.create_dataset('time', data=data.time)
            f.create_dataset('B_field', data=data.B_field)

            if data.position is not None:
                f.create_dataset('position', data=data.position)

            if data.frequency is not None:
                f.create_dataset('frequency', data=data.frequency)

            if data.metadata is not None:
                for key, val in data.metadata.items():
                    if isinstance(val, (int, float, str)):
                        f.attrs[key] = val


class MicrowaveAntennaSimulator:
    """
    Generate synthetic B-field patterns from microwave antennas.

    Useful for testing and when FEM data is not available.
    """

    @staticmethod
    def wire_antenna(position: np.ndarray, wire_length: float = 0.5,
                    power: float = 1.0, frequency: float = 2.87) -> float:
        """
        Calculate B-field amplitude from a wire antenna (dipole approximation).

        Parameters
        ----------
        position : array [x, y, z]
            Position relative to antenna center (cm)
        wire_length : float
            Antenna length (cm)
        power : float
            Microwave power (W)
        frequency : float
            MW frequency (GHz)

        Returns
        -------
        B_amplitude : float
            B-field amplitude (Gauss)
        """
        x, y, z = position
        r = np.sqrt(x**2 + y**2 + z**2)

        if r < 0.001:  # Avoid singularity
            r = 0.001

        # Simplified dipole radiation pattern
        # B ~ (power * wire_length) / r^2
        # Empirical scaling to get realistic Gauss values
        B_amplitude = np.sqrt(power) * wire_length / (r**2 + 0.01)

        # Add angular dependence (perpendicular to wire is strongest)
        theta = np.arctan2(np.sqrt(x**2 + y**2), z)
        B_amplitude *= np.sin(theta)

        return B_amplitude

    @staticmethod
    def coplanar_waveguide(position: np.ndarray, width: float = 0.02,
                          gap: float = 0.01, power: float = 1.0) -> Tuple[float, float]:
        """
        Calculate B-field from coplanar waveguide (CPW).

        Parameters
        ----------
        position : array [x, y, z]
            Position relative to CPW center (cm)
        width : float
            Central conductor width (cm)
        gap : float
            Gap between center and ground (cm)
        power : float
            Microwave power (W)

        Returns
        -------
        B_amplitude : float
            B-field amplitude (Gauss)
        angle : float
            Field direction angle (radians)
        """
        x, y, z = position

        # Distance from CPW surface
        h = np.abs(z)
        if h < 0.0001:
            h = 0.0001

        # Distance from center conductor
        x_dist = np.abs(x)

        # Simplified CPW field model
        # Field is strongest at edges of center conductor
        edge_pos = width / 2

        # Current distribution creates circulating B-field
        if x_dist < edge_pos:
            # Above center conductor
            B_amplitude = np.sqrt(power) * 2.0 / (h + gap)
        else:
            # Above gap or ground
            B_amplitude = np.sqrt(power) * 1.0 / (h + gap)

        # Decay with height
        B_amplitude *= np.exp(-h / (width + gap))

        # Field is primarily transverse (perpendicular to current)
        angle = np.arctan2(y, x)

        return B_amplitude, angle

    @staticmethod
    def generate_pulsed_mw(duration: float, frequency: float = 2.87,
                          amplitude: float = 1.0, envelope: str = 'gaussian',
                          rise_time: float = 0.01, n_points: int = 1000,
                          phase: float = 0.0) -> FEMBFieldData:
        """
        Generate time-domain microwave pulse with envelope.

        Parameters
        ----------
        duration : float
            Total pulse duration (microseconds)
        frequency : float
            MW carrier frequency (GHz)
        amplitude : float
            Peak B-field amplitude (Gauss)
        envelope : str
            'gaussian', 'rectangular', 'sech', or 'raised_cosine'
        rise_time : float
            Rise/fall time for edges (microseconds)
        n_points : int
            Number of time points
        phase : float
            Carrier phase (radians)

        Returns
        -------
        data : FEMBFieldData
            MW pulse field data
        """
        # Time array
        time = np.linspace(0, duration, n_points)
        t_center = duration / 2

        # Generate envelope
        if envelope == 'gaussian':
            sigma = duration / 6  # 6-sigma fits in duration
            env = np.exp(-0.5 * ((time - t_center) / sigma)**2)

        elif envelope == 'rectangular':
            env = np.ones_like(time)
            # Add rise/fall edges
            if rise_time > 0:
                rise_samples = int(rise_time / duration * n_points)
                env[:rise_samples] = np.linspace(0, 1, rise_samples)
                env[-rise_samples:] = np.linspace(1, 0, rise_samples)

        elif envelope == 'sech':
            t_scaled = (time - t_center) / (duration / 4)
            env = 1.0 / np.cosh(t_scaled)

        elif envelope == 'raised_cosine':
            env = 0.5 * (1 - np.cos(2 * np.pi * time / duration))

        else:
            raise ValueError(f"Unknown envelope: {envelope}")

        # Carrier oscillation (convert GHz to rad/µs)
        omega = 2 * np.pi * frequency * 1000  # GHz → rad/µs
        carrier = np.cos(omega * time + phase)

        # Modulated signal
        B_x = amplitude * env * carrier
        B_y = amplitude * env * np.sin(omega * time + phase)  # Quadrature component
        B_z = np.zeros_like(time)  # MW field is transverse

        B_field = np.column_stack([B_x, B_y, B_z])

        metadata = {
            'type': 'synthetic_mw_pulse',
            'frequency': frequency,
            'envelope': envelope,
            'amplitude': amplitude,
            'duration': duration
        }

        return FEMBFieldData(time=time, B_field=B_field,
                           frequency=frequency, metadata=metadata)

    @staticmethod
    def generate_spatial_field(antenna_type: str, grid_size: Tuple[int, int, int],
                              grid_spacing: float, time_points: np.ndarray,
                              frequency: float = 2.87, power: float = 1.0,
                              **kwargs) -> FEMBFieldData:
        """
        Generate spatially-resolved B-field from antenna.

        Parameters
        ----------
        antenna_type : str
            'wire' or 'cpw'
        grid_size : tuple (nx, ny, nz)
            Number of grid points in each direction
        grid_spacing : float
            Spacing between grid points (cm)
        time_points : array
            Time points (microseconds)
        frequency : float
            MW frequency (GHz)
        power : float
            MW power (W)
        **kwargs : dict
            Additional antenna parameters

        Returns
        -------
        data : FEMBFieldData
            Spatially-resolved field data
        """
        nx, ny, nz = grid_size

        # Create spatial grid
        x = np.linspace(-nx/2, nx/2, nx) * grid_spacing
        y = np.linspace(-ny/2, ny/2, ny) * grid_spacing
        z = np.linspace(0, nz, nz) * grid_spacing

        # Store grid
        position = (x, y, z)

        # Initialize field array
        n_times = len(time_points)
        B_field = np.zeros((n_times, nx, ny, nz, 3))

        # Calculate static field pattern
        B_static = np.zeros((nx, ny, nz, 3))

        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                for k, zk in enumerate(z):
                    pos = np.array([xi, yj, zk])

                    if antenna_type == 'wire':
                        B_amp = MicrowaveAntennaSimulator.wire_antenna(
                            pos, frequency=frequency, power=power, **kwargs)
                        # Field primarily in x-y plane
                        angle = np.arctan2(yj, xi)
                        B_static[i, j, k, 0] = B_amp * np.cos(angle)
                        B_static[i, j, k, 1] = B_amp * np.sin(angle)

                    elif antenna_type == 'cpw':
                        B_amp, angle = MicrowaveAntennaSimulator.coplanar_waveguide(
                            pos, power=power, **kwargs)
                        B_static[i, j, k, 0] = B_amp * np.cos(angle)
                        B_static[i, j, k, 1] = B_amp * np.sin(angle)

        # Modulate with time
        omega = 2 * np.pi * frequency * 1000  # rad/µs
        for t_idx, t in enumerate(time_points):
            carrier = np.cos(omega * t)
            B_field[t_idx, :, :, :, :] = B_static * carrier

        metadata = {
            'type': f'spatial_{antenna_type}',
            'grid_size': grid_size,
            'grid_spacing': grid_spacing,
            'antenna_type': antenna_type
        }

        return FEMBFieldData(time=time_points, B_field=B_field,
                           position=position, frequency=frequency,
                           metadata=metadata)


def create_mw_field_function(fem_data: FEMBFieldData,
                             position: Optional[np.ndarray] = None) -> Callable:
    """
    Create time-dependent field function from FEM data.

    Parameters
    ----------
    fem_data : FEMBFieldData
        FEM field data
    position : array, optional
        Position for spatial data extraction

    Returns
    -------
    field_function : callable
        Function f(t, args) returning B_field [Bx, By, Bz]
    """
    # Extract local field if spatial
    if fem_data.is_spatial:
        if position is None:
            raise ValueError("Position required for spatial field data")
        B_local = fem_data.get_field_at_position(position)
    else:
        B_local = fem_data.B_field

    # Create interpolators for each component
    interp_x = interp1d(fem_data.time, B_local[:, 0], kind='cubic',
                       bounds_error=False, fill_value=0.0)
    interp_y = interp1d(fem_data.time, B_local[:, 1], kind='cubic',
                       bounds_error=False, fill_value=0.0)
    interp_z = interp1d(fem_data.time, B_local[:, 2], kind='cubic',
                       bounds_error=False, fill_value=0.0)

    def field_function(t, args=None):
        """Return B-field at time t"""
        Bx = float(interp_x(t))
        By = float(interp_y(t))
        Bz = float(interp_z(t))
        return np.array([Bx, By, Bz])

    return field_function


if __name__ == "__main__":
    print("FEM B-Field Handler for NV-Center Simulation")
    print("=" * 50)
    print("\nCapabilities:")
    print("  • Load FEM data: CSV, NPY, HDF5")
    print("  • Generate synthetic antenna patterns")
    print("  • Time-domain MW pulses with envelopes")
    print("  • Spatial field profiles")
    print("  • Integration with NVCenter simulator")
    print("\nExample usage:")
    print("  from fem_bfield import MicrowaveAntennaSimulator")
    print("  mw_pulse = MicrowaveAntennaSimulator.generate_pulsed_mw(")
    print("      duration=1.0, frequency=2.87, amplitude=0.5)")
