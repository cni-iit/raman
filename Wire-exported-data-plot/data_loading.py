import numpy as np
import pandas as pd
from pathlib import Path

def parse_raman_map(filepath, progress_callback=None):
    """
    Parse Raman spectral map data from tab-separated file.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to the .txt file containing Raman map data
    progress_callback : callable, optional
        Function to call with progress updates (0-100)
    
    Returns:
    --------
    dict : Dictionary containing:
        - 'data': 3D numpy array (y, x, wavenumber)
        - 'x_coords': 1D array of x coordinates
        - 'y_coords': 1D array of y coordinates  
        - 'raman_shifts': 1D array of Raman shift values
        - 'metadata': dict with scan parameters
    """
    
    print(f"Loading Raman map data from: {filepath}")
    
    # Read the data file
    df = pd.read_csv(filepath, sep='\t', header=0)
    
    # Extract unique coordinates and Raman shifts
    x_coords = np.sort(df.iloc[:, 0].unique())  # x position (column 0)
    y_coords = np.sort(df.iloc[:, 1].unique())  # y position (column 1)
    raman_shifts = np.sort(df.iloc[:, 2].unique())  # Raman shift (column 2)
    
    # Dimensions
    nx, ny, n_wavenumbers = len(x_coords), len(y_coords), len(raman_shifts)
    
    print(f"Map dimensions: {nx} x {ny} pixels, {n_wavenumbers} wavenumbers")
    print(f"X range: {x_coords[0]:.2f} to {x_coords[-1]:.2f}")
    print(f"Y range: {y_coords[0]:.2f} to {y_coords[-1]:.2f}")
    print(f"Raman shift range: {raman_shifts[0]:.1f} to {raman_shifts[-1]:.1f} cm⁻¹")
    
    # Initialize 3D array: (y, x, wavenumber)
    spectral_map = np.zeros((ny, nx, n_wavenumbers))
    
    # Create coordinate-to-index mappings for fast lookup
    x_to_idx = {x: i for i, x in enumerate(x_coords)}
    y_to_idx = {y: i for i, y in enumerate(y_coords)}
    raman_to_idx = {r: i for i, r in enumerate(raman_shifts)}
    
    # Fill the 3D array
    total_points = len(df)
    for idx, (_, row) in enumerate(df.iterrows()):
        x_pos, y_pos, raman_shift, intensity = row.iloc[0], row.iloc[1], row.iloc[2], row.iloc[3]
        
        # Get indices
        x_idx = x_to_idx[x_pos]
        y_idx = y_to_idx[y_pos]
        r_idx = raman_to_idx[raman_shift]
        
        # Store intensity
        spectral_map[y_idx, x_idx, r_idx] = intensity
        
        # Progress callback
        if progress_callback and idx % (total_points // 20) == 0:
            progress_callback(int(100 * idx / total_points))
    
    if progress_callback:
        progress_callback(100)
    
    # Calculate metadata
    metadata = {
        'filename': Path(filepath).name,
        'map_size': (nx, ny),
        'total_spectra': nx * ny,
        'wavenumbers_count': n_wavenumbers,
        'x_step': x_coords[1] - x_coords[0] if len(x_coords) > 1 else 0,
        'y_step': y_coords[1] - y_coords[0] if len(y_coords) > 1 else 0,
        'wavenumber_step': raman_shifts[1] - raman_shifts[0] if len(raman_shifts) > 1 else 0,
        'data_shape': spectral_map.shape,
        'total_data_points': total_points
    }
    
    return {
        'data': spectral_map,
        'x_coords': x_coords,
        'y_coords': y_coords,
        'raman_shifts': raman_shifts,
        'metadata': metadata
    }

def get_spectrum_at_position(raman_map, x_pos, y_pos):
    """
    Extract spectrum at specific x, y position.
    
    Parameters:
    -----------
    raman_map : dict
        Output from parse_raman_map()
    x_pos, y_pos : float
        Coordinates of desired position
    
    Returns:
    --------
    tuple : (raman_shifts, intensities) or None if position not found
    """
    
    # Find closest coordinates
    x_idx = np.argmin(np.abs(raman_map['x_coords'] - x_pos))
    y_idx = np.argmin(np.abs(raman_map['y_coords'] - y_pos))
    
    spectrum = raman_map['data'][y_idx, x_idx, :]
    
    return raman_map['raman_shifts'], spectrum

def get_intensity_map_at_wavenumber(raman_map, target_wavenumber, tolerance=5):
    """
    Extract 2D intensity map at specific Raman shift.
    
    Parameters:
    -----------
    raman_map : dict
        Output from parse_raman_map()
    target_wavenumber : float
        Target Raman shift (cm⁻¹)
    tolerance : float
        Tolerance for wavenumber matching (cm⁻¹)
    
    Returns:
    --------
    tuple : (intensity_map, actual_wavenumber) or None if not found
    """
    
    # Find closest wavenumber
    diff = np.abs(raman_map['raman_shifts'] - target_wavenumber)
    closest_idx = np.argmin(diff)
    
    if diff[closest_idx] <= tolerance:
        intensity_map = raman_map['data'][:, :, closest_idx]
        actual_wavenumber = raman_map['raman_shifts'][closest_idx]
        return intensity_map, actual_wavenumber
    else:
        print(f"No wavenumber found within {tolerance} cm⁻¹ of {target_wavenumber}")
        return None, None

# Example usage
if __name__ == "__main__":
    # Example usage - replace with your file path
    filepath = "raman_map_data.txt"
    
    try:
        # Parse the map data
        def print_progress(percent):
            print(f"Progress: {percent}%")
        
        raman_map = parse_raman_map(filepath, progress_callback=print_progress)
        
        print("\n=== Parsing Complete ===")
        print(f"Data shape: {raman_map['data'].shape}")
        print(f"Memory usage: {raman_map['data'].nbytes / 1024**2:.1f} MB")
        
        # Example: Get spectrum at specific position
        x_pos, y_pos = raman_map['x_coords'][5], raman_map['y_coords'][3]
        wavenumbers, spectrum = get_spectrum_at_position(raman_map, x_pos, y_pos)
        print(f"\nSpectrum at ({x_pos}, {y_pos}): {len(spectrum)} data points")
        
        # Example: Get intensity map at specific wavenumber
        target_peak = 1580  # Example: G-band of graphene
        intensity_map, actual_wn = get_intensity_map_at_wavenumber(raman_map, target_peak)
        if intensity_map is not None:
            print(f"Intensity map at {actual_wn:.1f} cm⁻¹: {intensity_map.shape}")
            print(f"Intensity range: {intensity_map.min():.0f} - {intensity_map.max():.0f}")
        
        # Save processed data (optional)
        # np.savez_compressed('raman_map_processed.npz', 
        #                    data=raman_map['data'],
        #                    x_coords=raman_map['x_coords'],
        #                    y_coords=raman_map['y_coords'], 
        #                    raman_shifts=raman_map['raman_shifts'])
        
    except FileNotFoundError:
        print(f"File {filepath} not found. Please check the file path.")
    except Exception as e:
        print(f"Error processing file: {e}")