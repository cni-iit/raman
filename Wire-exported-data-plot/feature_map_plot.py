import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread
import matplotlib.colors as colors
from matplotlib import rcParams
from scipy.interpolate import griddata


def load_colormap_from_snake_scan(file_path):
    """
    Load colormap data from a tab-separated file with three columns:
    x position, y position, and colormap value, where data are collected
    in a snake scan pattern.
    
    Parameters:
    -----------
    file_path : str
        Path to the tab-separated text file
        
    Returns:
    --------
    numpy.ndarray
        2D array of colormap values properly arranged by x,y coordinates
    """
    # Load the data from the file
    data = pd.read_csv(file_path, sep='\t', header=0)
    
    # Extract columns
    x = data.iloc[:, 0].values
    y = data.iloc[:, 1].values
    z = data.iloc[:, 2].values
    
    # Find unique x and y coordinates
    x_unique = np.sort(np.unique(x))
    y_unique = np.sort(np.unique(y))
    
    nx = len(x_unique)
    ny = len(y_unique)
    
    # Create empty 2D grid for the data
    z_grid = np.zeros((ny, nx)) #* np.nan
    
    # Map each (x,y) point to its correct position in the grid
    for i in range(len(x)):
        # Find indices in the grid
        x_idx = np.where(x_unique == x[i])[0][0]
        y_idx = np.where(y_unique == y[i])[0][0]
        
        # Place value in grid
        z_grid[y_idx, x_idx] = z[i]
    
    # Check for any NaN values (missing data points)
    if np.isnan(z_grid).any():
        print(f"Warning: File {os.path.basename(file_path)} has missing data points. Interpolating...")
        # Simple nearest neighbor interpolation for missing points
        mask = np.isnan(z_grid)
        z_grid[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), z_grid[~mask])
    
    return np.flip(z_grid, axis=0)


def process_raman_colormap(
    file_path, 
    output_file='raman_feature_map.png',
    colormap='autumn', 
    min_value=10,
    max_value=100,
    colorbar_label="feature (a.u.)",
    figsize=(12, 10),
    dpi=300,
    fontsize=15
    ):
    """
    Overlay multiple colormaps from txt files on a JPG image and add colorbars outside the plot.
    
    Parameters:
    -----------
    file_path : str
        Path to tab-separated file with Raman feature map data
    output_file : str
        Name of output file (default: 'rraman_feature_map.png')
    colormap : str, optional
        Colormap to be used in the plot
    min_value : int, optional
        Minimum value of the color axis
    max_value : int, optional
        Maximum value of the color axis
    colorbar_label : str, optional
        String to label the colormap
    figsize : tuple, optional
        Figure size in inches
    dpi : int, optional
        Resolution of output image
    fontsize : int, optional
        Value to adjust the font throughout the picture
    """
    # Process each txt file
    try:
        file_name = os.path.basename(file_path)
        
        # Load colormap data
        colormap_data = load_colormap_from_snake_scan(file_path)
    except Exception as e:
        print(f"No txt files found matching '{file_name}'")
        return
    
    # Update all font sizes
    rcParams['font.size'] = fontsize
    
    # Create figure, axes and turn off the axes
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    
    # Plot the colormap
    im = ax.imshow(
        colormap_data, 
        cmap=colormap, 
        vmin=min_value, 
        vmax=max_value, 
        origin='lower'
        )
    
    # Add colorbar
    cbar = fig.colorbar(im, label=colorbar_label)
    
    # Save the figure
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Image saved to {output_file}")


# Usage
if __name__ == "__main__":
    
    process_raman_colormap(
        file_path="example.txt",  # Replace with your actual file path
        output_file="raman_feature_map.png",  # Replace with the desired picture name
        colormap='autumn',
        min_value=  10,
        max_value= 100,
        colorbar_label="feature (a.u.)",
        figsize=(12, 10),
        dpi=300,
        fontsize=15
        )