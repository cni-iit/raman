import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib import rcParams
import matplotlib.transforms as mtransforms

def plot_raman_spectrum(
        file_path, 
        output_file='raman_spectrum.png', 
        break_xaxis=False, 
        break_start=None, 
        break_end=None, 
        figsize=(10, 3), 
        dpi=300, 
        fontsize=15
        ):
    """
    Plot a Raman spectrum from a tab-separated text file and export as image.
    
    Parameters:
    -----------
    file_path : str
        Path to tab-separated file with Raman data
    output_file : str, optional
        Name of output file (default: 'raman_spectrum.png')
    break_xaxis : bool, optional
        Whether to include a break in the x-axis (default: False)
    break_start : float, optional
        Start of the x-axis break range (required if break_xaxis=True)
    break_end : float, optional
        End of the x-axis break range (required if break_xaxis=True)
    figsize : tuple, optional
        Figure dimensions (width, height) in inches
    dpi : int, optional
        Resolution of output image
    fontsize : int, optional
        Value to adjust the font throughout the picture
    """
    # Read data from file
    try:
        data = np.genfromtxt(file_path, delimiter='\t', skip_header=1, usecols=(2, 3))
        raman_shift = data[:, 0]
        intensity = data[:, 1]
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Normalize intensity
    normalized_intensity = intensity / np.max(intensity)
    
    # Setup plot style
    rcParams['font.size'] = fontsize
    
    # Prepare data and plot for broken-x-axis type of plot
    if break_xaxis and break_start is not None and break_end is not None:
        # Split data for broken axis
        mask_left = raman_shift < break_start
        mask_right = raman_shift > break_end
        
        # Calculate data ranges for each section to determine width ratios
        left_range = break_start - np.min(raman_shift[mask_left])
        right_range = np.max(raman_shift[mask_right]) - break_end
        
        # Set width ratios to maintain the same x-axis scale
        width_ratio = left_range / right_range
        
        # Create figure with two subplots for broken axis
        fig, (ax1, ax2) = plt.subplots(
            1, 
            2, 
            sharey=True, 
            figsize=figsize,
            gridspec_kw={'width_ratios': [width_ratio, 1], 'wspace': 0.05}
            )
        
        # Left subplot
        ax1.plot(raman_shift[mask_left], normalized_intensity[mask_left], 'k-', linewidth=1)
        ax1.set_xlim(np.min(raman_shift[mask_left]), break_start)
        
        # Right subplot
        ax2.plot(raman_shift[mask_right], normalized_intensity[mask_right], 'k-', linewidth=1)
        ax2.set_xlim(break_end, np.max(raman_shift[mask_right]))
        
        # Hide the spines between ax1 and ax2
        ax1.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        
        # Remove ticks from the left side of ax2
        ax2.tick_params(left=False, which='both')  # Remove both major and minor ticks
        
        # Add slanted break lines to x-axis
        # For x-axis
        kwargs = dict(
            marker=[(-0.5, -1), (0.5, 1)], 
            markersize=12, 
            linestyle="none",
            color='k', 
            mec='k', 
            mew=1, 
            clip_on=False
            )
        
        # Add break marks on x-axis
        ax1.plot([1], [0], transform=ax1.transAxes, **kwargs)
        ax1.plot([1], [1], transform=ax1.transAxes, **kwargs)
        ax2.plot([0], [0], transform=ax2.transAxes, **kwargs)
        ax2.plot([0], [1], transform=ax2.transAxes, **kwargs)
        
        # Set labels - only one common x-axis label
        ax1.set_ylabel('Normalized Intensity (a.u.)')
        fig.text(0.5, 0.04, 'Raman Shift (cm$^{-1}$)', ha='center', va='center', fontsize=fontsize)
        
        # Adding minor ticks (but no grid)
        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        ax2.xaxis.set_minor_locator(AutoMinorLocator())
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        
        plt.subplots_adjust(bottom=0.15)
    
    # Plot the data without breaking the x axis
    else:
        # Create a single plot without breaks
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(raman_shift, normalized_intensity, 'k-', linewidth=1.5)
        
        # Set labels and ticks
        ax.set_xlabel('Raman Shift (cm$^{-1}$)')
        ax.set_ylabel('Normalized Intensity (a.u.)')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        
        plt.tight_layout()
    
    # Save the figure
    try:
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"Raman spectrum saved as {output_file}")
    except Exception as e:
        print(f"Error saving file: {e}")
    
    # Display the plot
    plt.show()


# Usage
if __name__ == "__main__":
    
    # Example with broken x-axis
    plot_raman_spectrum(
        file_path="example.txt",  # Replace with your actual file path
        output_file="raman_spectrum_with_break.png",  # Replace with the desired picture name
        break_xaxis= True,
        break_start= 1990,
        break_end=   2510,
        figsize=(10, 3), 
        dpi=300, 
        fontsize=15
    )