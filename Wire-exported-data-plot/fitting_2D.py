import numpy as np
from scipy.optimize import curve_fit
from scipy.special import wofz
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class RamanSpectraFitter:
    """
    A class for fitting Raman spectra with a single Voigt profile and 
    a constrained envelope of four Voigt profiles.
    """
    
    def __init__(self, raman_shift, envelope_constraints=None):
        """
        Initialize the fitter.
        
        Parameters:
        -----------
        raman_shift : array-like
            The Raman shift values (wavenumbers)
        envelope_constraints : dict, optional
            Constraints for the envelope. Should contain:
            - 'center_spacings': list of 3 values for relative positions of peaks 2,3,4
            - 'width_ratios': list of 3 values for relative widths of peaks 2,3,4
            - 'amplitude_ratios': list of 3 values for relative amplitudes of peaks 2,3,4
        """
        self.raman_shift = np.array(raman_shift)
        
        # Default envelope constraints (example for a typical 4-peak pattern)
        if envelope_constraints is None:
            self.envelope_constraints = {
                'center_spacings': [50, 100, 150],  # Peak spacings in wavenumbers
                'width_ratios': [1.0, 1.2, 0.8],   # Relative widths
                'amplitude_ratios': [0.8, 0.6, 0.4] # Relative amplitudes
            }
        else:
            self.envelope_constraints = envelope_constraints
    
    @staticmethod
    def voigt_profile(x, center, width, amplitude, gamma_ratio):
        """
        Compute a Voigt profile.
        
        Parameters:
        -----------
        x : array-like
            Independent variable (Raman shift)
        center : float
            Peak center position
        width : float
            Peak width (FWHM)
        amplitude : float
            Peak amplitude
        gamma_ratio : float
            Ratio of Lorentzian to Gaussian character (0=pure Gaussian, 1=pure Lorentzian)
        
        Returns:
        --------
        array-like
            Voigt profile values
        """
        # Convert FWHM to standard deviations
        sigma = width / (2 * np.sqrt(2 * np.log(2)))  # Gaussian component
        gamma = width * gamma_ratio / 2  # Lorentzian component
        
        # Voigt profile using Faddeeva function
        z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
        voigt = np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
        
        return amplitude * voigt
    
    def envelope_four_voigt(self, x, center1, width1, amplitude1, gamma_ratio):
        """
        Compute the envelope of four constrained Voigt profiles.
        
        Parameters:
        -----------
        x : array-like
            Independent variable (Raman shift)
        center1 : float
            Center of the first peak
        width1 : float
            Width of the first peak
        amplitude1 : float
            Amplitude of the first peak
        gamma_ratio : float
            Shared Gaussian/Lorentzian character for all peaks
        
        Returns:
        --------
        array-like
            Sum of four Voigt profiles
        """
        # First peak
        envelope = self.voigt_profile(x, center1, width1, amplitude1, gamma_ratio)
        
        # Additional three peaks with constraints
        constraints = self.envelope_constraints
        
        for i, (spacing, width_ratio, amp_ratio) in enumerate(zip(
            constraints['center_spacings'],
            constraints['width_ratios'], 
            constraints['amplitude_ratios']
        )):
            center_i = center1 + spacing
            width_i = width1 * width_ratio
            amplitude_i = amplitude1 * amp_ratio
            
            envelope += self.voigt_profile(x, center_i, width_i, amplitude_i, gamma_ratio)
        
        return envelope
    
    def full_model(self, x, single_center, single_width, single_amplitude, single_gamma,
                   env_center, env_width, env_amplitude, env_gamma):
        """
        Full model: single Voigt + envelope of four Voigt profiles.
        
        Parameters:
        -----------
        x : array-like
            Independent variable (Raman shift)
        single_* : float
            Parameters for the single Voigt profile
        env_* : float
            Parameters for the envelope
        
        Returns:
        --------
        array-like
            Combined model
        """
        single_voigt = self.voigt_profile(x, single_center, single_width, 
                                        single_amplitude, single_gamma)
        envelope = self.envelope_four_voigt(x, env_center, env_width, 
                                          env_amplitude, env_gamma)
        
        return single_voigt + envelope
    
    def fit_single_spectrum(self, intensity, initial_guess=None, bounds=None):
        """
        Fit a single spectrum.
        
        Parameters:
        -----------
        intensity : array-like
            Spectrum intensity values
        initial_guess : list, optional
            Initial parameter guess [single_center, single_width, single_amplitude, single_gamma,
                                   env_center, env_width, env_amplitude, env_gamma]
        bounds : tuple, optional
            Parameter bounds (lower_bounds, upper_bounds)
        
        Returns:
        --------
        dict
            Fitting results containing parameters, covariance, and fit quality metrics
        """
        if initial_guess is None:
            # Auto-generate initial guess based on spectrum
            max_idx = np.argmax(intensity)
            max_pos = self.raman_shift[max_idx]
            max_int = np.max(intensity)
            
            initial_guess = [
                max_pos - 50,  # single_center
                20,            # single_width
                max_int * 0.3, # single_amplitude
                0.5,           # single_gamma
                max_pos + 50,  # env_center
                30,            # env_width
                max_int * 0.7, # env_amplitude
                0.5            # env_gamma
            ]
        
        if bounds is None:
            # Default bounds
            raman_min, raman_max = self.raman_shift.min(), self.raman_shift.max()
            lower_bounds = [raman_min, 1, 0, 0, raman_min, 1, 0, 0]
            upper_bounds = [raman_max, 200, np.inf, 1, raman_max, 200, np.inf, 1]
            bounds = (lower_bounds, upper_bounds)
        
        try:
            popt, pcov = curve_fit(self.full_model, self.raman_shift, intensity,
                                 p0=initial_guess, bounds=bounds, maxfev=5000)
            
            # Calculate fit quality metrics
            fitted_intensity = self.full_model(self.raman_shift, *popt)
            residuals = intensity - fitted_intensity
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((intensity - np.mean(intensity))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Calculate individual components
            single_component = self.voigt_profile(self.raman_shift, popt[0], popt[1], popt[2], popt[3])
            envelope_component = self.envelope_four_voigt(self.raman_shift, popt[4], popt[5], popt[6], popt[7])
            
            # Calculate relative intensities (integrated areas)
            single_intensity = np.trapz(single_component, self.raman_shift)
            envelope_intensity = np.trapz(envelope_component, self.raman_shift)
            total_intensity = single_intensity + envelope_intensity
            
            relative_single = single_intensity / total_intensity if total_intensity != 0 else 0
            relative_envelope = envelope_intensity / total_intensity if total_intensity != 0 else 0
            
            return {
                'success': True,
                'parameters': popt,
                'covariance': pcov,
                'r_squared': r_squared,
                'fitted_spectrum': fitted_intensity,
                'single_component': single_component,
                'envelope_component': envelope_component,
                'residuals': residuals,
                'relative_intensities': {
                    'single': relative_single,
                    'envelope': relative_envelope
                },
                'parameter_names': ['single_center', 'single_width', 'single_amplitude', 
                                  'single_gamma', 'env_center', 'env_width', 
                                  'env_amplitude', 'env_gamma']
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'parameters': np.full(8, np.nan),
                'relative_intensities': {'single': np.nan, 'envelope': np.nan}
            }
    
    def fit_spectral_map(self, spectral_map, initial_guess=None, bounds=None, 
                        progress_bar=True):
        """
        Fit an entire 3D spectral map.
        
        Parameters:
        -----------
        spectral_map : ndarray
            3D array with shape (y_positions, x_positions, raman_shift_points)
        initial_guess : list, optional
            Initial parameter guess
        bounds : tuple, optional
            Parameter bounds
        progress_bar : bool
            Whether to show progress bar
        
        Returns:
        --------
        dict
            Dictionary containing fitted parameters and relative intensities for each position
        """
        y_size, x_size, _ = spectral_map.shape
        
        # Initialize result arrays
        parameters = np.full((y_size, x_size, 8), np.nan)
        relative_single = np.full((y_size, x_size), np.nan)
        relative_envelope = np.full((y_size, x_size), np.nan)
        r_squared_map = np.full((y_size, x_size), np.nan)
        success_map = np.zeros((y_size, x_size), dtype=bool)
        
        # Fit each spectrum
        total_spectra = y_size * x_size
        progress_iter = tqdm(range(total_spectra), desc="Fitting spectra") if progress_bar else range(total_spectra)
        
        for idx in progress_iter:
            y_idx = idx // x_size
            x_idx = idx % x_size
            
            spectrum = spectral_map[y_idx, x_idx, :]
            
            # Skip if spectrum is all zeros or has insufficient signal
            if np.max(spectrum) < np.std(spectrum) * 2:
                continue
            
            result = self.fit_single_spectrum(spectrum, initial_guess, bounds)
            
            if result['success']:
                parameters[y_idx, x_idx, :] = result['parameters']
                relative_single[y_idx, x_idx] = result['relative_intensities']['single']
                relative_envelope[y_idx, x_idx] = result['relative_intensities']['envelope']
                r_squared_map[y_idx, x_idx] = result['r_squared']
                success_map[y_idx, x_idx] = True
        
        return {
            'parameters': parameters,
            'relative_intensities': {
                'single': relative_single,
                'envelope': relative_envelope
            },
            'r_squared': r_squared_map,
            'success_map': success_map,
            'parameter_names': ['single_center', 'single_width', 'single_amplitude', 
                              'single_gamma', 'env_center', 'env_width', 
                              'env_amplitude', 'env_gamma']
        }
    
    def plot_fit_example(self, spectrum, fit_result, title="Raman Spectrum Fit"):
        """
        Plot an example fit to visualize the results.
        
        Parameters:
        -----------
        spectrum : array-like
            Original spectrum
        fit_result : dict
            Result from fit_single_spectrum
        title : str
            Plot title
        """
        if not fit_result['success']:
            print(f"Cannot plot: fitting failed with error: {fit_result['error']}")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Main plot
        ax1.plot(self.raman_shift, spectrum, 'k-', label='Experimental', linewidth=1.5)
        ax1.plot(self.raman_shift, fit_result['fitted_spectrum'], 'r--', 
                label='Total Fit', linewidth=2)
        ax1.plot(self.raman_shift, fit_result['single_component'], 'b:', 
                label='Single Voigt', linewidth=2)
        ax1.plot(self.raman_shift, fit_result['envelope_component'], 'g:', 
                label='4-Voigt Envelope', linewidth=2)
        
        ax1.set_xlabel('Raman Shift (cm⁻¹)')
        ax1.set_ylabel('Intensity')
        ax1.set_title(f'{title} (R² = {fit_result["r_squared"]:.4f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Residuals
        ax2.plot(self.raman_shift, fit_result['residuals'], 'k-', linewidth=1)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Raman Shift (cm⁻¹)')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Fit Residuals')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print relative intensities
        rel_int = fit_result['relative_intensities']
        print(f"Relative Intensities:")
        print(f"  Single Voigt: {rel_int['single']:.3f}")
        print(f"  4-Voigt Envelope: {rel_int['envelope']:.3f}")

# Example usage and demonstration
if __name__ == "__main__":
    # Generate example data
    np.random.seed(42)
    
    # Create synthetic Raman shift axis
    raman_shift = np.linspace(800, 1800, 500)
    
    # Create synthetic 3D spectral map (small example)
    y_size, x_size = 5, 5
    spectral_map = np.zeros((y_size, x_size, len(raman_shift)))
    
    # Generate synthetic spectra with varying parameters
    fitter = RamanSpectraFitter(raman_shift)
    
    for y in range(y_size):
        for x in range(x_size):
            # Vary parameters across the map
            single_params = [1000 + y*10, 25, 100 + x*20, 0.5]
            env_params = [1200 + x*15, 40, 200 + y*30, 0.3]
            
            # Generate synthetic spectrum
            spectrum = (fitter.voigt_profile(raman_shift, *single_params) + 
                       fitter.envelope_four_voigt(raman_shift, *env_params))
            
            # Add noise
            noise = np.random.normal(0, 5, len(raman_shift))
            spectral_map[y, x, :] = spectrum + noise
    
    print("Example: Fitting synthetic Raman spectral map...")
    
    # Fit the spectral map
    results = fitter.fit_spectral_map(spectral_map, progress_bar=True)
    
    print(f"Fitting completed!")
    print(f"Success rate: {np.sum(results['success_map']) / results['success_map'].size * 100:.1f}%")
    print(f"Average R²: {np.nanmean(results['r_squared']):.4f}")
    
    # Plot an example fit
    example_spectrum = spectral_map[2, 2, :]
    example_fit = fitter.fit_single_spectrum(example_spectrum)
    fitter.plot_fit_example(example_spectrum, example_fit, 
                           title="Example Fit (Position [2,2])")
    
    # Display relative intensity maps
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = axes[0].imshow(results['relative_intensities']['single'], 
                        cmap='viridis', aspect='equal')
    axes[0].set_title('Single Voigt Relative Intensity')
    axes[0].set_xlabel('X Position')
    axes[0].set_ylabel('Y Position')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(results['relative_intensities']['envelope'], 
                        cmap='plasma', aspect='equal')
    axes[1].set_title('4-Voigt Envelope Relative Intensity')
    axes[1].set_xlabel('X Position')
    axes[1].set_ylabel('Y Position')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.show()
    
    print("\nTo use with your data:")
    print("1. Load your 3D numpy array (y, x, raman_shift)")
    print("2. Create fitter: fitter = RamanSpectraFitter(your_raman_shift_axis)")
    print("3. Optionally set envelope constraints via fitter.envelope_constraints")
    print("4. Fit: results = fitter.fit_spectral_map(your_spectral_map)")
    print("5. Access relative intensities: results['relative_intensities']")