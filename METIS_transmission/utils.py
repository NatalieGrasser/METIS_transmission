import numpy as np
import astropy.constants as const
from astropy import units as u
from scipy.ndimage import convolve1d
import pickle
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Define your custom color palette
custom_colors = [
   '#240b3cff', # first color
   '#6929bbff',
   '#2faaeaff',
   '#ffe437ff' # last color
]

# Create a ListedColormap
custom_cmap = LinearSegmentedColormap.from_list("custom_gradient", custom_colors)

def save_pickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# from DGonzalezPicos/broadpy
class InstrumentalBroadening:
    
    c = const.c.to(u.km/u.s).value
    sqrt8ln2 = np.sqrt(8 * np.log(2))
    
    available_kernels = ['gaussian','gaussian_variable']
    
    def __init__(self, x, y):
        
        self.x = x # units of wavelength
        self.y = y # units of flux (does not matter)
        self.spacing = np.mean(2*np.diff(self.x) / (self.x[1:] + self.x[:-1]))
    
    def __call__(self, res=None, fwhm=None, gamma=None, truncate=4.0, kernel='auto'):
        '''Instrumental broadening
        provide either instrumental resolution lambda/delta_lambda or FWHM in km/s'''
        kernel = self.__read_kernel(res=res, fwhm=fwhm, gamma=gamma) if kernel == 'auto' else kernel
        
        if kernel == 'gaussian':
            fwhm = fwhm if fwhm is not None else (self.c / res)
            _kernel = self.gaussian_kernel(fwhm, truncate)
            
        if kernel == 'gaussian_variable':
            _kernels, lw = self.gaussian_variable_kernel(fwhm, truncate)
            y_pad = np.pad(self.y, (lw, lw), mode='reflect')
            y_matrix = np.lib.stride_tricks.sliding_window_view(y_pad, window_shape=(2 * lw + 1))
            y_lsf = np.einsum('ij, ij->i', _kernels, y_matrix)
            return y_lsf
            
        y_lsf = convolve1d(self.y, _kernel, mode='nearest')
        return y_lsf
    
    @classmethod
    def gaussian_profile(self, x, x0, sigma):
        '''Gaussian function'''
        return np.exp(-0.5 * ((x - x0) / sigma)**2)# / (sigma * np.sqrt(2*np.pi))
    
    def gaussian_kernel(self,fwhm,truncate=4.0,):
        ''' Gaussian kernel
        
        Parameters
        ----------
        fwhm : float
            Full width at half maximum of the Gaussian kernel in km/s
        truncate : float
            Truncate the kernel at this many standard deviations from the mean (default: 4.0)
        
        Returns
        -------
        kernel : array
            Convolution kernel
        '''
        # Adapted from scipy.ndimage.gaussian_filter1d        
        sd = (fwhm/self.c) / self.sqrt8ln2 / self.spacing
        lw = int(truncate * sd + 0.5)
    
        kernel_x = np.arange(-lw, lw+1)
        kernel = self.gaussian_profile(kernel_x, 0, sd)
        kernel /= np.sum(kernel)  # normalize the kernel
        return kernel

def instrumental_broadening(wave, flux, resolution=100000, fwhm=None):

    IB = InstrumentalBroadening(wave, flux)
    if fwhm==None: 
        flux_LSF = IB(res=resolution, kernel='gaussian')
    else: # fwhm in km/s
        flux_LSF = IB(fwhm=fwhm, kernel='gaussian')
    return flux_LSF

class PSG_input:
    def __init__(self,name):
        self.name = name
        self.table = self.create_table()
        self.pressure = self.table['Pressure'].to_numpy()
        self.temperature = self.table['Temperature'].to_numpy()

    def create_table(self): # convert PSG input file into useable table

        with open(f'./{self.name}_psg_input.txt', "r") as file:
            lines = file.readlines()

        rows = []
        for line in lines:
            if "<ATMOSPHERE-LAYER-" in line:
                index = line.index(">")
                rows.append(line[index+1:-1]) # remove \n from end of row

        columns = [ "Pressure", "Temperature", "Altitude", "H2", "He", "H2O", "CH4", "C2H6", "CO2", "C2H2", "C2H4", "CO",
                    "H2CO", "NH3", "SO2", "H2S", "SO", "CS2", "OCS", "C2H6S", "C2H6S2"]

        df = pd.DataFrame([row.split(",") for row in rows])
        df.columns = columns
        df = df.astype(float) # Convert all columns to float
        df = df.iloc[::-1] # reverse order, bc pRT reads temps from top to bottom of atmosphere

        return df
    
def copy_axes_contents(src_ax, dst_ax):
    # Copy lines by recreating them
    for line in src_ax.get_lines():
        dst_ax.plot(
            line.get_xdata(),
            line.get_ydata(),
            linestyle=line.get_linestyle(),
            linewidth=line.get_linewidth(),
            color=line.get_color(),
            marker=line.get_marker(),
            markersize=line.get_markersize(),
            markerfacecolor=line.get_markerfacecolor(),
            markeredgecolor=line.get_markeredgecolor(),
            alpha=line.get_alpha(),
            label=line.get_label(),
            zorder=line.get_zorder(),
        )

    # Copy axis settings
    dst_ax.set_xlim(src_ax.get_xlim())
    dst_ax.set_ylim(src_ax.get_ylim())
    dst_ax.set_xscale(src_ax.get_xscale())
    dst_ax.set_yscale(src_ax.get_yscale())
    dst_ax.set_xlabel(src_ax.get_xlabel())
    dst_ax.set_ylabel(src_ax.get_ylabel())
    dst_ax.set_title(src_ax.get_title())

    # Copy legend if present
    leg = src_ax.get_legend()
    if leg is not None:
        dst_ax.legend(
            handles=dst_ax.get_lines(),
            labels=[t.get_text() for t in leg.get_texts()],
            loc=leg._loc,
        )

def snr_per_order_and_pixel(flux, err, n_trans=0):
    """
    Compute S/N per wavelength pixel and per spectral order for one transit,
    with order S/N defined as the mean of the per-pixel S/N.

    Returns
    -------
    snr_pixel_list : list of arrays
        S/N per wavelength pixel for each order
    snr_order : ndarray
        Average S/N per order (mean of pixel S/N)
    """

    _, n_exp, n_orders = flux.shape
    snr_order = np.zeros(n_orders)
    snr_pixel_list = []

    for o in range(n_orders):
        flux_list = []
        err_list = []

        for e in range(n_exp):
            f = flux[n_trans, e, o]
            s = err[n_trans, e, o]

            if f is None or s is None:
                continue

            mask = np.isfinite(f) & np.isfinite(s) & (s > 0)
            if np.any(mask):
                flux_list.append(f[mask])
                err_list.append(s[mask])

        if len(flux_list) == 0:
            snr_pixel_list.append(np.array([]))
            snr_order[o] = np.nan
            continue

        flux_stack = np.vstack(flux_list)
        err_stack = np.vstack(err_list)

        # Average flux over exposures
        flux_mean = np.mean(flux_stack, axis=0)

        # Combine errors in quadrature over exposures
        sigma_mean = np.sqrt(np.sum(err_stack**2, axis=0)) / flux_stack.shape[0]

        # Compute per-pixel S/N
        valid = sigma_mean > 0
        snr_pixel = np.zeros_like(sigma_mean)
        snr_pixel[valid] = flux_mean[valid] / sigma_mean[valid]

        snr_pixel_list.append(snr_pixel)

        # Average S/N per order: mean of per-pixel S/N
        if np.any(valid):
            snr_order[o] = np.mean(snr_pixel[valid])
        else:
            snr_order[o] = np.nan

    return snr_pixel_list, snr_order

def fix_zero_errors(flux, err, flux_threshold=10.0):
    """
    Update error array in-place to replace zeros.

    Parameters
    ----------
    flux : object array, shape (n_trans, n_exp, n_orders)
        Each entry is a 1D flux array (variable wavelength)
    err : object array, same shape
        Each entry is a 1D error array
    flux_threshold : float
        Threshold to decide if flux is "near zero"

    Returns
    -------
    err_fixed : object array
        New error array with zeros replaced
    """

    n_trans, n_exp, n_orders = flux.shape
    err_fixed = np.empty_like(err)

    # Compute overall median of all nonzero errors for fallback
    all_nonzero_err = np.hstack([err[t,e,o][err[t,e,o] > 0] 
                                 for t in range(n_trans) 
                                 for e in range(n_exp) 
                                 for o in range(n_orders) 
                                 if err[t,e,o] is not None])
    median_err_global = np.median(all_nonzero_err)

    for t in range(n_trans):
        for e in range(n_exp):
            for o in range(n_orders):
                f = flux[t,e,o]
                s = err[t,e,o]

                if f is None or s is None:
                    err_fixed[t,e,o] = s
                    continue

                s_fixed = s.copy()
                mask_zero = (s_fixed == 0)

                if np.any(mask_zero):
                    # Case 1: flux near zero
                    mask_flux_near0 = mask_zero & (f < flux_threshold)
                    if np.any(mask_flux_near0):
                        # average error where flux is low and error is > 0
                        candidate_errors = s[(f < flux_threshold) & (s > 0)]
                        if len(candidate_errors) > 0:
                            avg_err = np.mean(candidate_errors)
                        else:
                            avg_err = median_err_global
                        s_fixed[mask_flux_near0] = avg_err

                    # Case 2: flux not near zero
                    mask_flux_not0 = mask_zero & (f >= flux_threshold)
                    s_fixed[mask_flux_not0] = median_err_global

                err_fixed[t,e,o] = s_fixed

    return err_fixed
