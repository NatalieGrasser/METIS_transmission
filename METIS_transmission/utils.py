import numpy as np
import astropy.constants as const
from astropy import units as u
from scipy.ndimage import convolve1d
import pickle
import pandas as pd

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
