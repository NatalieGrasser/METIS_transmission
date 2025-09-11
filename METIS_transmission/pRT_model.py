import numpy as np
import astropy.units as u
import os
#from petitRADTRANS.config import petitradtrans_config_parser
#petitradtrans_config_parser.set_input_data_path('/net/lem/data2/pRT3_formatted')
from utils import *
from petitRADTRANS.radtrans import Radtrans
import pathlib
from astropy.io import fits
from astropy.table import QTable

import getpass
if getpass.getuser() == "grasser": # when runnig from LEM
    import matplotlib
    matplotlib.use('Agg') # disable interactive plotting
    fastchem_tables = '/net/lem/data2/regt/fastchem_tables'
elif getpass.getuser() == "natalie": # when testing from my laptop
    fastchem_tables = '/home/natalie/fastchem_tables'

class pRT_spectrum:

    def __init__(self,
                 parameters):
        
        self.params = parameters
        self.project_path = parameters['project_path']
        self.pressure = parameters['pressure']
        self.n_atm_layers = len(self.pressure)
        temp = parameters['temperature']
        self.temperature = np.array(temp) if not np.isscalar(temp) else temp * np.ones(self.n_atm_layers)
        
        self.species_info = pd.read_csv('species_info.csv', index_col=0)
        #self.species_names = [k.replace("log_", "") for k in parameters.keys() 
                              #if k.startswith("log_") and k != "log_g"]
        self.species_names = parameters['species_names']
        line_species = [s for s in self.species_names if s not in ('H2', 'He')]
        self.species_pRT, self.species_hill = self.get_pRT_hill(line_species)
        self.gravity = 10 ** parameters['log_g']
        self.planet_radius = parameters['planet_radius']
        self.wave_range = parameters['pRT_wave_range_um']

        if self.params['chemistry']=='equ': # use equilibium chemistry
            self.mass_fractions = self.equ_chemistry(self.species_pRT,self.params)

        elif self.params['chemistry']=='free': # use free chemistry with defined VMRs
            self.mass_fractions, self.CO, self.FeH = self.free_chemistry(self.species_pRT,self.params)

        elif self.params['chemistry'] in ['Sorg1X','Sorg20X']: # altitude-dependent VMRs
            self.VMRs = {}
            for species_i in self.species_names:
                self.VMRs[species_i] = self.params[species_i]
            self.VMRs = self.get_H2(self.VMRs)
            self.mass_fractions = self.VMR_to_MF(self.VMRs)

        self.MMW = self.mass_fractions['MMW']

    def get_pRT_hill(self,species_names): # get pRT species name and hill notations
        species_pRT=[] # pRT names
        species_hill=[] # hill notation
        for species_i in species_names:
            species_pRT.append(self.species_info.loc[species_i,'pRT_name'])
            species_hill.append(self.species_info.loc[species_i,'Hill_notation'])
        return species_pRT, species_hill
        
    def read_species_info(self,species,info_key):
        if info_key == 'pRT_name':
            return self.species_info.loc[species,info_key]
        if info_key == 'pyfc_name':
            return self.species_info.loc[species,'Hill_notation']
        if info_key == 'mass':
            return self.species_info.loc[species,info_key]
        if info_key == 'COH':
            return list(self.species_info.loc[species,['C','O','H']])
        if info_key in ['C','O','H']:
            return self.species_info.loc[species,info_key]
        if info_key == 'c' or info_key == 'color':
            return self.species_info.loc[species,'color']
        if info_key == 'label':
            return self.species_info.loc[species,'mathtext_name']
        
    def get_H2(self,VMR_dict): # get H2 abundance as the remainder of the total VMR

        VMR_wo_H2 = np.sum([VMR_i for VMR_i in VMR_dict.values()], axis=0)
        VMR_dict['H2'] = 1 - VMR_wo_H2

        return VMR_dict

    def VMR_to_MF(self,VMR_dict):
        MMW = 0.
        for species_i, VMR_i in VMR_dict.items():
            mass_i = self.read_species_info(species_i, 'mass')
            MMW += mass_i * VMR_i

        # Convert to mass-fractions using mass-ratio
        mass_fractions = {'MMW': MMW * np.ones(self.n_atm_layers)}
        for species_i, VMR_i in VMR_dict.items():            
            species_pRT_i = self.read_species_info(species_i, 'pRT_name')
            mass_i = self.read_species_info(species_i, 'mass')
            mass_fractions[species_pRT_i] = VMR_i * mass_i/MMW

        return mass_fractions
        
    def equ_chemistry(self,species_pRT,params):
        species_info = pd.read_csv(os.path.join('species_info.csv'))

        def load_interp_tables():
            import h5py
            def load_hdf5(file, key):
                with h5py.File(f'{fastchem_tables}/{file}', 'r') as f:
                    return f[key][...]
                
            # Load the interpolation grid (ignore N/O)
            self.P_grid = load_hdf5('grid.hdf5', 'P')
            self.T_grid = load_hdf5('grid.hdf5', 'T')
            self.CO_grid  = load_hdf5('grid.hdf5', 'C/O')
            self.FeH_grid = load_hdf5('grid.hdf5', 'Fe/H')
            points = (self.P_grid, self.T_grid, self.CO_grid, self.FeH_grid)

            from scipy.interpolate import RegularGridInterpolator
            self.interp_tables = {}
            for species_i, hill_i in zip([*species_pRT, 'MMW'], [*self.species_hill, 'MMW']):
                key = 'MMW' if species_i=='MMW' else 'log_VMR'
                arr = load_hdf5(f'{hill_i}.hdf5', key=key)  # Load equchem abundance tables
                
                # Generate interpolation functions
                self.interp_tables[species_i] = RegularGridInterpolator(
                    values=arr[:,:,:,0,:], points=points, method='linear', # arr[P,T,C/O,N/O (const, solar value),FeH]
                    #bounds_error=False, fill_value=None
                        )        
                
        def get_VMRs(ParamTable):
            self.VMRs = {}
            self.VMRs = {'He':0.15*np.ones(self.n_atm_layers)}

            def apply_bounds(val, grid):
                val=np.array(val)
                val[val > grid.max()] = grid.max()
                val[val < grid.min()] = grid.min()
                return val

            # Update the parameters
            self.CO  = ParamTable.get('C/O')
            self.FeH = ParamTable.get('Fe/H')

            # Apply the bounds of the grid
            P = apply_bounds(self.pressure.copy(), grid=self.P_grid)
            T = apply_bounds(self.temperature.copy(), grid=self.T_grid)
            CO  = apply_bounds(np.array([self.CO]).copy(), grid=self.CO_grid)[0]
            FeH = apply_bounds(np.array([self.FeH]).copy(), grid=self.FeH_grid)[0]
            
            # Interpolate abundances
            for pRT_name_i, interp_func_i in self.interp_tables.items():

                # Interpolate the equilibrium abundances
                arr_i = interp_func_i(xi=(P, T, CO, FeH))

                if pRT_name_i != 'MMW':
                    species_i=species_info.loc[species_info["pRT_name"]==pRT_name_i]['name'].values[0]
                    self.VMRs[species_i] = 10**arr_i # log10(VMR)
                else:
                    self.MMW = arr_i.copy() # Mean-molecular weight
            return self.VMRs

        load_interp_tables()
        self.VMRs = get_VMRs(params)
        self.VMRs = self.get_H2(self.VMRs)
        self.mass_fractions = self.VMR_to_MF(self.VMRs)

        return self.mass_fractions

    def free_chemistry(self,species_pRT,params):
        VMR_He = 0.15
        VMR_wo_H2 = 0 + VMR_He  # Total VMR without H2, starting with He
        mass_fractions = {} # Create a dictionary for all used species
        C, O, H = 0, 0, 0

        for species_i in self.species_info.index:
            species_pRT_i = self.read_species_info(species_i,'pRT_name')
            mass_i = self.read_species_info(species_i, 'mass')
            COH_i  = self.read_species_info(species_i, 'COH')

            if species_i in ['H2', 'He']:
                continue
            if species_pRT_i in species_pRT:
                VMR_i = 10**(params[f'log_{species_i}'])*np.ones(self.n_atm_layers) #  use constant, vertical profile

                # Convert VMR to mass fraction using molecular mass number
                mass_fractions[species_pRT_i] = mass_i * VMR_i
                VMR_wo_H2 += VMR_i

                # Record C, O, and H bearing species for C/O and metallicity
                C += COH_i[0] * VMR_i
                O += COH_i[1] * VMR_i
                H += COH_i[2] * VMR_i

        # Add the H2 and He abundances
        mass_fractions['He'] = self.read_species_info('He', 'mass')*VMR_He
        mass_fractions['H2'] = self.read_species_info('H2', 'mass')*(1-VMR_wo_H2)
        H += self.read_species_info('H2','H')*(1-VMR_wo_H2) # Add to the H-bearing species
        
        if VMR_wo_H2.any() > 1:
            print('VMR_wo_H2 > 1. Other species are too abundant!')
            print('\n',VMR_wo_H2,"\n")

        MMW = 0 # Compute the mean molecular weight from all species
        for mass_i in mass_fractions.values():
            MMW += mass_i
        MMW *= np.ones(self.n_atm_layers)
        
        for species_pRT_i in mass_fractions.keys():
            mass_fractions[species_pRT_i] /= MMW # Turn the molecular masses into mass fractions
        mass_fractions['MMW'] = MMW # pRT requires MMW in mass fractions dictionary
        CO = C/O if np.sum(O)!=0 else np.inf
        log_CH_solar = 8.46 - 12 # Asplund et al. (2021)
        FeH = np.log10(C/H)-log_CH_solar if (C/H).any()!=0.0 else -np.inf*np.ones((len(C)))
        CO = np.nanmean(CO)
        FeH = np.nanmean(FeH)

        return mass_fractions, CO, FeH

    def make_spectrum(self, save_as=None):
        radtrans = Radtrans(
                pressures=self.pressure,
                line_species=self.species_pRT,
                rayleigh_species=['H2', 'He'],
                gas_continuum_contributors=['H2--H2', 'H2--He'],
                wavelength_boundaries=[self.wave_range[0],self.wave_range[1]], # microns (L-band)
                line_opacity_mode='lbl')
    
        
        planet_wl_cm, transit_radii_cm, _ = radtrans.calculate_transit_radii(temperatures=self.temperature,
                                                                        mass_fractions=self.mass_fractions,
                                                                        mean_molar_masses=self.MMW,
                                                                        reference_gravity=self.gravity,
                                                                        planet_radius=self.planet_radius,
                                                                        reference_pressure=0.01)
        
        transit_radii_cm = instrumental_broadening(planet_wl_cm, transit_radii_cm, resolution=1e5)
        transit_radii_um = (transit_radii_cm*1e4)*u.um
        planet_wl_um = (planet_wl_cm*1e4)*u.um

        if save_as is not None:
            output_dir = pathlib.Path(f'{self.project_path}/pRT_spectra')
            output_dir.mkdir(parents=True, exist_ok=True)
            planet_spectrum = pathlib.Path(f'{output_dir}/{save_as}.fits')
            tbl = QTable([planet_wl_um, transit_radii_um], names=['wavelength', 'flux'])
            tbl.write(planet_spectrum, overwrite=True)

        return planet_wl_um, transit_radii_um