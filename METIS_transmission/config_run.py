from petitRADTRANS.config import petitradtrans_config_parser
#petitradtrans_config_parser.set_input_data_path('/net/lem/data2/pRT3_formatted')
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from petitRADTRANS import physical_constants as cst
import builtins as _builtins
import os
os.environ['OMP_NUM_THREADS'] = '1' # to avoid using too many CPUs, important for MPI
from system import System
from METIS import METIS
from crosscorr import CrossCorr
from utils import *

def init_simulation(project_name='Sorg20X',chemistry=None):

    project_name = project_name
    if 'Sorg' in project_name:
        chemistry = project_name 
    elif chemistry is not None:
        chemistry = chemistry # free / equ / Sorg1X / Sorg20X
    else:
        chemistry = 'free'

    if 'Sorg' in project_name:
        species_names = ['H2','He','H2O','CH4','C2H6','CO2','NH3','C2H4','CO','H2S','C2H6S']
    elif project_name=='test':
        species_names = ['H2','He','H2O','CH4']
    
    observation_properties = {'exptime_per_frame': 200*u.s,
                            'METIS_wave_range_um': [3.15,3.45],  # 0.3um wide
                            'pRT_wave_range_um': [3.0,3.6],} # wider range for cross-correlation

    system_properties = {'planet_radius': 0.23* cst.r_jup_mean,
                        'R_star': 0.5 * u.R_sun, # stellar radius
                        'd_star': 38 * u.pc, # distance to K2-18
                        'period': 32.9*u.day,
                        'inclination': 89.57*u.deg,
                        'sma': 0.15*u.au,
                        'e': 0.0,
                        'transit_duration': 2.67*u.h,
                        #'vsini': 5*u.km/u.s,
                        }

    atmosphere_properties = {'pressure': np.logspace(-6, 2, 100),
                            'temperature': 300,
                            'log_g': 3}

    chemsitry_properties = {'species_names': species_names,
                            'chemistry' : chemistry}

    if chemistry=='free':
        chemsitry_properties['log_H2O'] = -2
        chemsitry_properties['log_CH4'] = -2

    elif chemistry=='equ':
        chemsitry_properties['Fe/H'] = 0.0
        chemsitry_properties['C/O'] = 0.59

    elif chemistry in ['Sorg1X','Sorg20X']: # Hycean model with 1X or 20X organic sulfur
        tab = PSG_input(chemistry).table
        atmosphere_properties['pressure'] = tab['Pressure'].to_numpy()
        atmosphere_properties['temperature'] = tab['Temperature'].to_numpy()
        for species_i in species_names:
            chemsitry_properties[species_i] =  tab[species_i].to_numpy()

    if project_name=='test': # had different setup
        observation_properties['METIS_wave_range_um'] = [3,3.3]
        observation_properties['pRT_wave_range_um'] =  [2.8,3.7]
        system_properties['planet_radius'] = 1.0 * cst.r_jup_mean
        system_properties['period'] = 10.0*u.day # for larger Kp

    parameters = {**observation_properties, 
                **system_properties, 
                **atmosphere_properties, 
                **chemsitry_properties}
    parameters['project_path'] = os.path.join(os.getcwd(),project_name)
    system_obj = System(parameters,plot=True)

    return system_obj

def run_simulation(system_obj):

    parameters = system_obj.parameters
    species_names = system_obj.parameters['species_names']

    # for the future, center at ~3.30 µm (≈ 3.15–3.45 µm) for DMS, CH4, H2O
    wavelength_range = parameters['METIS_wave_range_um']  # 0.3um
    wave_width = 0.05 # determined wavelength range per simulation: 0.04674999999999985
    n_wave_orders = int((wavelength_range[-1]-wavelength_range[0])/wave_width)
    central_waves = np.arange(wavelength_range[0]+wave_width/2,wavelength_range[-1],wave_width)
    transit_flux_array = system_obj.get_transit_array(wavelength_range,plot=True)
    print(f'\n{n_wave_orders} orders, range {wavelength_range[0]}-{wavelength_range[1]}um \n')
    #central_waves = [central_waves[0]] # for testing

    plot_exp = 0
    plot_order= 0
    wl_obs, fl_obs, err_obs = [],[],[]
    for i,central_wave in enumerate(central_waves):
        print(f'\n ***** Simulating order {i}, central wavelength {central_wave}um ***** \n')
        metis = METIS(central_wave,system_obj,order=i)
        wl, fl, err = metis.get_observations(plot_exp=plot_exp)
        wl_obs.append(wl)
        fl_obs.append(fl)
        err_obs.append(err)

    # reshape to (num_exp, num_orders, num_wave) makes more sense
    wl_obs, fl_obs, err_obs = [np.array(var) for var in [wl_obs, fl_obs, err_obs]]
    num_orders, num_exp, num_wave = fl_obs.shape
    fl_obs, err_obs = [var.reshape((num_exp, num_orders, num_wave)) for var in [fl_obs, err_obs]]

    print(f'\n ***** Cross-correlating... ***** \n')
    CrossCorr(system_obj, wl_obs, fl_obs, err_obs, plot_order=plot_order) # cc with input

    line_species = [s for s in species_names if s not in ('H2', 'He')]
    for species_i in line_species:
        CrossCorr(system_obj, wl_obs, fl_obs, err_obs, plot_order=plot_order, template=species_i)

if __name__ == '__main__':
    from datetime import datetime
    import sys, shutil, os
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    import matplotlib
    matplotlib.use('Agg') # disable interactive plotting

    class SuppressOutput:
        def __enter__(self):
            if rank != 0:
                # Redirect both stdout and stderr to devnull
                self._stdout = os.dup(1)
                self._stderr = os.dup(2)
                self.devnull = os.open(os.devnull, os.O_WRONLY)
                os.dup2(self.devnull, 1)
                os.dup2(self.devnull, 2)
        def __exit__(self, exc_type, exc_val, exc_tb):
            if rank != 0:
                # Restore original stdout and stderr
                os.dup2(self._stdout, 1)
                os.dup2(self._stderr, 2)
                os.close(self.devnull)

    with SuppressOutput():

        # pass configuration as command line argument
        # example: config_run.py Sorg20X
        project_name = sys.argv[1] # test / Sorg20X / Sorg1X
        system_obj = init_simulation(project_name)
        print(f'\n *********** RUNNING SIMULATION FOR {project_name} *********** \n')
        
        if rank == 0:  # only first process saves a copy of config file
            if not any(fname.startswith("config_run_") for fname in os.listdir(system_obj.project_path)):
                timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M")
                shutil.copy(__file__, os.path.join(system_obj.project_path, f"config_run_{timestamp}.py"))

        run_simulation(system_obj)