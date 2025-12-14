import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from petitRADTRANS import physical_constants as cst
import builtins as _builtins
import pathlib
import os
os.environ['OMP_NUM_THREADS'] = '1' # to avoid using too many CPUs, important for MPI
from system import System
from METIS import METIS
from crosscorr import CrossCorr
from pRT_model import *
from PyAstronomy.pyasl import helcorr
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
from utils import *

def init_simulation(project_name='Sorg20X',chemistry=None,test_on_input=False):

    project_name = project_name
    if 'Sorg' in project_name:
        chemistry = project_name 
    elif chemistry is not None:
        chemistry = chemistry # free / equ / Sorg1X / Sorg20X
    else:
        chemistry = 'free'

    if 'Sorg' in project_name:
        species_names = ['H2','He','H2O','CH4','C2H6','CO2','NH3','C2H4','H2S',
                         'H2CO','CO','C2H2','OCS','SO2','SO']
    elif project_name=='test':
        species_names = ['H2','He','H2O','CH4']
    
    observation_properties = {#'exptime_per_frame': 800*u.s,
                            'num_exp_in_transit': 5,
                            'num_overhead': 10, 
                            'METIS_wave_range_um': [3.05,3.35],  # 0.3um wide
                            'pRT_wave_range_um': [2.8,3.6],
                            'n_transits': 3
                            } # wider range for cross-correlation

    system_properties = {'planet_radius': 0.23* cst.r_jup_mean,
                        'R_star': 0.5 * u.R_sun, # stellar radius
                        'd_star': 38 * u.pc, # distance to K2-18
                        #'d_star': 10 * u.pc, # testing!!
                        'period': 32.9*u.day,
                        'inclination': 89.57*u.deg,
                        'sma': 0.15*u.au,
                        'e': 0.0,
                        'transit_duration': 2.67*u.h,
                        #'vsini': 5*u.km/u.s,
                        'vsys': 0.328, # simbad
                        'ra':"11h30m14.5177445592s", # simbad
                        'dec': "07d35m18.255348492s",
                        'JD': 2456836.17187 # Howard 2025 https://exoplanetarchive.ipac.caltech.edu/overview/k2-18b
                        }

    atmosphere_properties = {'pressure': np.logspace(-6, 0, 100),
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

    # model from https://arxiv.org/pdf/2403.14805
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

    lat  = Angle("-24d35m00s").degree # ELT @ Cerro Armazones
    lon  = Angle("-70d11m00s").degree 
    coords = SkyCoord(ra=system_properties['ra'], dec=system_properties['dec'], frame='icrs')
    vbary, _ = helcorr(obs_long=lon, obs_lat=lat, obs_alt=3040, 
                                ra2000=coords.ra.value,dec2000=coords.dec.value,
                                jd=system_properties['JD'])
    system_properties['vbary'] = vbary
    print('Barycentric velocity:', np.round(vbary,decimals=2),'km/s')

    parameters = {'project_name':project_name,
                **observation_properties, 
                **system_properties, 
                **atmosphere_properties, 
                **chemsitry_properties}
    parameters['project_path'] = os.path.join(os.getcwd(),project_name)
    parameters['test_on_input'] = test_on_input
    system_obj = System(parameters,plot=True)

    return system_obj

def run_simulation(system_obj):

    parameters = system_obj.parameters
    project_name = parameters['project_name']
    species_names = parameters['species_names']
    test_on_input = parameters['test_on_input']
    n_transits = parameters['n_transits']

    # for the future, center at ~3.30 µm (≈ 3.15–3.45 µm) for DMS, CH4, H2O
    wavelength_range = parameters['METIS_wave_range_um']  # 0.3um
    wave_width = 0.05 # determined wavelength range per simulation: 0.04674999999999985
    n_wave_orders = int((wavelength_range[-1]-wavelength_range[0])/wave_width)
    central_waves = np.arange(wavelength_range[0]+wave_width/2,wavelength_range[-1],wave_width)
    transit_flux_array = system_obj.get_transit_array(wavelength_range,plot=True)
    print(f'{n_wave_orders} orders, range {wavelength_range[0]}-{wavelength_range[1]}um \n')

    if project_name=='test':
        central_waves = [central_waves[0]] # for testing
    
    #central_waves = [central_waves[0],central_waves[1]]
    #central_waves = [central_waves[0]]
    #central_waves = [central_waves[-1]]
    #central_waves = central_waves[:-1]

    n_orders = len(central_waves)
    n_exp = system_obj.num_exp

    if test_on_input:
        print('*** Testing on input data. ***')
        n_transits = 1
        system_obj.parameters['n_transits'] = 1

    plot_exp = 0
    plot_order= 0
    fl_obs = np.empty((n_transits,n_orders,n_exp),dtype=object)
    err_obs = np.empty((n_transits,n_orders,n_exp),dtype=object)
    wl_obs = np.empty((n_orders),dtype=object)
    for nt in range(n_transits):
        print(f'######## TRANSIT {nt+1} ########')
        for order,central_wave in enumerate(central_waves):
            print(f'\n ***** Simulating order {order}, central wavelength {np.round(central_wave,decimals=3)}um ***** \n')
            metis = METIS(central_wave,system_obj,order=order,n_transit=nt)
            wl, fl, err = metis.get_observations(plot_exp=plot_exp)

            if test_on_input:
                transit_input_folder = pathlib.Path(f'{system_obj.project_path}/test_input')
                transit_input_folder.mkdir(parents=True, exist_ok=True)
                transit_path_i = pathlib.Path(f'{transit_input_folder}/input_timeseries_{order}.npy')
                if transit_path_i.exists():
                    fl = np.load(transit_path_i) # save only flux, take wl & err from sim data
                else:
                    transit_flux_i = fl.copy() # init array shaped like simulated flux
                    n_exp = len(fl)
                    for exp in range(n_exp):
                        interp_flux = np.interp(wl,system_obj.planet_wl_obs_range.value,transit_flux_array[exp].value)
                        transit_flux_i[exp] = interp_flux
                    fl = transit_flux_i
                    np.save(transit_path_i, fl)
            wl_obs[order]= wl
            for exp in range(n_exp):
                fl_obs[nt,order,exp] = fl[exp]
                err_obs[nt,order,exp] = err[exp]

    print('*** Full observation-time series is ready. ***')

    # flux has shape (n_trans,n_order,n_exp,n_wl), reformat into (n_trans,n_exp,n_order,n_wl)
    # wavelength arrays may have different lengths
    
    fl_array = np.empty((n_transits, n_exp, n_orders), dtype=object)
    err_array = np.empty((n_transits, n_exp, n_orders), dtype=object)
    for nt in range(n_transits):
        for exp in range(n_exp):
            for ord in range(n_orders):
                fl_array[nt][exp][ord] = fl_obs[nt][ord][exp]  
                err_array[nt][exp][ord] = err_obs[nt][ord][exp]

    print(f'\n ***** Cross-correlating... ***** \n')
    cc_obj = CrossCorr(system_obj, wl_obs, fl_array, err_array, plot_order=0) # cc with input

    line_species = [s for s in species_names if s not in ('H2', 'He')]
    line_species = ['H2O','CH4','C2H2','C2H4','C2H6'] # detectable in noiseless telluricless input
    for species_i in line_species:
        CrossCorr(system_obj, wl_obs, fl_array, err_array, plot_order=plot_order, template=species_i)

    cc_obj.plot_ccfs()
    
    print(f'\n ***** All done. Exiting. ***** \n')

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

    with SuppressOutput(): # print only once when running parallel

        # pass configuration as command line argument
        # example: nohup mpiexec -np 20 python config_run.py Sorg20X > & output.out &
        project_name = sys.argv[1] # test / Sorg20X / Sorg1X
        test_on_input = True if len(sys.argv)>2 else False # test cross-corr on input, not METIS sim
        print(f'\n *********** RUNNING SIMULATION FOR {project_name} *********** \n')
        system_obj = init_simulation(project_name,test_on_input=test_on_input)
        
        if rank == 0:  # only first process saves a copy of config file
            if not any(fname.startswith("config_run_") for fname in os.listdir(system_obj.project_path)):
                timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M")
                shutil.copy(__file__, os.path.join(system_obj.project_path, f"config_run_{timestamp}.py"))

        run_simulation(system_obj)