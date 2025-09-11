import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from petitRADTRANS import physical_constants as cst
import os
os.environ['OMP_NUM_THREADS'] = '1' # to avoid using too many CPUs, important for MPI
from mpi4py import MPI 
comm = MPI.COMM_WORLD # important for MPI
rank = comm.Get_rank() # important for MPI
import matplotlib
matplotlib.use('Agg') # disable interactive plotting

from system import System
from METIS import METIS
from crosscorr import CrossCorr
from utils import *

project_name = 'test'
chemistry = 'free' # / equ / Sorg20X
species_names = ['H2','He','H2O','CH4']

observation_properties = {'exptime_per_frame': 200*u.s,
                         'METIS_wave_range_um': [3,3.3],  # 0.3um wide
                         'pRT_wave_range_um': [2.8,3.7],} # wider range for cross-correlation

system_properties = {'planet_radius': 1.0 * cst.r_jup_mean, # 0.23* cst.r_jup_mean
                    'R_star': 0.5 * u.R_sun, # stellar radius
                    'd_star': 38 * u.pc, # distance to K2-18
                    'period': 10.0*u.day, #32.9*u.day real
                    'inclination': 89.57*u.deg,
                    'sma': 0.15*u.au,
                    'e': 0.0,
                    'transit_duration': 2.67*u.h}

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

parameters = {**observation_properties, 
              **system_properties, 
              **atmosphere_properties, 
              **chemsitry_properties}
parameters['project_path'] = os.path.join(os.getcwd(),project_name)

system_obj = System(parameters,plot=True)

# for the future, center at ~3.30 µm (≈ 3.15–3.45 µm) for DMS, CH4, H2O
# worth including for L-band: H₂O, CH₄, NH₃, C₂H₂, C₂H₄, DMS
wavelength_range = parameters['METIS_wave_range_um']  # 0.3um
wave_width = 0.05 # determined wavelength range per simulation: 0.04674999999999985
n_wave_orders = (wavelength_range[-1]-wavelength_range[0])/wave_width
central_waves = np.arange(wavelength_range[0]+wave_width/2,wavelength_range[-1],wave_width)
transit_flux_array = system_obj.get_transit_array(wavelength_range,plot=True)

central_waves = [central_waves[0]] # for testing

plot_exp = 0
plot_order= 0
wl_obs, fl_obs, err_obs = [],[],[]
for i,central_wave in enumerate(central_waves):
    metis = METIS(central_wave,system_obj)
    transit, dark, flat, sky = metis.observe_transit_calib(system_obj.transit_flux_array,f'order{i}',plot_exp=plot_exp)
    wl, fl, err = metis.rectify_calibrate_extract(transit, dark, flat, sky, plot_exp=plot_exp)
    wl_obs.append(wl)
    fl_obs.append(fl)
    err_obs.append(err)

# reshape to (num_exp, num_orders, num_wave) makes more sense
wl_obs, fl_obs, err_obs = [np.array(var) for var in [wl_obs, fl_obs, err_obs]]
num_orders, num_exp, num_wave = fl_obs.shape
fl_obs, err_obs = [var.reshape((num_exp, num_orders, num_wave)) for var in [fl_obs, err_obs]]

CrossCorr(system_obj, wl_obs, fl_obs, err_obs, plot_order=plot_order) # cc with input
for species_i in ['H2O','CH4']:
    CrossCorr(system_obj, wl_obs, fl_obs, err_obs, plot_order=plot_order, template=species_i)

