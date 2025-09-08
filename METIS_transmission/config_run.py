import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import os
from system import System
from METIS import METIS
from crosscorr import CrossCorr

project_name = 'test'
exptime_per_frame = 200*u.s
system_obj = System(project_name,exptime_per_frame)

wavelength_range = [3,3.3] # 0.3um
wave_width = 0.05 # determined wavelength range per simulation: 0.04674999999999985
n_wave_parts = (wavelength_range[-1]-wavelength_range[0])/wave_width
central_waves = np.arange(wavelength_range[0]+wave_width/2,wavelength_range[-1],wave_width)
transit_flux_array = system_obj.get_transit_array(wavelength_range)


central_waves = central_waves[0] # for testing

wl_obs, fl_obs, err_obs = [],[],[]
for central_wave in central_waves:
    metis = METIS(central_wave,system_obj)
    transit, dark, flat, sky = metis.observe_transit_calib(system_obj.transit_flux_array)
    wl, fl, err = metis.rectify_calibrate_extract(transit, dark, flat, sky)
    wl_obs.append(wl)
    fl_obs.append(wl)
    err_obs.append(wl)

wl_obs, fl_obs, err_obs = [np.array(var) for var in [wl_obs, fl_obs, err_obs]]
CrossCorr(system_obj, wl_obs, fl_obs, err_obs)
