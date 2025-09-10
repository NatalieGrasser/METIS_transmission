import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
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

project_name = 'test'
exptime_per_frame = 200*u.s
system_obj = System(project_name,exptime_per_frame)

wavelength_range = [3,3.3] # 0.3um
wave_width = 0.05 # determined wavelength range per simulation: 0.04674999999999985
n_wave_orders = (wavelength_range[-1]-wavelength_range[0])/wave_width
central_waves = np.arange(wavelength_range[0]+wave_width/2,wavelength_range[-1],wave_width)
transit_flux_array = system_obj.get_transit_array(wavelength_range)

central_waves = [central_waves[0]] # for testing

#system_obj.plot_stellar_spectrum()
#system_obj.plot_planet_transmission()
#system_obj.plot_transit_timeseries()

plot_exp = 10
plot_order= 0
wl_obs, fl_obs, err_obs = [],[],[]
for i,central_wave in enumerate(central_waves):
    metis = METIS(central_wave,system_obj)
    transit, dark, flat, sky = metis.observe_transit_calib(system_obj.transit_flux_array,f'part{i}',plot_exp=plot_exp)
    wl, fl, err = metis.rectify_calibrate_extract(transit, dark, flat, sky, plot_exp=plot_exp)
    wl_obs.append(wl)
    fl_obs.append(fl)
    err_obs.append(err)

# reshape to (num_exp, num_orders, num_wave) makes more sense
wl_obs, fl_obs, err_obs = [np.array(var) for var in [wl_obs, fl_obs, err_obs]]
num_orders, num_exp, num_wave = fl_obs.shape
fl_obs, err_obs = [var.reshape((num_exp, num_orders, num_wave)) for var in [fl_obs, err_obs]]

fig=plt.figure(figsize=(5,2),dpi=150)
im=plt.imshow(fl_obs[:,plot_order,:],aspect='auto')
plt.colorbar(im)
fig.savefig(f'{system_obj.data_path}/figures/flux_obs.pdf', bbox_inches='tight')
plt.close()

CrossCorr(system_obj, wl_obs, fl_obs, err_obs)
