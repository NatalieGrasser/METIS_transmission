import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import os
from astropy.io import fits
import astropy.constants as const
from utils import *
from scipy.interpolate import interp1d

class CrossCorr:

    def __init__(self, system_obj, planet_wl_obs_range, transit_flux_array, transit_errors_array, plot=False):

        inherit_attributes = ['data_path','delta_lambda','phase_obs','in_transit',
                              'planet_wl_um','Kp','rv_transit','phase_transit']

        for attr in inherit_attributes:  # list of attributes to pass down
            setattr(self, attr, getattr(system_obj, attr))

        num_exp = len(transit_flux_array)

        def remove_continuum(wave, spec, err, poly_order=3, get_cont=False):
            wave = np.asarray(wave.value if isinstance(wave, u.Quantity) else wave)
            spec = np.asarray(spec.value if isinstance(spec, u.Quantity) else spec)
            err = np.asarray(err.value if isinstance(err, u.Quantity) else err)
            spec_2d = np.atleast_2d(spec)
            err_2d = np.atleast_2d(err)

            flux_contrem, err_contrem, continua = [], [], []
            for s,e in zip(spec_2d,err_2d):
                continuum = np.poly1d(np.polyfit(wave, s, poly_order))(wave)
                continua.append(continuum)
                flux_contrem.append(s / continuum)
                err_contrem.append(e / continuum)

            flux_contrem = np.array(flux_contrem)
            err_contrem = np.array(err_contrem)
            continua = np.array(continua)

            if spec.ndim == 1:
                flux_contrem, err_contrem, continua = flux_contrem[0], err_contrem[0], continua[0]

            return (flux_contrem, err_contrem, continua) if get_cont else (flux_contrem,err_contrem)

        flux_contrem, err_contrem, cont = remove_continuum(planet_wl_obs_range,transit_flux_array,transit_errors_array,get_cont=True)
        
        if plot:
            plt.figure(figsize=(5,2))
            im=plt.imshow(flux_contrem,aspect='auto')
            plt.colorbar(im)

        flux_off_transit_avg = np.zeros_like(flux_contrem)
        off_transit_sum = np.zeros_like(flux_contrem[0])
        for exp in range(num_exp):
            if exp in in_transit:
                pass
            else:
                fl=flux_contrem[exp]
                off_transit_sum +=fl
        flux_off_transit_avg = off_transit_sum/np.nanmedian(off_transit_sum)

        flux_avgrem=np.zeros_like(flux_contrem)
        for exp in range(num_exp):
            flux_avgrem[exp] = flux_contrem[exp]/flux_off_transit_avg

        if plot: 
            plt.figure(figsize=(5,2))
            im=plt.imshow(flux_avgrem,aspect='auto')
            plt.xlim(0,1000)
            plt.colorbar(im)

        def sysrem(flux,fluxerr,num_modes=4,plot=False):
            #flux= (flux.T/np.nanmean(flux, axis=1)).T # normalize first, axis=1 is mean over each exposure
            flux= (flux.T-np.nanmean(flux, axis=1)).T # shift mean to zero

            max_iterations_per_mode=1000
            a = np.ones(flux.shape[0]) # number of exposures
            residuals=np.copy(flux)
            for i in range(num_modes):
                correction = np.zeros_like(residuals)
                for j in range(max_iterations_per_mode):
                    prev_correction = correction
                    c = np.nansum(residuals.T*a/fluxerr.T**2,axis=1)/np.nansum(a**2/fluxerr.T**2,axis=1)
                    a = np.nansum(c*residuals/fluxerr**2,axis=1)/np.nansum(c**2/fluxerr**2, axis=1)
                    correction = np.dot(a[:, np.newaxis], c[np.newaxis, :])
                    fractional_dcorr = np.nansum(np.abs(correction - prev_correction))/(np.nansum(np.abs(prev_correction))+1e-5)
                    if j > 1 and fractional_dcorr < 1e-3:
                        break
                residuals -= correction

            if plot:
                cmap="pink"
                fig,ax=plt.subplots(2,1,figsize=(5,3),sharex=True)
                im0=ax[0].imshow(flux, aspect='auto', interpolation='none',cmap=cmap)
                ax[0].set_title("Before SysRem")
                im1=ax[1].imshow(residuals, aspect='auto', interpolation='none',cmap=cmap)
                ax[1].set_title("After SysRem")
                ax[1].set_xlabel('Wavelength bins')
                for i in range(2):
                    ax[i].set_ylabel('Exposure #')
                fig.tight_layout()

            return residuals    

        temp_err = np.ones_like(transit_flux_array)*np.nanmedian(flux_avgrem)*0.01
        flux_sysrem = sysrem(flux_avgrem,temp_err,num_modes=1,plot=True)

        if plot:
            plt.figure(figsize=(5,2))
            im=plt.imshow(flux_sysrem,aspect='auto')
            plt.xlim(0,1000)
            plt.colorbar(im)

        RVs=np.arange(-200,200,1) # km/s
        beta=1.0-RVs/const.c.to('km/s').value

        template_wl = planet_wl_um 
        template_flux = np.ones_like(delta_lambda)-delta_lambda 

        wl=planet_wl_obs_range
        fl=np.copy(flux_sysrem) # get all exposures
        flerr=np.copy(temp_err)
        flerr[flerr==0] = np.inf #every value with err=inf is not included in cc because of divide by err (weighting)
        nans = np.isnan(fl)+np.isnan(flerr)
        fl[nans] = 0
        flerr[nans] = np.inf
        wl_shift = wl[:, np.newaxis]*beta[np.newaxis, :]
        template_shift=np.interp(wl_shift,template_wl,template_flux)
        # do same for template as for spectra, remove cont with polydeg=3
        template_shift=np.array([remove_continuum(wav,temp) for wav,temp in zip(wl_shift.T,template_shift.T)])
        template_shift = template_shift - np.mean(template_shift, axis=0) # simplified continuum removal
        CCF = (fl/flerr**2).dot(template_shift.T) # error weighted

        if plot:
            xmin, xmax = -30,30
            plt.figure(figsize=(5,3))
            im = plt.imshow(CCF,origin="lower",aspect='auto',
                    extent=[np.min(RVs),np.max(RVs),np.min(phase_obs),np.max(phase_obs)])
            plt.colorbar(im,label='CCF')
            plt.plot(rv_transit,phase_transit,c='white',linestyle='dashed',alpha=0.5,lw=2)
            plt.hlines(phase_transit[-1],xmin, xmax,color='white',linestyle='dashdot',alpha=0.8,lw=2)
            plt.hlines(phase_transit[0],xmin, xmax,color='white',linestyle='dashdot',alpha=0.8,lw=2)
            plt.xlabel('Radial velocity [km/s]')
            plt.ylabel('Phase')
            plt.xlim(xmin, xmax)

        Kp_list = np.arange(0,300,1)#km/s
        vsys_list=np.arange(-100,101,1) #km/s system velocities we want to shift to

        def shift_and_sum(ccf,vsys_list,rv_list):
            # shift and sum everything in the transit to combine the signal
            # this will be one row of the ccf map for one Kp value
            ccf_rest=np.zeros((num_exp,len(vsys_list)))
            tot_ccf =np.zeros(vsys_list.shape) # or tot_ccf=0
            for exp in in_transit:
                x = RVs
                y = ccf[exp] 
                interp = interp1d(x, y, assume_sorted=True,bounds_error=False)(rv_list[exp]+vsys_list) 
                tot_ccf = np.nansum(np.stack((tot_ccf,interp)),axis=0)
            return tot_ccf    

        def get_velocity_map(CCFs):
            CCF_tot = np.zeros((len(Kp_list),len(vsys_list)))
            for ikp in range(len(Kp_list)):
                rv_list = Kp_list[ikp]*np.sin(2*np.pi*phase_obs) # calc planet rv for different Kp values
                CCF_tot[ikp] = shift_and_sum(CCFs,vsys_list,rv_list)
            return CCF_tot    

        Kp_vsys_map = get_velocity_map(CCF)
        noise = np.std(Kp_vsys_map[:,np.abs(vsys_list)>40],axis=1)[:, None]

        # mask out regions close to the expected values (small rvs close to 0), because they create a larger std
        Kp_vsys_map /= noise # normalize along rv axis to get ccf map in snr units
        maxlocid=np.where(Kp_vsys_map==np.nanmax(Kp_vsys_map))

        # S/N at expected Kp and vsys
        vsys0 = vsys_list[find_nearest(vsys_list,value=0)]
        Kp0 = Kp_list[find_nearest(Kp_list,value=Kp.value)]
        SNR_planet = Kp_vsys_map[Kp0,vsys0]

        if plot:
            plt.figure(figsize=(5,3))
            im=plt.imshow(Kp_vsys_map,aspect='auto',origin='lower',
                        extent=[np.min(vsys_list),np.max(vsys_list),np.min(Kp_list),np.max(Kp_list)])
            plt.scatter(vsys0,Kp0,marker='x',c='white',s=20)
            plt.xlabel(r'$\Delta v_{\mathrm{sys}}$ [km/s]')
            plt.ylabel(r'K$_{\mathrm{p}}$')
            plt.colorbar(im,label='S/N')