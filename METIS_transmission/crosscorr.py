import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import os
from astropy.io import fits
import astropy.constants as const
from utils import *
from scipy.interpolate import interp1d
import pathlib
from astropy.table import QTable
from pRT_model import *

class CrossCorr:

    def __init__(self, system_obj, wave, flux, err, outlier_sigma=3, 
                 systrem_iter=3, plot_order=None, template=None):

        inherit_attributes = ['project_path','delta_lambda','phase_obs','in_transit',
                              'planet_wl_um','Kp','rv_transit','phase_transit',
                              'planet_wl_obs_range','parameters']

        for attr in inherit_attributes:  # list of attributes to pass down
            setattr(self, attr, getattr(system_obj, attr))

        n_exp, n_orders, n_wl  = flux.shape
        self.figs_dir = pathlib.Path(f'{self.project_path}/figures/crosscorr')
        self.figs_dir.mkdir(parents=True, exist_ok=True)

        def mask_absorption_emission(spec,specerr,lowerlim=0.7,upperlim=1.2):
            flux_out = np.copy(spec)
            fluxerr_out=np.copy(specerr)
            for exp in range(n_exp):
                for order in range(n_orders):
                    fl=flux_out[exp,order]
                    flerr=fluxerr_out[exp,order]
                    flux_norm = fl/np.nanmedian(fl) #normalized spectrum
                    goodpix = ((flux_norm>lowerlim)&(flux_norm<upperlim))
                    flux_out[exp,order][~goodpix] = np.nan
                    fluxerr_out[exp,order][~goodpix] = np.inf            
            return flux_out,fluxerr_out

        def remove_continuum(wave, spec, err=None, poly_order=3, get_cont=False):

            return_only_flux = False
            if err is None:
                return_only_flux = True
                err = np.ones_like(spec)

            wave = np.asarray(wave.value if isinstance(wave, u.Quantity) else wave, dtype=float)
            spec = np.asarray(spec.value if isinstance(spec, u.Quantity) else spec, dtype=float)
            err  = np.asarray(err.value  if isinstance(err, u.Quantity)  else err,  dtype=float)
            nans = np.isnan(spec)

            if spec.ndim == 1: # for template
                continuum = np.poly1d(np.polyfit(wave[~nans],spec[~nans],poly_order))(wave)
                flux_contrem = spec / continuum
                err_contrem = err / continuum
                if return_only_flux:
                    return flux_contrem
                return (flux_contrem, err_contrem, continuum) if get_cont else (flux_contrem, err_contrem)

            elif spec.ndim == 3: 
                n_exp, n_orders, n_wave = spec.shape
                flux_contrem = np.full(spec.shape,np.nan)
                err_contrem=np.full(spec.shape,np.inf)
                continua=np.full(spec.shape,np.nan)
                for exp in range(n_exp):
                    for order in range(n_orders):
                        fl=spec[exp,order]
                        flerr=err[exp,order]
                        wl=wave[order]
                        nans = np.isnan(fl)
                        continuum_model = np.poly1d(np.polyfit(wl[~nans],fl[~nans],poly_order))
                        continuum = continuum_model(wl)
                        flux_contrem[exp,order][~nans] = fl[~nans]/continuum[~nans]
                        err_contrem[exp,order][~nans] = flerr[~nans]/continuum[~nans]
                        continua[exp,order] = continuum        

                return (flux_contrem, err_contrem, continua) if get_cont else (flux_contrem, err_contrem)

        def mask_outliers(spec,specerr,sigma=outlier_sigma):
            flux_out = np.copy(spec)
            err_out=np.copy(specerr)
            for exp in range(n_exp):
                for order in range(n_orders):
                    fl=flux_out[exp,order]
                    std = np.nanstd(fl)
                    mean = np.nanmedian(fl)
                    outliers = (np.abs(fl-mean)/std)>sigma
                    flux_out[exp,order][outliers] = np.nan
                    err_out[exp,order][outliers] = np.inf
            return flux_out,err_out

        flux_contrem, err_contrem = remove_continuum(wave, flux, err)
        # mask tellurics after removing continuum bc flux slope is too strong
        flux_mask0,err_mask0 = mask_absorption_emission(flux_contrem,err_contrem,lowerlim=0.7,upperlim=np.nanmax(flux_contrem))
        flux_mask1,err_mask1 = mask_outliers(flux_mask0,err_mask0,sigma=outlier_sigma)

        flux_off_transit_avg = np.zeros_like(flux[0,:])
        for order in range(n_orders):
            off_transit_sum = np.zeros_like(flux[0,0])
            for exp in range(n_exp):
                if exp in self.in_transit:
                    pass
                else:
                    fl=flux_mask1[exp,order]
                    off_transit_sum +=fl
            off_transit_avg= off_transit_sum/np.nanmedian(off_transit_sum)
            flux_off_transit_avg[order]=off_transit_avg

        flux_avgrem=np.zeros_like(flux_contrem)
        fluxerr_avgrem=np.zeros_like(flux_contrem)
        for order in range(n_orders):
            for exp in range(n_exp):
                flux_avgrem[exp,order] = flux_mask1[exp,order]/flux_off_transit_avg[order]
                fluxerr_avgrem[exp,order] = err_mask1[exp,order]/flux_off_transit_avg[order]

        def sysrem(flux,fluxerr,num_modes=systrem_iter):

            #flux= (flux.T-np.nanmean(flux, axis=1)).T # shift mean to zero
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

            return residuals    

        flux_sysrem=np.zeros((n_exp,n_orders,n_wl))
        for i in range(n_orders): 
            fl=flux_avgrem[:,i]
            flerr=fluxerr_avgrem[:,i]
            fl_new=sysrem(fl,flerr)
            flux_sysrem[:,i]=fl_new

        # one last outlier flagging before crosscorr
        flux_mask2,fluxerr_mask2 =mask_outliers(flux_sysrem,fluxerr_avgrem,sigma=3)

        self.RVs=np.arange(-200,200,1) # km/s
        beta=1.0-self.RVs/const.c.to('km/s').value

        if template is None: # use input
            template_wl = self.planet_wl_um 
            template_flux = np.ones_like(self.delta_lambda)-self.delta_lambda 
            self.template_name = 'input'
        else:
            # single-species template, must be name of species
            template_path = pathlib.Path(f'{self.project_path}/pRT_spectra/{template}.fits')

            if template_path.exists():
                tbl = QTable.read(template_path)
                template_wl = tbl['wavelength'] # in um
                transit_radii_um = tbl['flux'] # in um

            else:
                params2 = self.parameters.copy()
                params2['chemistry'] = 'free'
                params2['species_names'] = ['H2','He', template]
                params2[f'log_{template}'] = -2
                template_wl, transit_radii_um = pRT_spectrum(params2).make_spectrum(save_as=template)

            transit_radii_cm = (transit_radii_um.to(u.cm))
            delta_lambda = ((transit_radii_cm / system_obj.R_star.to(u.cm))**2).value
            template_flux = np.ones_like(delta_lambda)-delta_lambda
            self.template_name = template

        self.CCF = np.zeros((n_exp,n_orders,len(self.RVs)))
        for order in range(n_orders):
            wl=wave[order] # wavelengths are same per exposure
            fl=np.copy(flux_mask2[:,order]) # get all exposures per order
            flerr=np.copy(fluxerr_mask2[:,order])
            bad_pixels = ~np.isfinite(fl) | ~np.isfinite(flerr)
            fl[bad_pixels] = 0
            flerr[bad_pixels] = np.inf
            wl_shift=wl[:, np.newaxis]*beta[np.newaxis, :]
            template_shift=interp1d(template_wl,template_flux)(wl_shift) #interpolate template onto shifted wl-grid
            # do same for template as for spectra, remove cont with polydeg=3
            template_shift=np.array([remove_continuum(wav,temp) for wav,temp in zip(wl_shift.T,template_shift.T)])
            template_shift = template_shift - np.mean(template_shift, axis=0) # simplified continuum removal
            self.CCF[:,order] = (fl/flerr**2).dot(template_shift.T) #error weighted

        self.Kp_list = np.arange(0,300,1)#km/s
        self.vsys_list=np.arange(-100,101,1) #km/s system velocities we want to shift to

        def shift_and_sum(ccf,vsys_list,rv_list):
            # shift and sum everything in the transit to combine the signal
            # this will be one row of the ccf map for one Kp value
            tot_ccf =np.zeros(vsys_list.shape) # or tot_ccf=0
            for exp in self.in_transit:
                x = self.RVs
                y = ccf[exp]
                interp=interp1d(x, y, assume_sorted=True,bounds_error=False)(rv_list[exp]+vsys_list) 
                tot_ccf=np.nansum(np.stack((tot_ccf,interp)),axis=0)
            return tot_ccf    
            
        def get_Kp_vsys_map(CCFs):
            CCF_tot = np.zeros((len(self.Kp_list),len(self.vsys_list)))
            for ikp in range(len(self.Kp_list)):
                rv_list = self.Kp_list[ikp]*np.sin(2*np.pi*self.phase_obs) #calc planet rv for different Kp values
                CCF_tot[ikp] = shift_and_sum(CCFs,self.vsys_list,rv_list)
            return CCF_tot    

        CCF_sum=np.sum(self.CCF,axis=1) # sum CCF over all orders
        Kp_vsys_map = get_Kp_vsys_map(CCF_sum)
        noise=np.std(Kp_vsys_map[:,np.abs(self.vsys_list)>30],axis=1)[:, None]
        self.Kp_vsys_map = Kp_vsys_map/noise #normalize along rv axis to get ccf map in snr units

        # S/N at expected self.Kp and vsys
        Kp_idx = find_nearest(self.Kp_list,value=self.Kp.value)
        vsys_idx = find_nearest(self.vsys_list,value=0)
        self.vsys0 = self.vsys_list[vsys_idx]
        self.Kp0 = self.Kp_list[Kp_idx]
        self.SNR_planet = self.Kp_vsys_map[Kp_idx,vsys_idx]

        if plot_order is not None:
            self.plot_calib_steps(wave,flux,flux_contrem,flux_avgrem,flux_sysrem,plot_order)
            self.plot_CCF(plot_order=0)
            self.plot_Kp_vsys()

    def plot_calib_steps(self,wl,fl_orig,fl_contrem,fl_avgrem,fl_sysrem,plot_order):

        fig,(ax1,ax2,ax3,ax4)=plt.subplots(4,1,figsize=(7,6),dpi=200,sharex=True)
        extent=[np.min(wl),np.max(wl),0,fl_orig.shape[0]]

        def imshow(ax,arr,title,vmin=0.5,vmax=99.5):
            slice_ = arr[:,plot_order,:]
            im=ax.imshow(slice_,aspect='auto', origin ='lower',extent=extent,
                        vmin=np.nanpercentile(slice_, vmin), vmax=np.nanpercentile(slice_, vmax))
            fig.colorbar(im,ax=ax)
            ax.set_title(title)

        imshow(ax1,fl_orig,f'Original flux, order {plot_order}')
        imshow(ax2,fl_contrem,'Continuum removed')
        imshow(ax3,fl_avgrem,'Average off-transit removed')
        imshow(ax4,fl_sysrem,'After Sysrem')

        ax4.set_xlabel('Wavelength [um]')
        fig.tight_layout()
        fig.savefig(f'{self.figs_dir}/calib_steps.pdf', bbox_inches='tight')
        plt.close()

    def plot_CCF(self,plot_order):
        xmin, xmax = -30,30
        fig = plt.figure(figsize=(5,3),dpi=150)
        im = plt.imshow(self.CCF[:,plot_order,:],origin="lower",aspect='auto',
                extent=[np.min(self.RVs),np.max(self.RVs),np.min(self.phase_obs),np.max(self.phase_obs)])
        plt.colorbar(im,label='CCF')
        plt.plot(self.rv_transit,self.phase_transit,c='white',linestyle='dashed',alpha=0.5,lw=2)
        plt.hlines(self.phase_transit[-1],xmin, xmax,color='white',linestyle='dashdot',alpha=0.8,lw=2)
        plt.hlines(self.phase_transit[0],xmin, xmax,color='white',linestyle='dashdot',alpha=0.8,lw=2)
        plt.xlabel('Radial velocity [km/s]')
        plt.ylabel('Phase')
        plt.xlim(xmin, xmax)
        fig.savefig(f'{self.figs_dir}/CCF_{self.template_name}.pdf', bbox_inches='tight')
        plt.close()
    
    def plot_Kp_vsys(self):
        fig = plt.figure(figsize=(5,3),dpi=150)
        im=plt.imshow(self.Kp_vsys_map,aspect='auto',origin='lower',
                    extent=[np.min(self.vsys_list),np.max(self.vsys_list),np.min(self.Kp_list),np.max(self.Kp_list)])
        plt.scatter(self.vsys0,self.Kp0,marker='x',c='hotpink',s=20,label=fr"S/N$_{{\mathrm{{planet}}}}$ = {self.SNR_planet:.1f}")
        maxlocid=np.where(self.Kp_vsys_map==np.nanmax(self.Kp_vsys_map))
        SNR_max = self.Kp_vsys_map[maxlocid[0][0],maxlocid[1][0]]
        #plt.scatter(self.vsys_list[maxlocid[1][0]],self.Kp_list[maxlocid[0][0]],marker='x',c='r',s=20,label=fr"S/N$_{{\mathrm{{max}}}}$ = {SNR_max:.1f}")
        plt.xlabel(r'$\Delta v_{\mathrm{sys}}$ [km/s]')
        plt.ylabel(r'$K_{\mathrm{p}}$ [km/s]')
        plt.colorbar(im,label='S/N')
        plt.legend()
        fig.savefig(f'{self.figs_dir}/Kp_vsys_{self.template_name}.pdf', bbox_inches='tight')
        plt.close()