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
                 sysrem_iter=1, plot_order=None, template=None):

        inherit_attributes = ['project_path','delta_lambda','phase_obs','in_transit',
                              'planet_wl_um','Kp','rv_transit','phase_transit',
                              'planet_wl_obs_range','parameters']

        for attr in inherit_attributes:  # list of attributes to pass down
            setattr(self, attr, getattr(system_obj, attr))

        self.vbary = self.parameters['vbary']
        self.vsys = self.parameters['vsys']
        self.n_transits = self.parameters['n_transits']
        self.noiserange = 50

        n_trans, n_exp, n_orders = flux.shape
        test_on_input = system_obj.parameters['test_on_input']
        if test_on_input:
            self.figs_dir = pathlib.Path(f'{self.project_path}/figures/CC_input_sysit{sysrem_iter}_n{self.n_transits}')
        else:
            self.figs_dir = pathlib.Path(f'{self.project_path}/figures/CC_sysit{sysrem_iter}_n{self.n_transits}')
        self.figs_dir.mkdir(parents=True, exist_ok=True)

        self.ccf_dir = pathlib.Path(f'{self.project_path}/CCFs/CC_sysit{sysrem_iter}_n{self.n_transits}')
        self.ccf_dir.mkdir(parents=True, exist_ok=True)

        def mask_absorption_emission(spec,specerr,lowerlim=0.7,upperlim=None):
            flux_out = np.copy(spec)
            fluxerr_out=np.copy(specerr)
            for nt in range(n_trans):
                for exp in range(n_exp):
                    for order in range(n_orders):
                        fl=flux_out[nt][exp][order]
                        flux_norm = fl/np.nanmedian(fl) #normalized spectrum
                        if upperlim is None:
                            goodpix = (flux_norm>lowerlim)
                        else:
                            goodpix = ((flux_norm>lowerlim)&(flux_norm<upperlim))
                        flux_out[nt][exp][order][~goodpix] = np.nan
                        fluxerr_out[nt][exp][order][~goodpix] = np.inf            
            return flux_out,fluxerr_out

        def remove_continuum(wave, spec, err=None, poly_order=3, get_cont=False):

            return_only_flux = False
            if err is None:
                return_only_flux = True
                err = np.ones_like(spec)
            
            if np.ndim(spec) == 1:
                nans = np.isnan(spec)
                continuum = np.poly1d(np.polyfit(wave[~nans],spec[~nans],poly_order))(wave)
                flux_contrem = spec / continuum
                err_contrem = err / continuum
                if return_only_flux:
                    return flux_contrem
                return (flux_contrem, err_contrem, continuum) if get_cont else (flux_contrem, err_contrem)

            else: # 3D flux
                flux_contrem = np.full((n_trans,n_exp,n_orders),np.nan,dtype=object)
                err_contrem=np.full((n_trans,n_exp,n_orders),np.inf,dtype=object)
                continua=np.full((n_trans,n_exp,n_orders),np.nan,dtype=object)
                for nt in range(n_trans):
                    for exp in range(n_exp):
                        for order in range(n_orders):
                            fl=spec[nt][exp][order]
                            flerr=err[nt][exp][order]
                            wl=wave[order]
                            nans = np.isnan(fl)
                            continuum_model = np.poly1d(np.polyfit(wl[~nans],fl[~nans],poly_order))
                            continuum = continuum_model(wl)

                            # Replace scalar NaN with a copy of the original spectrum before masking
                            flux_contrem[nt][exp][order] = fl.copy()*np.nan
                            err_contrem[nt][exp][order] = flerr.copy()*np.nan
                            continua[nt][exp][order] = continuum*np.nan

                            flux_contrem[nt][exp][order][~nans] = fl[~nans]/continuum[~nans]
                            err_contrem[nt][exp][order][~nans] = flerr[~nans]/continuum[~nans]
                            continua[nt][exp][order] = continuum   

                return (flux_contrem, err_contrem, continua) if get_cont else (flux_contrem, err_contrem)

        def mask_outliers(spec,specerr,sigma=outlier_sigma):
            flux_out = np.copy(spec)
            err_out=np.copy(specerr)
            for nt in range(n_trans):
                for exp in range(n_exp):
                    for order in range(n_orders):
                        fl=flux_out[nt][exp][order]
                        std = np.nanstd(fl)
                        mean = np.nanmedian(fl)
                        outliers = (np.abs(fl-mean)/std)>sigma
                        flux_out[nt][exp][order][outliers] = np.nan
                        err_out[nt][exp][order][outliers] = np.inf
            return flux_out,err_out

        flux_contrem, err_contrem = remove_continuum(wave, flux, err)
        # mask tellurics after removing continuum bc flux slope is too strong
        flux_mask0,err_mask0 = mask_absorption_emission(flux_contrem,err_contrem,lowerlim=0.7)
        flux_mask1,err_mask1 = mask_outliers(flux_mask0,err_mask0,sigma=outlier_sigma)

        flux_off_transit_avg = np.empty((n_trans,n_orders),dtype=object)#np.zeros_like(flux[0,:])
        for nt in range(n_trans):
            for order in range(n_orders):
                off_transit_sum = np.zeros(len(flux[nt][0][order])) #np.zeros_like(flux[0,0])
                for exp in range(n_exp):
                    if exp in self.in_transit:
                        pass
                    else:
                        fl=flux_mask1[nt][exp][order]
                        off_transit_sum +=fl
                off_transit_avg= off_transit_sum/np.nanmedian(off_transit_sum)
                flux_off_transit_avg[nt][order]=off_transit_avg

        flux_avgrem=np.zeros_like(flux_contrem)
        fluxerr_avgrem=np.zeros_like(flux_contrem)
        for nt in range(n_trans):
            for order in range(n_orders):
                for exp in range(n_exp):
                    flux_avgrem[nt][exp][order] = flux_mask1[nt][exp][order]/flux_off_transit_avg[nt][order]
                    fluxerr_avgrem[nt][exp][order] = err_mask1[nt][exp][order]/flux_off_transit_avg[nt][order]

        def sysrem(flux,fluxerr,num_modes=sysrem_iter):

            if flux.dtype==object:
                flux = np.vstack(flux)
                fluxerr = np.vstack(fluxerr)
            
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

        flux_sysrem=np.zeros((n_trans,n_exp,n_orders),dtype=object)
        if sysrem_iter>0:
            for nt in range(n_trans):
                for ord in range(n_orders): 
                    fl=flux_avgrem[nt][:,ord]
                    flerr=fluxerr_avgrem[nt][:,ord]
                    fl_new=sysrem(fl,flerr)
                    for exp in range(n_exp): # required for inhomogeneous object
                        flux_sysrem[nt][exp,ord] = fl_new[exp]
        else:
            #flux_sysrem = flux_avgrem   
            #n_trans, n_exp, n_orders = flux.shape
            for nt in range(n_trans):
                for order in range(n_orders):
                    fl = flux_avgrem[nt][:,order]
                    # median in-transit
                    flux_sysrem[nt][:,order]= fl -np.nanmedian(fl[n_exp//2])

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

        self.CCF = np.zeros((n_trans,n_exp,n_orders,len(self.RVs)))
        for nt in range(n_trans):
            for order in range(n_orders):
                wl=wave[order] # wavelengths are same per exposure
                fl=np.copy(flux_mask2[nt][:,order]) # get all exposures per order
                flerr=np.copy(fluxerr_mask2[nt][:,order])
                
                if fl.dtype==object: # all within same order have same length
                    fl = np.vstack(fl)
                    flerr = np.vstack(flerr)

                bad_pixels = ~np.isfinite(fl) | ~np.isfinite(flerr)
                fl[bad_pixels] = 0
                flerr[bad_pixels] = np.inf
                wl_shift=wl[:, np.newaxis]*beta[np.newaxis, :]
                template_shift=interp1d(template_wl,template_flux)(wl_shift) #interpolate template onto shifted wl-grid
                # do same for template as for spectra, remove cont with polydeg=3
                template_shift=np.array([remove_continuum(wav,temp) for wav,temp in zip(wl_shift.T,template_shift.T)])
                template_shift = template_shift - np.mean(template_shift, axis=0) # simplified continuum removal
                self.CCF[nt][:,order] = (fl/flerr**2).dot(template_shift.T) #error weighted

        self.Kp_list = np.arange(0,100,1)#km/s
        self.vsys_list=np.arange(-150,151,1) #km/s system velocities we want to shift to

        def shift_and_sum(ccf,vsys_list,rv_list):
            # shift and sum everything in the transit to combine the signal
            # this will be one row of the ccf map for one Kp value
            tot_ccf =np.zeros(vsys_list.shape) # or tot_ccf=0
            for nt in range(n_trans):
                for exp in self.in_transit:
                    x = self.RVs
                    y = ccf[nt][exp]
                    interp=interp1d(x, y, assume_sorted=True,bounds_error=False)(rv_list[exp]+vsys_list) 
                    tot_ccf=np.nansum(np.stack((tot_ccf,interp)),axis=0)
            return tot_ccf    
            
        def get_Kp_vsys_map(CCFs):
            CCF_tot = np.zeros((len(self.Kp_list),len(self.vsys_list)))
            for ikp in range(len(self.Kp_list)):
                rv_list = -self.vbary+self.vsys+self.Kp_list[ikp]*np.sin(2*np.pi*self.phase_obs) #calc planet rv for different Kp values
                CCF_tot[ikp] = shift_and_sum(CCFs,self.vsys_list,rv_list)
            return CCF_tot    

        CCF_sum=np.sum(self.CCF,axis=2) # sum CCF over all orders
        Kp_vsys_map = get_Kp_vsys_map(CCF_sum)
        noise=np.std(Kp_vsys_map[:,np.abs(self.vsys_list)>self.noiserange],axis=1)[:, None]
        self.Kp_vsys_map = Kp_vsys_map/noise #normalize along rv axis to get ccf map in snr units

        # S/N at expected self.Kp and vsys
        self.Kp_idx = find_nearest(self.Kp_list,value=self.Kp.value)
        self.vsys_idx = find_nearest(self.vsys_list,value=0)
        self.vsys0 = self.vsys_list[self.vsys_idx]
        self.Kp0 = self.Kp_list[self.Kp_idx]
        self.SNR_planet = self.Kp_vsys_map[self.Kp_idx,self.vsys_idx]

        if plot_order is not None:
            if plot_order=='all':
                for order in range(6):
                    self.plot_calib_steps(wave,flux,flux_contrem,flux_avgrem,flux_sysrem,order)
            else:
                self.plot_calib_steps(wave,flux,flux_contrem,flux_avgrem,flux_sysrem,plot_order)
            if self.template_name == 'input':
                self.plot_CCF(plot_order=plot_order)
                self.plot_Kp_vsys()

            ccf = self.Kp_vsys_map[self.Kp_idx]
            fname = f'{self.ccf_dir}/CCF_{self.template_name}.txt'
            np.savetxt(fname, ccf)

    def plot_calib_steps(self,wl,fl_orig,fl_contrem,fl_avgrem,fl_sysrem,plot_order,nt=0):

        fig,(ax1,ax2,ax3,ax4)=plt.subplots(4,1,figsize=(7,6),dpi=200,sharex=True)
        extent=[np.min(wl[plot_order]),np.max(wl[plot_order]),0,fl_orig.shape[0]]

        def imshow(ax,arr,title,vmin=0.5,vmax=99.5):
            slice_ = arr[nt][:,plot_order]
            if slice_.dtype==object:
                slice_ = np.vstack(slice_)
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
        fig.savefig(f'{self.figs_dir}/calib_steps_{plot_order}.pdf', bbox_inches='tight')
        plt.close()

    def plot_CCF(self,plot_order,nt=0):
        pm = 20
        xmin, xmax = min(self.rv_transit.value)-pm, max(self.rv_transit.value)+pm
        fig = plt.figure(figsize=(5,3),dpi=150)
        im = plt.imshow(self.CCF[nt][:,plot_order,:],origin="lower",aspect='auto',
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

    def plot_ccfs(self,species=['H2O','CH4','C2H2','C2H4','C2H6','input']):

        species_info = pd.read_csv('species_info.csv', index_col=0)
        fig,ax=plt.subplots(len(species),1,figsize=(4,len(species)),
                            dpi=200,sharex=True)
        
        for i,species_i in enumerate(species):
            fname = f'{self.ccf_dir}/CCF_{species_i}.txt'
            ccf = np.loadtxt(fname)
            if species_i=='input':
                col='k'
                mathtext = 'Input'
            else:
                col=species_info.loc[species_i,'color']
                mathtext=species_info.loc[species_i,'mathtext_name']
            ax[i].axvspan(-self.noiserange,self.noiserange,color='k',alpha=0.05)
            ax[i].axvline(x=0,color='k',lw=0.6,alpha=0.3)
            ax[i].axhline(y=0,color='k',lw=0.6,alpha=0.3)
            ax[i].plot(self.vsys_list,ccf,c=col)
            SNR = ccf[self.vsys_idx]
            label=f'{mathtext}\nS/N={np.round(SNR,decimals=1)}'
            ax[i].text(0.05, 0.9, label,transform=ax[i].transAxes,
                       fontsize=10,verticalalignment='top',color=col)
            ax[i].set_xlim(min(self.vsys_list),max(self.vsys_list))
            
        fig.supxlabel(r'$\Delta v_{\mathrm{sys}}$ [km/s]')
        fig.supylabel('S/N')
        fig.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        fig.savefig(f'{self.figs_dir}/CCFs_1D.pdf', bbox_inches='tight')
        plt.close()