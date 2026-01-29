import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import os
import copy
import sys
from astropy.io import fits
import astropy.constants as const
from utils import *
from scipy.interpolate import interp1d
import pathlib
from astropy.table import QTable
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, to_rgb
import matplotlib.colors as mcolors
import colorsys
from pRT_model import *
import shutil
import matplotlib as mpl
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import pandas as pd

class CrossCorr:

    def __init__(self, system_obj, wave, flux, err, outlier_sigma=3, 
                 sysrem_iter=1, plot_order=None,
                 use_transits=None,
                 plot_flux_err=True):

        inherit_attributes = ['project_path','delta_lambda','phase_obs','in_transit',
                              'planet_wl_um','Kp','rv_transit','phase_transit',
                              'planet_wl_obs_range','parameters',
                              'phase_ingress','phase_egress',
                              'rv_ingress','rv_egress']

        for attr in inherit_attributes:  # list of attributes to pass down
            setattr(self, attr, getattr(system_obj, attr))

        self.system_obj = system_obj
        self.wave = wave
        self.vbary = self.parameters['vbary']
        self.vsys = self.parameters['vsys']
        self.n_transits = self.parameters['n_transits']
        self.noiserange = 50
        self.window = 121 # Window length for Savitzky-Golay filter (must be odd)
        self.delta_RV = 0.2
        self.vsys_list=np.arange(-200,201,self.delta_RV) #km/s
        self.RVs = np.arange(-300, 300, self.delta_RV)
        self.Kp_list = np.arange(0,100,1) #km/s 

        self.n_trans, self.n_exp, self.n_orders = flux.shape
        if use_transits is not None:
            n_trans = use_transits
            self.n_transits = use_transits

        test_on_input = system_obj.parameters['test_on_input']
        inp = '_input' if test_on_input else ''
        self.figs_dir = pathlib.Path(f'{self.project_path}/figures/CC{inp}_sysit{sysrem_iter}_n{self.n_transits}')
        self.figs_dir.mkdir(parents=True, exist_ok=True)
        self.ccf_dir = pathlib.Path(f'{self.project_path}/CCFs/CC{inp}_sysit{sysrem_iter}_n{self.n_transits}')
        self.ccf_dir.mkdir(parents=True, exist_ok=True)

        err = fix_zero_errors(flux, err)
        #print('Bad errors',np.any(err <= 0), np.any(~np.isfinite(err))) 
        # both should be false
        #bad_nonfinite = any(np.any(~np.isfinite(np.asarray(e, dtype=float).ravel())) for e in err if e is not None)
        #bad_negative  = any(np.any(np.asarray(e, dtype=float).ravel() <= 0) for e in err if e is not None)
        #print("Bad errors:", bad_negative, bad_nonfinite)
        if plot_flux_err:
            self.plot_flux_errors(wave,flux,err)

        #snr_pixel_list, snr_order = snr_per_order_and_pixel(flux,err)
        #for i,sn in enumerate(snr_order):
            #print(f'S/N order {i} = ',sn)

        def mask_absorption_emission(spec,specerr,lowerlim=0.7,upperlim=None):
            flux_out = np.copy(spec)
            fluxerr_out=np.copy(specerr)
            for nt in range(self.n_trans):
                for exp in range(self.n_exp):
                    for order in range(self.n_orders):
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
                flux_contrem = np.full((n_trans,self.n_exp,self.n_orders),np.nan,dtype=object)
                err_contrem=np.full((n_trans,self.n_exp,self.n_orders),np.inf,dtype=object)
                continua=np.full((n_trans,self.n_exp,self.n_orders),np.nan,dtype=object)
                for nt in range(n_trans):
                    for exp in range(self.n_exp):
                        for order in range(self.n_orders):
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
            for nt in range(self.n_trans):
                for exp in range(self.n_exp):
                    for order in range(self.n_orders):
                        fl=flux_out[nt][exp][order]
                        std = np.nanstd(fl)
                        mean = np.nanmedian(fl)
                        outliers = (np.abs(fl-mean)/std)>sigma
                        flux_out[nt][exp][order][outliers] = np.nan
                        err_out[nt][exp][order][outliers] = np.inf
            return flux_out,err_out

        #flux_contrem, err_contrem = remove_continuum(wave, flux, err)
        
        # mask tellurics after removing continuum bc flux slope is too strong
        #flux_mask0,err_mask0 = mask_absorption_emission(flux_contrem,err_contrem,lowerlim=0.7)
        #flux_mask1,err_mask1 = mask_outliers(flux_mask0,err_mask0,sigma=outlier_sigma)

        #flux_hp, err_hp = highpass(wave, flux, err)
        #flux_mask0, err_mask0 = mask_highpass(flux_hp, err_hp)
        
        # Subtract mean per exposure & order
        #flux_contrem = flux_mask0.copy()
        #for nt in range(n_trans):
        #    for exp in range(n_exp):
        #        for order in range(n_orders):
        #            fl = flux_contrem[nt][exp][order]
        #            if fl is None or not isinstance(fl, np.ndarray):
        #                continue
        #            flux_contrem[nt][exp][order] = fl #- np.nanmean(fl)
        #err_mask1 = err_mask0
        #flux_mask1= flux_contrem

        # mask deep telluric regions
        telluric_ref = np.zeros((self.n_trans, self.n_orders), dtype=object)

        for nt in range(self.n_trans):
            for order in range(self.n_orders):
                stack = []
                for exp in range(self.n_exp):
                    if exp not in self.in_transit:
                        stack.append(flux[nt][exp][order])
                telluric_ref[nt][order] = np.nanmedian(stack, axis=0)

        telluric_ref_norm = np.zeros_like(telluric_ref)

        for nt in range(self.n_trans):
            for order in range(self.n_orders):
                ref = telluric_ref[nt][order]
                cont = np.nanmedian(ref)
                telluric_ref_norm[nt][order] = ref / cont

        telluric_mask = np.zeros((self.n_trans, self.n_orders), dtype=object)

        TELLURIC_CUTOFF = 0.7   # typical: 0.6–0.8 for L band

        for nt in range(self.n_trans):
            for order in range(self.n_orders):
                ref = telluric_ref_norm[nt][order]
                telluric_mask[nt][order] = ref > TELLURIC_CUTOFF

        flux_mask1 = copy.deepcopy(flux)
        err_mask1  = copy.deepcopy(err)

        for nt in range(self.n_trans):
            for exp in range(self.n_exp):
                for order in range(self.n_orders):
                    good = telluric_mask[nt][order]
                    flux_mask1[nt][exp][order][~good] = np.nan
                    
        # remove off-transit average
        flux_off_transit_avg = np.empty((self.n_trans,self.n_orders),dtype=object)#np.zeros_like(flux[0,:])
        for nt in range(self.n_trans):
            for order in range(self.n_orders):
                oot_stack = [] # out of transit
                for exp in range(self.n_exp):
                    if exp not in self.in_transit:
                        oot_stack.append(flux_mask1[nt][exp][order])

                oot_stack = np.array(oot_stack)
                off_transit_avg = np.nanmedian(oot_stack, axis=0)
                off_transit_avg /= np.nanmedian(off_transit_avg)
                #off_transit_avg = median_filter(off_transit_sum, size=31)
                #off_transit_avg /= np.nanmedian(off_transit_avg)
                flux_off_transit_avg[nt][order]=off_transit_avg

        flux_avgrem=np.zeros_like(flux)
        fluxerr_avgrem=np.zeros_like(flux)
        for nt in range(self.n_trans):
            for order in range(self.n_orders):
                for exp in range(self.n_exp):
                    flux_avgrem[nt][exp][order] = flux_mask1[nt][exp][order]/flux_off_transit_avg[nt][order]
                    fluxerr_avgrem[nt][exp][order] = err_mask1[nt][exp][order]/flux_off_transit_avg[nt][order]

        # Remove broadband transit depth per exposure
        for nt in range(self.n_trans):
            for exp in range(self.n_exp):
                for order in range(self.n_orders):
                    fl = flux_avgrem[nt][exp][order]
                    if fl is None or not isinstance(fl, np.ndarray):
                        continue
                    flux_avgrem[nt][exp][order] = fl / np.nanmedian(fl)

        # --- SysRem definition ---
        def sysrem(flux, fluxerr, mask=None, num_modes=sysrem_iter):
            """
            SysRem algorithm.
            flux: (n_exp, n_pix)
            fluxerr: same shape
            mask: boolean array for training (out-of-transit)
            """
            if flux.dtype == object:
                flux = np.vstack(flux)
                fluxerr = np.vstack(fluxerr)

            flux_corrected = flux.copy()

            if mask is None:
                mask = np.ones(flux.shape[0], dtype=bool)

            f_train = flux[mask]
            ferr_train = fluxerr[mask]

            a = np.ones(f_train.shape[0])
            residuals_train = f_train.copy()

            for _ in range(num_modes):
                # Learn one SysRem mode
                c = np.nansum(residuals_train.T * a / ferr_train.T**2, axis=1) / np.nansum(a**2 / ferr_train.T**2, axis=1)
                a = np.nansum(c * residuals_train / ferr_train**2, axis=1) / np.nansum(c**2 / ferr_train**2, axis=1)
                residuals_train -= np.outer(a, c)

            # Apply learned mode to all exposures
            a_full = np.nansum(flux * c / fluxerr**2, axis=1) / np.nansum(c**2 / fluxerr**2, axis=1)
            flux_corrected -= np.outer(a_full, c)

            return flux_corrected

        # --- Apply SysRem to all exposures ---
        flux_sysrem = np.zeros((self.n_trans, self.n_exp, self.n_orders), dtype=object)
        mask_train = np.ones(self.n_exp, dtype=bool)
        mask_train[self.in_transit] = False  # train only on out-of-transit

        for nt in range(self.n_trans):
            for ord in range(self.n_orders):
                fl = flux_avgrem[nt][:, ord]
                flerr = fluxerr_avgrem[nt][:, ord]
                fl_new = sysrem(fl, flerr, mask_train)
                for exp in range(self.n_exp):
                    flux_sysrem[nt][exp, ord] = fl_new[exp]

        # --- Outlier masking ---
        flux_mask2, fluxerr_mask2 = mask_outliers(flux_sysrem, fluxerr_avgrem, sigma=3)

        if plot_order is not None:
            if plot_order == 'all':
                for order in range(self.n_orders):
                    self.plot_calib_steps(wave, flux, flux_mask1, flux_avgrem, flux_sysrem, order)
            else:
                self.plot_calib_steps(wave, flux, flux_mask1, flux_avgrem, flux_sysrem, plot_order)

        self.flux_mask2 = flux_mask2
        self.fluxerr_mask2 = fluxerr_mask2
    
    def cross_correlate(self, flux_mask2, fluxerr_mask2, plot_order=0,abund=-10,
                        template=None, plot_masked_contributions=False):
        
        if abund=='best' and template!='input':
            best_vmrs, _ = retrieve_best_vmrs(self.system_obj)
            abund = np.log10(best_vmrs[template])
            print(f'Using VMR = {abund} for {template}')
        if template is None or template=='input':
            template_wl = self.planet_wl_um
            template_flux = np.ones_like(self.delta_lambda) - self.delta_lambda
            self.template_name = 'input'
        else:
            template_path = pathlib.Path(f'{self.project_path}/pRT_spectra/1e{int(abund)}/{template}.fits')
            template_path.parent.mkdir(parents=True, exist_ok=True)
            if template_path.exists():
                tbl = QTable.read(template_path)
                template_wl = tbl['wavelength']
                transit_radii_um = tbl['flux']
            else:
                params2 = self.parameters.copy()
                params2['chemistry'] = 'free'
                #params2['species_names'] = [template]
                #params2[f'log_{template}'] = 0
                if abund==0:
                    params2['species_names'] = [template]
                else:
                    params2['species_names'] = [template,'H2','He']
                params2[f'log_{template}'] = abund
                params2['temperature'] = 300
                #if template=='C2H2':
                   # params2['resolution'] = 1e5/2
                   #print('Using resolution',params2['resolution'],'for C2H2')
                template_wl, transit_radii_um = pRT_spectrum(params2).make_spectrum(save_path=template_path,species=template)

            transit_radii_cm = transit_radii_um.to(u.cm)
            delta_lambda = ((transit_radii_cm / self.system_obj.R_star.to(u.cm))**2).value
            template_flux = np.ones_like(delta_lambda) - delta_lambda
            self.template_name = template

        if self.template_name=='input' and plot_masked_contributions:
            self.plot_transit_contributions_masked(self.wave,flux_mask2)

        if False: #self.template_name=='C2H2':
            self.plot_maskedflux_species(self.wave,flux_mask2,
                                        template_wl,template_flux,
                                        self.template_name,
                                        abund=abund,plot_order=0)

        # --- CCF computation ---
        beta = 1.0 - self.RVs / const.c.to('km/s').value
        self.CCF = np.zeros((self.n_trans, self.n_exp, self.n_orders, len(self.RVs)))
        self.ACF = np.zeros((self.n_orders, len(self.RVs)))

        # Order-selection rules for each template
        # (0-indexed order numbers)
        order_selector = {
            'C2H2': [0],               # Only the first order
            'C2H6': [4, 5],            # Last two orders
            'CO2':  [0, 1, 2, 3],      # All except last two
            'C2H4': [1, 2, 3, 4, 5]    # All except first order
        }

        if template in order_selector and self.n_orders==6:
            selected_orders = order_selector[template]
        else:
            selected_orders = list(range(self.n_orders))  # Default: use all orders

        print(f"Using orders {selected_orders} for template {template}")

        def bin_last_axis(arr, factor):
            shape = arr.shape
            n_bins = shape[-1] // factor
            trimmed = arr[..., :n_bins*factor]  # drop extra elements
            binned = trimmed.reshape(*shape[:-1], n_bins, factor).mean(axis=-1)
            return binned

        for nt in range(self.n_trans):
            for order in selected_orders:
                wl = self.wave[order]
                fl = np.copy(flux_mask2[nt][:, order])
                flerr = np.copy(fluxerr_mask2[nt][:, order])

                # Fix object dtype
                if fl.dtype == object:
                    fl = np.vstack(fl)
                    flerr = np.vstack(flerr)

                # Error → weights
                bad = ~np.isfinite(fl) | ~np.isfinite(flerr) | (flerr <= 0)
                fl[bad] = 0.0
                weights = np.zeros_like(fl)
                weights[~bad] = 1.0 / flerr[~bad]**2
                #weights[~bad] = weights[~bad] / np.nanmedian(weights[~bad])
                fl -= np.nanmean(fl, axis=1, keepdims=True)
                fl /= np.nanstd(fl, axis=1, keepdims=True)

                # normalize the data like the template
                #for i in range(fl.shape[0]):
                #    good = weights[i] > 0
                #    if np.sum(good) > 10:
                #        fl[i, good] -= np.nanmean(fl[i, good])
                #        fl[i, good] /= np.sqrt(np.nanmean(fl[i, good]**2))

                # Doppler-shifted template
                wl_shift = wl[:, np.newaxis] * beta[np.newaxis, :]
                template_shift = interp1d(
                    template_wl, template_flux,
                    bounds_error=False, fill_value=0.0
                )(wl_shift)
                template_shift -= np.mean(template_shift, axis=0)
                template_shift /= np.nanstd(template_shift, axis=0, keepdims=True)
                template_shift = template_shift.T
                #template_shift /= np.sqrt(np.nanmean(template_shift**2, axis=0))

                # High-pass filter the template
                #template_shift = np.array([
                #    self.highpass(wav, temp)
                #    for wav, temp in zip(wl_shift.T, template_shift.T)
                #])

                # just to see what it looks like for one
                #_ = self.highpass(wl_shift.T[0], template_shift.T[0],species=self.template_name,plot=True)

                # 1. Compute pixel-wise template weight: sum over RVs
                #temp_pixel_weight = np.sqrt(np.nansum(template_shift**2, axis=0))  # shape (n_pix,)
                # 2. Multiply data weights per exposure by template pixel weight
                #weights_eff = weights * temp_pixel_weight[np.newaxis, :]  # shape (n_exp, n_pix)
                #weights = weights_eff

                #if template=='C2H2':
                #    fl = bin_last_axis(fl, factor=3)
                #    weights = bin_last_axis(weights, factor=3)
                #    template_shift = bin_last_axis(template_shift, factor=3)

                #temp_pixel_weight = np.nansum(template_shift**2, axis=0)
                #temp_pixel_weight /= np.nanmax(temp_pixel_weight)
               # weights_eff = weights * temp_pixel_weight[np.newaxis, :]
               # weights = weights_eff

                self.CCF[nt][:, order, :] = (fl * weights).dot(template_shift.T)
                #self.CCF[nt][:, order, :] -= np.nanmedian(
                #    self.CCF[nt][:, order, :],
                #    axis=1,
                #    keepdims=True)
                #std = np.nanstd(self.CCF[nt][:, order, :], axis=1, keepdims=True)
                #self.CCF[nt][:, order, :] /= std

                # Take RV=0 as reference
                rv0_idx = len(self.RVs)//2
                template0 = template_shift[rv0_idx, :]  # shape (n_pix,)
                template0 -= np.median(template0)
                # ACF in RV space
                template_shift -= np.mean(template_shift, axis=0)
                acf_1d = (template0).dot(template_shift.T)  # shape (n_rv,)
                self.ACF[order] = acf_1d

        # --- Kp-vsys map ---
        CCF_sum_orders = np.sum(self.CCF, axis=2)
        Kp_vsys_maps_nt = np.zeros((self.n_trans, len(self.Kp_list), len(self.vsys_list)))

        for nt in range(self.n_trans):
            CCF_nt = CCF_sum_orders[nt]
            for ikp, Kp in enumerate(self.Kp_list):
                tot_ccf = np.zeros(len(self.vsys_list))
                rv_list = -self.vbary + self.vsys + Kp * np.sin(2 * np.pi * self.phase_obs)
                for exp in self.in_transit:
                    interp = interp1d(
                        self.RVs, CCF_nt[exp],
                        bounds_error=False,
                        fill_value=0.0,
                        assume_sorted=True
                    )(rv_list[exp] + self.vsys_list)
                    tot_ccf += interp
                tot_ccf /= np.sqrt(len(self.in_transit))
                Kp_vsys_maps_nt[nt, ikp] = tot_ccf

        def highpass_1d(y, width):
            baseline = gaussian_filter1d(y, width)
            return y - baseline

        for nt in range(self.n_trans):
            for ikp in range(len(self.Kp_list)):
                Kp_vsys_maps_nt[nt, ikp] = highpass_1d(
                    Kp_vsys_maps_nt[nt, ikp],
                    width=50)

        # --- Normalize per night ---
        mask_noise = np.abs(self.vsys_list) > self.noiserange
        noise_nt = np.array([np.std(Kp_vsys_maps_nt[nt][:, mask_noise]) for nt in range(self.n_trans)])
        Kp_vsys_maps_nt /= noise_nt[:, None, None]
        self.Kp_vsys_maps_per_transit = Kp_vsys_maps_nt.copy()

        # --- Stack nights ---
        self.Kp_vsys_map = np.sum(Kp_vsys_maps_nt, axis=0)

        # --- S/N at expected planet location ---
        self.Kp_idx = find_nearest(self.Kp_list, value=self.Kp.value)
        self.vsys_idx = find_nearest(self.vsys_list, value=0)
        self.vsys0 = self.vsys_list[self.vsys_idx]
        self.Kp0 = self.Kp_list[self.Kp_idx]
        self.ccf_1d = self.Kp_vsys_map[self.Kp_idx]

        self.acf_1d = np.sum(self.ACF[selected_orders], axis=0)
        mask_noise_acf = np.abs(self.RVs) > self.noiserange
        noise_acf = np.std(self.acf_1d[mask_noise_acf])
        self.acf_1d /= noise_acf

        dv = 1.0
        mask = np.abs(self.vsys_list - self.vsys0) <= dv
        self.SNR_planet = local_peak_snr(self.vsys_list[mask], self.ccf_1d[mask], dv=dv)

        if plot_order is not None and self.template_name =='input':
            plot_order = plot_order if plot_order!='all' else 0
            self.plot_CCF(plot_order=plot_order)
            self.plot_Kp_vsys()

        #fname = f'{self.ccf_dir}/CCF_{self.template_name}.txt'
        #np.savetxt(fname, self.ccf_1d)

    def highpass(self, wave, spec, err=None, W=None, poly=2, species=None, get_cont=False, plot=False):
        """
        High-pass filter a spectrum or 3D array of spectra (n_trans, n_exp, n_orders),
        in the same style as remove_continuum.

        Parameters
        ----------
        wave : array-like
            Wavelength array (for 1D spectra) or list/array of arrays per order.
        spec : array-like
            1D spectrum or 3D array of spectra.
        err : array-like or None
            Uncertainties, same shape as spec. If None, treated as ones.
        W : int
            Window length for Savitzky-Golay filter (must be odd).
        poly : int
            Polynomial order for Savitzky-Golay filter.
        get_cont : bool
            If True, return the baseline as well.

        Returns
        -------
        flux_hp : high-pass filtered flux (same shape as input)
        err_hp : filtered errors (same shape as input)
        baseline : baseline(s) if get_cont=True
        """
        W=self.window if W is None else W
        return_only_flux = False
        if err is None:
            return_only_flux = True
            err = np.ones_like(spec)

        if np.ndim(spec) == 1:
            # 1D spectrum
            nans = np.isnan(spec)
            if np.sum(~nans) < 5:
                baseline = np.zeros_like(spec)
            else:
                W_eff = min(W, (np.sum(~nans) // 2) * 2 + 1)
                baseline = savgol_filter(spec[~nans], W_eff, poly, mode='interp')
                full_baseline = np.full_like(spec, np.nan)
                full_baseline[~nans] = baseline 
                baseline = full_baseline

            flux_hp = spec / baseline - 1.0
            err_hp = err / baseline

            if plot==True:
                fig, ax = plt.subplots(1, 1, figsize=(5,3))
                ax.plot(wave,spec,c='k',label='Input')
                ax2 = ax.twinx()
                ax2.plot(wave,flux_hp,c='r',label='High-pass')
                fig.tight_layout()
                fig.savefig(f'{self.figs_dir}/highpass_W{self.window}_{species}.pdf', bbox_inches='tight')
                plt.close()
            
            if return_only_flux:
                return flux_hp
            return (flux_hp, err_hp, baseline) if get_cont else (flux_hp, err_hp)

        else:
            # 3D array: (n_trans, n_exp, n_orders)
            n_trans, n_exp, n_orders = spec.shape
            flux_hp = np.full((n_trans, n_exp, n_orders), np.nan, dtype=object)
            err_hp = np.full((n_trans, n_exp, n_orders), np.inf, dtype=object)
            baseline_arr = np.full((n_trans, n_exp, n_orders), np.nan, dtype=object)
            
            for nt in range(n_trans):
                for exp in range(n_exp):
                    for order in range(n_orders):
                        fl = spec[nt][exp][order]
                        flerr = err[nt][exp][order]
                        wl = wave[order]# if np.ndim(wave) > 1 else wave

                        if plot==True and nt==0 and exp==0 and order==0:
                            fig, ax = plt.subplots(1, 1, figsize=(5,3))
                            ax.plot(wl,fl,c='k',label='Input')

                        nans = np.isnan(fl)
                        if np.sum(~nans) < 5:
                            baseline = np.zeros_like(fl)
                        else:
                            W_eff = min(W, (np.sum(~nans) // 2) * 2 + 1)
                            baseline_model = savgol_filter(fl[~nans], W_eff, poly, mode='interp')
                            baseline = np.full_like(fl, np.nan)
                            baseline[~nans] = baseline_model

                        flux_hp[nt][exp][order] = fl.copy()*np.nan
                        err_hp[nt][exp][order] = flerr.copy()*np.nan
                        baseline_arr[nt][exp][order] = baseline.copy()*np.nan

                        flux_hp[nt][exp][order][~nans] = fl[~nans]/baseline[~nans] - 1.0
                        err_hp[nt][exp][order][~nans] = flerr[~nans]/baseline[~nans]
                        baseline_arr[nt][exp][order] = baseline

                        if plot==True and nt==0 and exp==0 and order==0:
                            ax2 = ax.twinx()
                            ax2.plot(wl,flux_hp[nt][exp][order],c='r',label='High-pass')
                            fig.tight_layout()
                            fig.savefig(f'{self.figs_dir}/highpass_W{self.window}.pdf', bbox_inches='tight')
                            plt.close()

            return (flux_hp, err_hp, baseline_arr) if get_cont else (flux_hp, err_hp)

    def plot_calib_steps(self,wl,fl_orig,fl_masked,fl_avgrem,fl_sysrem,plot_order,nt=0):

        fig,(ax1,ax2,ax3,ax4)=plt.subplots(4,1,figsize=(7,6),dpi=200,sharex=True)
        extent=[np.min(wl[plot_order]),np.max(wl[plot_order]),0,fl_orig.shape[0]]

        def imshow(ax,arr,title,vmin=0.5,vmax=99.5):
            slice_ = arr[nt][:,plot_order]
            if slice_.dtype==object:
                slice_ = np.vstack(slice_)
            im=ax.imshow(slice_,aspect='auto', origin ='lower',extent=extent, cmap=custom_cmap,
                        vmin=np.nanpercentile(slice_, vmin), vmax=np.nanpercentile(slice_, vmax))
            fig.colorbar(im,ax=ax)
            ax.set_title(title)

        imshow(ax1,fl_orig,f'Original flux, order {plot_order}')
        imshow(ax2,fl_masked,'Tellurics masked')
        imshow(ax3,fl_avgrem,'Average off-transit removed')
        imshow(ax4,fl_sysrem,'After SYSREM')

        ax4.set_xlabel(r'Wavelength [$\mathrm{\mu}$m]')
        fig.tight_layout()
        fig.savefig(f'{self.figs_dir}/calib_steps_{plot_order}.pdf', bbox_inches='tight')
        plt.close()

    def plot_flux_errors(self,wl,fl,err,nt=0):

        figs=[]
        for plot_order in range(self.n_orders):
            data_wave = wl[plot_order]
            fig,(ax1,ax2)=plt.subplots(2,1,figsize=(7,4),dpi=200,sharex=True)
            extent=[np.min(data_wave),np.max(data_wave),0,fl.shape[0]]
            slice_flux = fl[nt][:,plot_order]
            if slice_flux.dtype==object:
                slice_flux = np.vstack(slice_flux)
            im1 = ax1.imshow(slice_flux,aspect='auto', origin ='lower',extent=extent, cmap=custom_cmap,
                        vmin=np.nanpercentile(slice_flux, 0.5), vmax=np.nanpercentile(slice_flux, 99.5))
            ax1.set_title(f'Flux (top), error (bottom), order {plot_order}')
            slice_err = err[nt][:,plot_order]
            if slice_err.dtype==object:
                slice_err = np.vstack(slice_err)
            im2 = ax2.imshow(slice_err,aspect='auto', origin ='lower',extent=extent, cmap=custom_cmap,
                        vmin=np.nanpercentile(slice_err, 0.5), vmax=np.nanpercentile(slice_err, 99.5))
            ax2.set_xlabel('Wavelength (um)')
            fig.colorbar(im1, ax=ax1)
            fig.colorbar(im2, ax=ax2)
            fig.tight_layout()
            plt.subplots_adjust(wspace=0, hspace=0)
            figs.append(fig)
        with PdfPages(f'{self.figs_dir}/flux_err.pdf') as pdf:
            for fig in figs:
                plt.figure(fig.number)
                pdf.savefig()
                plt.close()

    def plot_maskedflux_species(self,wl,fl,wl_temp,fl_temp,species_i,plot_order,nt=0,abund=-10):

        species_info = pd.read_csv('species_info.csv', index_col=0)
        if species_i != 'input':
            col=species_info.loc[species_i,'color']
            mathtext=species_info.loc[species_i,'mathtext_name']
        else:
            col = 'k'
            mathtext = 'Input'

        data_wave = wl[plot_order]
        fig,(ax1,ax2)=plt.subplots(2,1,figsize=(7,4),dpi=200,sharex=True)
        extent=[np.min(data_wave),np.max(data_wave),0,fl.shape[0]]

        slice_ = fl[nt][:,plot_order]
        if slice_.dtype==object:
            slice_ = np.vstack(slice_)
        data_flux_1row = slice_[0]
        ax1.imshow(slice_,aspect='auto', origin ='lower',extent=extent, cmap=custom_cmap,
                    vmin=np.nanpercentile(slice_, 0.5), vmax=np.nanpercentile(slice_, 99.5))
        ax1.set_title(f'After SYSREM, order {plot_order}')

        # interp template onto data grid
        template_interp = interp1d(wl_temp, fl_temp, bounds_error=False, 
                                  fill_value=0.0)(data_wave)
        ax2.plot(data_wave,template_interp,alpha=0.3,c=col)
        mask = np.isfinite(data_flux_1row)
        template_interp[~mask] = np.nan
        ax2.plot(data_wave,template_interp,label=mathtext,c=col)
        ax2.set_xlabel(r'Wavelength [$\mathrm{\mu}$m]')
        fig.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        fig.savefig(f'{self.figs_dir}/processed_{species_i}_1e{int(abund)}.pdf', bbox_inches='tight')
        plt.close()

    def plot_transit_contributions_masked(self,wl,fl,nt=0):
        lw=1
        alph=1
        #species = self.parameters['species_names']
        # get species contributions to transit radius
        from config_run import init_simulation
        fig_init,ax_init=plt.subplots(1,1,figsize=(7,4),dpi=200)
        system_obj = init_simulation(self.parameters['project_name'],plot_spectrum=True,ax=ax_init)

        figs=[]
        for plot_order in range(6):
            data_wave = wl[plot_order]
            fig,(ax1,ax2)=plt.subplots(2,1,figsize=(7,4),dpi=200,sharex=True)
            extent=[np.min(data_wave),np.max(data_wave),0,fl.shape[0]]
            slice_ = fl[nt][:,plot_order]
            if slice_.dtype==object:
                slice_ = np.vstack(slice_)
            data_flux_1row = slice_[0]
            ax1.imshow(slice_,aspect='auto', origin ='lower',extent=extent, cmap=custom_cmap,
                        vmin=np.nanpercentile(slice_, 0.5), vmax=np.nanpercentile(slice_, 99.5))
            ax1.set_title(f'After SYSREM, order {plot_order}')

            copy_axes_contents(ax_init, ax2)
            ax2.set_xlim(np.min(data_wave),np.max(data_wave))
            mask = np.isfinite(data_flux_1row)
            bad = ~mask
            # Find start/end indices of contiguous bad regions
            edges = np.diff(bad.astype(int))
            starts = np.where(edges == 1)[0] + 1
            ends   = np.where(edges == -1)[0] + 1

            # Handle if mask starts or ends with a bad region
            if bad[0]:
                starts = np.r_[0, starts]
            if bad[-1]:
                ends = np.r_[ends, len(bad)]
            for s, e in zip(starts, ends):
                x0 = data_wave[s]
                x1 = data_wave[e-1]
                ax2.axvspan(x0, x1, color="white", alpha=0.7, zorder=10)
            ax2.set_ylim(0.218, 0.243)
            for line in ax2.get_lines():
                line.set_linewidth(lw)
                line.set_alpha(alph)
            ax2.set_xlabel('Wavelength (um)')
            fig.tight_layout()
            plt.subplots_adjust(wspace=0, hspace=0)
            figs.append(fig)
        with PdfPages(f'{self.figs_dir}/contributions_masked_order.pdf') as pdf:
            for fig in figs:
                plt.figure(fig.number)
                pdf.savefig()
                plt.close()

    def plot_CCF(self,plot_order,nt=0):
        pm = 20
        xmin, xmax = min(self.rv_transit.value)-pm, max(self.rv_transit.value)+pm
        fig = plt.figure(figsize=(5,3),dpi=150)
        im = plt.imshow(self.CCF[nt][:,plot_order,:],origin="lower",aspect='auto', cmap=custom_cmap,
                extent=[np.min(self.RVs),np.max(self.RVs),np.min(self.phase_obs),np.max(self.phase_obs)])
        plt.colorbar(im,label='CCF')
        plt.plot(self.rv_transit,self.phase_transit,c='white',linestyle='dashed',alpha=0.5,lw=2)
        plt.hlines(self.phase_egress,xmin, xmax,color='white',linestyle='dashdot',alpha=0.8,lw=2)
        plt.hlines(self.phase_ingress,xmin, xmax,color='white',linestyle='dashdot',alpha=0.8,lw=2)
        plt.xlabel('Radial velocity [km/s]')
        plt.ylabel('Phase')
        plt.xlim(xmin, xmax)
        fig.savefig(f'{self.figs_dir}/CCF_{self.template_name}.pdf', bbox_inches='tight')
        plt.close()
    
    def plot_Kp_vsys(self):
        fig = plt.figure(figsize=(5,3),dpi=150)
        im=plt.imshow(self.Kp_vsys_map,aspect='auto',origin='lower', cmap=custom_cmap,
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

    def plot_ccfs(self,species=['H2O','CH4','C2H2','C2H4','C2H6','input'],
                  plot_acf=True):

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

def local_peak_snr(vsys, ccf, v0=0, dv=1.0, min_curvature=1e-3):
    """
    Robust S/N estimate near expected planet velocity v0.
    - Tries a quadratic peak fit within ±dv km/s
    - If the fit is unreliable, returns S/N at exact planet velocity
    """

    # --- fallback value: exact planet position ---
    #idx0 = np.argmin(np.abs(vsys - v0))
    idx0 = find_nearest(vsys, value=v0)
    snr_exact = ccf[idx0]

    # --- select local window ---
    mask = np.abs(vsys - v0) <= dv
    x = vsys[mask]
    y = ccf[mask]

    # Not enough points → fallback
    if len(y) < 5:
        return snr_exact

    # Max must be inside window (not at edges)
    i = np.argmax(y)
    if i == 0 or i == len(y) - 1:
        return snr_exact

    # Quadratic fit around the maximum
    try:
        coeffs = np.polyfit(x[i-1:i+2], y[i-1:i+2], 2)
        a, b, c = coeffs

        # Curvature too small → flat / noisy
        if np.abs(2 * a) < min_curvature:
            return snr_exact

        # Peak location
        v_peak = -b / (2 * a)

        # Peak must lie close to expected velocity
        if np.abs(v_peak - v0) > dv:
            return snr_exact

        snr_peak = np.polyval(coeffs, v_peak)

        # Sanity check: fitted peak should exceed exact-point S/N
        if not np.isfinite(snr_peak) or snr_peak < snr_exact:
            return snr_exact
        return snr_peak

    except Exception:
        return snr_exact


def multi_transit_ccfs(system_obj, wl_obs, fl_array, err_array,
                       species=['H2O','CH4','C2H2','C2H4','C2H6','input'],
                       transits=1, sysit=1,plot_acf=True,abund=-10,
                       animate=True, fps=2,
                       single_transit_only=False,
                       single_n_trans=5):

    species_info = pd.read_csv('species_info.csv', index_col=0)

    if isinstance(transits, list):
        tr = transits[-1]
    else:
        tr = transits # max num of transits
        transits = range(1, tr + 1)

    # preprocessing
    cc_obj = CrossCorr(system_obj, wl_obs, fl_array, err_array,
                       sysrem_iter=sysit, use_transits=tr,plot_order='all')
    flux_mask2 = cc_obj.flux_mask2
    fluxerr_mask2 = cc_obj.fluxerr_mask2
    acf_dir = pathlib.Path(f'{system_obj.project_path}/ACFs')
    acf_dir.mkdir(parents=True, exist_ok=True)

    ccf_store = {s: {} for s in species}
    acf_store = {s: {} for s in species}
    snr_store = {s: {} for s in species}
    n_trans_max = max(transits)  # maximum number of nights

    # --- compute the full cross-correlation once per species ---
    maps_store = {}    # store full maps per species

    for species_i in species:
        # Compute for max transits
        cc_obj.n_trans = n_trans_max
        cc_obj.cross_correlate(flux_mask2,
                            fluxerr_mask2,
                            template=species_i,
                            abund=abund,
                            plot_masked_contributions=False)

        maps_store[species_i] = cc_obj.Kp_vsys_maps_per_transit
        acf_store[species_i] = cc_obj.acf_1d  # same acf for all n_trans

    # --- now just slice for fewer transits ---
    for n_trans in transits:
        for species_i in species:
            maps_per_transit = maps_store[species_i]
            Kp_idx = find_nearest(cc_obj.Kp_list, value=cc_obj.Kp.value)
            ccf_1d = np.sum(maps_per_transit[:n_trans], axis=0)[Kp_idx]
            ccf_store[species_i][n_trans] = ccf_1d
            snr_store[species_i][n_trans] = local_peak_snr(cc_obj.vsys_list, ccf_1d, v0=0.0, dv=1.0)

    if plot_acf:

        fig, ax = plt.subplots(len(species), 1,
                           figsize=(4, len(species)),
                           dpi=200, sharex=True)
        
        if len(species) == 1:
            ax = [ax]
        for i, species_i in enumerate(species):
            acf = acf_store[species_i]

            if species_i == 'input':
                col = 'k'
                mathtext = 'Input'
            else:
                col = species_info.loc[species_i, 'color']
                mathtext = species_info.loc[species_i, 'mathtext_name']

            # Plot new line with alpha=1
            ax[i].axvspan(-cc_obj.noiserange, cc_obj.noiserange,
                      color='k', alpha=0.05)
            ax[i].axvline(0, color='k', lw=0.6, alpha=0.3)
            ax[i].axhline(0, color='k', lw=0.6, alpha=0.3)
            ax[i].plot(cc_obj.RVs, acf, c=col, alpha=1,label=mathtext)
            ax[i].set_xlim(min(cc_obj.vsys_list), max(cc_obj.vsys_list))
            ax[i].legend(loc='upper left')

        ax[-1].set_xlabel(r'$\Delta v_{\mathrm{sys}}$ [km/s]',fontsize=11)
        fig.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        fname = f'{cc_obj.figs_dir}/ACFs.pdf'
        print('Saving as', fname)
        fig.savefig(fname, bbox_inches='tight')
        plt.close()

    fig, ax = plt.subplots(len(species), 1,
                           figsize=(4, len(species)),
                           dpi=200, sharex=True)

    if len(species) == 1:
        ax = [ax]

    ylims = {}
    for species_i in species:
        vals = np.concatenate(
            [ccf_store[species_i][n] for n in transits])
        ymin, ymax = vals.min(), vals.max()
        pad = 0.05 * (ymax - ymin)
        ylims[species_i] = (ymin - pad, ymax + pad)
    
    for i, species_i in enumerate(species):
        ax[i].axvspan(-cc_obj.noiserange, cc_obj.noiserange,
                      color='k', alpha=0.05)
        ax[i].axvline(0, color='k', lw=0.6, alpha=0.3)
        ax[i].axhline(0, color='k', lw=0.6, alpha=0.3)
        ax[i].set_xlim(min(cc_obj.vsys_list), max(cc_obj.vsys_list))
        ax[i].set_ylim(*ylims[species_i])

    if len(species) == 1:
        ax[0].set_ylim(*ylims[species_i])
        fig.set_size_inches(4, 2)

    ax[-1].set_xlabel(r'$\Delta v_{\mathrm{sys}}$ [km/s]',fontsize=11)
    fig.supylabel('S/N',fontsize=11)

    lines = {s: [] for s in species}
    text_artists = [None] * len(species)

    def update(frame):
        if single_transit_only:
            n_trans = single_n_trans
        else:
            n_trans = frame + 1

        if animate:
            fig.suptitle(f'Number of transits: {n_trans}', fontsize=11)
        else:
            label = system_obj.parameters['label']
            ax[0].set_title(label, fontsize=11)

        for i, species_i in enumerate(species):
            ccf = ccf_store[species_i][n_trans]

            if species_i == 'input':
                col = 'k'
                mathtext = 'Input'
            else:
                col = species_info.loc[species_i, 'color']
                mathtext = species_info.loc[species_i, 'mathtext_name']

            # Plot new line with alpha=1
            if single_transit_only:
                # Remove previously plotted CCF lines for this species
                for old_line in lines[species_i]:
                    old_line.remove()
                lines[species_i].clear()

                line, = ax[i].plot(
                    cc_obj.vsys_list,
                    ccf,
                    c=col,
                    alpha=1,
                    lw=1.5
                )
                lines[species_i].append(line)

            else:
                line, = ax[i].plot(cc_obj.vsys_list, ccf, c=col, alpha=1)
                lines[species_i].append(line)

                # Fade older curves
                for old_line in lines[species_i][:-1]:
                    old_line.set_alpha(0.2)

            # Update S/N text
            SNR = snr_store[species_i][n_trans]
            label = f'{mathtext}\nS/N={np.round(SNR, 1)}'

            if text_artists[i] is not None:
                text_artists[i].remove()

            text_artists[i] = ax[i].text(
                0.05, 0.9, label,
                transform=ax[i].transAxes,
                fontsize=10,
                verticalalignment='top',
                color=col,    
                bbox=dict(
                    facecolor=(1, 1, 1, 0.7),   # white, alpha=0.5
                    edgecolor=(0, 0, 0, 0.5),   # black, alpha=0.5
                    boxstyle='round,pad=0.3',
                    linewidth=0.8
                )
                        )

        fig.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        return []

    if animate:
        anim = FuncAnimation(fig, update,
                             frames=tr,
                             interval=1000 / fps,
                             blit=False)

        fname = f'{cc_obj.figs_dir}/CCFs_1D_n1-{tr}.gif'
        print('Saving GIF animation as', fname)

        writer = PillowWriter(fps=fps)
        anim.save(fname, writer=writer, dpi=200)

    else:
        for frame in range(tr):
            update(frame)

        fig.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)

        if len(species) == 1:
            sp=f'_{species[0]}'
        elif len(species) == 7:
            sp='_all'
        else:
            sp=''
        ab = f'1e{int(abund)}' if abund!='best' else 'best'
        tran = single_n_trans if single_transit_only else f'1-{tr}'
        fname = f'{cc_obj.figs_dir}/CCFs_1D_n{tran}{sp}_{ab}.pdf'
        print('Saving as', fname)
        fig.savefig(fname, bbox_inches='tight')

    plt.close()

def multi_transit_abundance_map(system_obj,
                                wl_obs,
                                fl_array,
                                err_array,
                                species=['C2H2', 'CH4'],
                                abunds=[-2, -10],
                                transits=[1, 2],
                                sysit=1,
                                cmap=custom_cmap,
                                overwrite=False,
                                contour_map=False):
    """
    Produce 2D S/N maps (abundance vs number of transits) for multiple species.
    """

    outdir = pathlib.Path(f'{system_obj.project_path}/SNR_maps')
    outdir.mkdir(parents=True, exist_ok=True)
    cache_file = outdir / f"SNR_maps_tr{max(transits)}_ab{int(min(abunds))}-{int(max(abunds))}.npz"

    species_info = pd.read_csv('species_info.csv', index_col=0)

    if cache_file.exists() and not overwrite:
        print(f"Loading cached S/N maps from {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        SNR_maps = data['SNR_maps'].item()
        abunds = data['abunds']
        transits = data['transits']
        species = data['species']
    else:
        print("Running cross-correlations...")

        cc_obj = CrossCorr(system_obj,
                            wl_obs,
                            fl_array,
                            err_array,
                            sysrem_iter=sysit,
                            use_transits=max(transits),
                            plot_order='all')

        flux_mask2 = cc_obj.flux_mask2
        fluxerr_mask2 = cc_obj.fluxerr_mask2

        SNR_maps = {s: np.zeros((len(abunds), len(transits))) for s in species}

        for si, species_i in enumerate(species):
            print(f"\nSpecies: {species_i}")

            for ai, abund in enumerate(abunds):
                print(f"  log abundance = {abund}")

                # run once for MAX number of transits
                cc_obj.n_trans = max(transits)
                cc_obj.cross_correlate(
                    flux_mask2,
                    fluxerr_mask2,
                    template=species_i,
                    abund=abund,
                    plot_masked_contributions=False
                )

                maps_per_transit = cc_obj.Kp_vsys_maps_per_transit
                Kp_idx = find_nearest(cc_obj.Kp_list, value=cc_obj.Kp.value)

                for ti, n_trans in enumerate(transits):
                    ccf_1d = np.sum(
                        maps_per_transit[:n_trans],
                        axis=0
                    )[Kp_idx]

                    snr = local_peak_snr(
                        cc_obj.vsys_list,
                        ccf_1d,
                        v0=0.0,
                        dv=1.0
                    )

                    SNR_maps[species_i][ai, ti] = snr

        np.savez(
            cache_file,
            SNR_maps=SNR_maps,
            abunds=np.array(abunds),
            transits=np.array(transits),
            species=np.array(species),
        )
        print(f"Saved S/N maps to {cache_file}")

    fig, axes = plt.subplots(1, len(species), figsize=(9.5, 4), dpi=200, sharey=True)

    for i, species_i in enumerate(species):
        ax = axes[i]
        snr_map = SNR_maps[species_i]

        title = species_info.loc[species_i, 'mathtext_name']
        species_color = species_info.loc[species_i, 'color']
        cmap_i = species_cmap_new(species_color)

        if contour_map:

            # Build grid for contours
            X, Y = np.meshgrid(transits, abunds)

            # Filled contour map
            im = ax.contourf(X,Y,snr_map,
                levels=10,           # smoothness of the map
                cmap=cmap_i,
                antialiased=False)

            # White S/N = 5 contour
            cs = ax.contour(X,Y,snr_map,levels=[5],colors='white',linewidths=2.0)

            # Label the contour
            ax.clabel(
                cs,
                fmt={5: 'S/N = 5'},
                inline=True,
                fontsize=9,
                colors='white'
            )

        else:
            im = ax.imshow(
                snr_map,
                origin='lower',
                aspect='auto',
                cmap=cmap_i,
                extent=[
                    min(transits) - 0.5,
                    max(transits) + 0.5,
                    abunds[0] - 0.5,
                    abunds[-1] + 0.5,
                ]
            )
            # detection outline
            label_positions = draw_detection_outline(
                ax,
                snr_map,
                transits,
                abunds,
                level=5.0
            )
            # --- invisible contour for labeling only ---
            X, Y = np.meshgrid(transits, abunds)

            cs = ax.contour(
                X,
                Y,
                snr_map,
                levels=[5.0],
                colors='none'   # invisible contour, label only
            )

            texts = ax.clabel(
                cs,
                fmt={5.0: 'S/N = 5'},
                inline=True,
                fontsize=9,
                colors='white'
            )

            # --- add semi-transparent black box behind the label ---
            for txt in texts:
                txt.set_bbox(dict(
                    facecolor='black',
                    edgecolor='none',
                    alpha=0.6,
                    boxstyle='round,pad=0.25'
                ))

        # --- overlay full white if no S/N ≥ 5 --- 
        if np.all(snr_map < 5.0):
            rect = Rectangle( (min(transits) - 0.5, abunds[0] - 0.5),
                              width=max(transits) - min(transits) + 1,  
                              height=abunds[-1] - abunds[0] + 1,  
                              facecolor='k', alpha=0.6, zorder=20)
            ax.add_patch(rect)

        ax.set_xticks(transits)
        ax.set_yticks(abunds)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Number of transits')

        if i == 0:
            ax.set_ylabel(r'$\log_{10}$ abundance')
        else:
            ax.tick_params(labelleft=False)

        cb = fig.colorbar(
            im,
            ax=ax,
            orientation='horizontal',
            pad=0.18,        
            fraction=0.08   # optional: thickness of colorbar
        )
        cb.set_label('S/N', fontsize=9)
        cb.ax.tick_params(labelsize=8)

    # --- global title ---
    label = system_obj.parameters['label']
    fig.suptitle(label, fontsize=14,y=0.95)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.05,hspace=0)

    sp = '_all' if len(species) == 6 else '_few'
    fname = outdir / f"SNR_2D_abundance{int(min(abunds))}-{int(max(abunds))}_vs_transits{max(transits)}{sp}.pdf"
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved figure to {fname}")


def draw_detection_outline(ax, snr_map, x_centers, y_centers,
                           level=5.0, color='white', lw=2,
                           return_label_pos=True):
    """
    Draw an edge-aligned outline around regions where snr_map >= level.
    Optionally return a good position for inline labeling.
    """

    x = np.array(x_centers)
    y = np.array(y_centers)
    Z = snr_map

    dx = np.diff(x).mean() if len(x) > 1 else 1.0
    dy = np.diff(y).mean() if len(y) > 1 else 1.0

    x_edges = np.concatenate(([x[0] - dx/2], x + dx/2))
    y_edges = np.concatenate(([y[0] - dy/2], y + dy/2))

    ny, nx = Z.shape
    label_candidates = []

    for iy in range(ny):
        for ix in range(nx):

            if Z[iy, ix] < level:
                continue

            x0, x1 = x_edges[ix], x_edges[ix + 1]
            y0, y1 = y_edges[iy], y_edges[iy + 1]

            # left
            if ix == 0 or Z[iy, ix - 1] < level:
                ax.plot([x0, x0], [y0, y1], color=color, lw=lw)
                label_candidates.append((x0, 0.5 * (y0 + y1)))

            # right
            if ix == nx - 1 or Z[iy, ix + 1] < level:
                ax.plot([x1, x1], [y0, y1], color=color, lw=lw)
                label_candidates.append((x1, 0.5 * (y0 + y1)))

            # bottom
            if iy == 0 or Z[iy - 1, ix] < level:
                ax.plot([x0, x1], [y0, y0], color=color, lw=lw)
                label_candidates.append((0.5 * (x0 + x1), y0))

            # top
            if iy == ny - 1 or Z[iy + 1, ix] < level:
                ax.plot([x0, x1], [y1, y1], color=color, lw=lw)
                label_candidates.append((0.5 * (x0 + x1), y1))

    if return_label_pos and label_candidates:
        # pick a central-ish candidate
        return np.array(label_candidates)

    return None

from matplotlib.patches import Rectangle

def overlay_no_detection(ax, snr_map, transits, abunds, alpha=0.5):
    """
    Overlay semi-transparent white rectangles where S/N < 5.
    """
    dx = 1  # assuming your plotting x-axis is in units of number of transits
    dy = 1  # assuming your y-axis is abundance steps

    # loop over grid
    for i, a in enumerate(abunds):
        for j, t in enumerate(transits):
            if snr_map[i, j] < 5.0:
                # bottom-left corner of rectangle
                x0 = t - 0.5
                y0 = a - 0.5
                rect = Rectangle(
                    (x0, y0),
                    width=dx,
                    height=dy,
                    facecolor='white',
                    alpha=alpha,
                    edgecolor=None,
                    zorder=15  # above the heatmap
                )
                ax.add_patch(rect)

def retrieve_best_vmrs(system_obj,
                       snr_threshold=5.0,
                       default_vmr=1e-2,
                       c2h2_default=1e-10):
    """
    Retrieve best-fit VMR per species from cached 2D S/N maps.

    If no cache file exists:
      - return default_vmr for all species
      - except C2H2, which gets c2h2_default
    """

    outdir = pathlib.Path(f'{system_obj.project_path}/SNR_maps')
    cache_files = sorted(outdir.glob("SNR_maps_tr*_ab*.npz"))

    # ---------- CASE 1: no cache exists ----------
    if len(cache_files) == 0:
        print("No S/N cache file found — returning default VMRs.")

        # we still need to know which species exist
        # fall back to species_info.csv
        species_info = np.loadtxt(
            'species_info.csv',
            delimiter=',',
            dtype=str,
            skiprows=1,
            usecols=0
        )

        best_vmrs = {}
        best_snrs = {}

        for sp in species_info:
            if sp == 'C2H2':
                best_vmrs[sp] = c2h2_default
            else:
                best_vmrs[sp] = default_vmr
            best_snrs[sp] = 0.0

        return best_vmrs, best_snrs

    # ---------- CASE 2: cache exists ----------
    cache_file = cache_files[-1]
    print(f"Loading S/N maps from {cache_file}")

    data = np.load(cache_file, allow_pickle=True)
    SNR_maps = data['SNR_maps'].item()
    abunds = data['abunds']      # log10(VMR)
    species = data['species']

    best_vmrs = {}
    best_snrs = {}

    for sp in species:
        snr_map = SNR_maps[sp]
        max_snr = np.nanmax(snr_map)
        best_snrs[sp] = max_snr

        if max_snr >= snr_threshold:
            ai, ti = np.unravel_index(np.nanargmax(snr_map), snr_map.shape)
            best_vmrs[sp] = 10.0 ** abunds[ai]
        else:
            if sp == 'C2H2':
                best_vmrs[sp] = c2h2_default
            else:
                best_vmrs[sp] = default_vmr

    return best_vmrs, best_snrs

def species_cmap(color, name='species_cmap', white_mix=0.3):
    """
    Create a colormap going from black → species color → slightly whitened color.

    Parameters
    ----------
    color : str or tuple
        Base species color (hex or RGB).
    name : str
        Name of the colormap.
    white_mix : float
        How much white to mix into the top color (0–1).
    """
    base = np.array(to_rgb(color))
    top = base * (1 - white_mix) + np.ones(3) * white_mix

    return LinearSegmentedColormap.from_list(
        name,
        [(0.0, 'black'),
         (1.0, top)]
    )

def species_cmap_new(
    base_color,
    name='species_cmap',
    n=256,
    dark_frac=0.08,
    light_boost=1.6,
):
    """
    Create a vivid species-specific colormap:
    - preserves hue
    - keeps saturation high
    - increases brightness without washing out (no white mixing)

    Parameters
    ----------
    base_color : str or tuple
        Matplotlib color (hex or RGB)
    dark_frac : float
        How dark the low end goes (0–1)
    light_boost : float
        >1 makes the high end brighter but still saturated
    """

    # Convert base color to RGB
    rgb = mcolors.to_rgb(base_color)

    # Convert to HLS
    h, l, s = colorsys.rgb_to_hls(*rgb)

    # Build lightness ramp (nonlinear = more interesting)
    t = np.linspace(0, 1, n)
    t = t**0.7  # emphasize high end contrast

    # Lightness: dark → original → brighter
    l_min = dark_frac * l
    l_max = min(0.95, l * light_boost)

    lightness = l_min + t * (l_max - l_min)

    # Slight saturation boost toward the top
    saturation = np.clip(s * (0.9 + 0.3 * t), 0, 1)

    colors = [
        colorsys.hls_to_rgb(h, li, si)
        for li, si in zip(lightness, saturation)
    ]

    return mcolors.LinearSegmentedColormap.from_list(name, colors)