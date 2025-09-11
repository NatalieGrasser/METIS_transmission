import os
os.environ["SCOPESIM_INSTRUMENTS"] = '/net/lem/data1/grasser/ScopeSim_Templates'
import scopesim as sim
import scopesim_templates as sim_tp
import numpy as np
import astropy.units as u
import pathlib
from astropy.io import fits
from scopesim.utils import find_file
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

class METIS:

    def __init__(self,central_wavelength,system_obj):

        self.project_path = system_obj.project_path
        self.exptime= system_obj.exptime
        self.planet_wl_obs_range = system_obj.planet_wl_obs_range
        self.cmd = sim.UserCommands(use_instrument="METIS",
                       set_modes=["lms"],
                       properties={"!OBS.wavelen": central_wavelength})
        self.metis = sim.OpticalTrain(self.cmd)
        
    def observe_transit_calib(self,transit_flux_array,output_dir,plot_exp=None):

        self.output_dir = pathlib.Path(f'{self.project_path}/METIS_data/{output_dir}')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figs_dir = pathlib.Path(f'{self.project_path}/figures/METIS_output')
        self.figs_dir.mkdir(parents=True, exist_ok=True)

        transit_simulated_frames = []
        for i,fl in enumerate(transit_flux_array):
            
            hdul_src_path = pathlib.Path(f'{self.output_dir}/transit_frame_{int(i)}.fits')
            
            if hdul_src_path.exists():
                hdul_src = fits.open(hdul_src_path)
            else:
                src = sim.Source(x=[0],y=[0],ref=[0], lam=self.planet_wl_obs_range, spectra=fl) 
                self.metis.observe(src)
                hdul_src = self.metis.readout(exptime=self.exptime.value)[0]
                dit  = self.metis.cmds.get("!OBS.dit", self.exptime)
                ndit = self.metis.cmds.get("!OBS.ndit", 1)
                hdr = hdul_src[1].header
                hdr["EXPTIME"] = (self.exptime.value, "Total exposure time [s]")
                hdr["DIT"]     = (dit, "Detector integration time [s]")
                hdr["NDIT"]    = (ndit, "Number of DITs")
                hdul_src.writeto(hdul_src_path, overwrite=True)
            transit_simulated_frames.append(hdul_src)
            
        self.hdr = hdul_src[1].header

        # dark current
        master_dark_path = pathlib.Path(f'{self.output_dir}/master_dark.fits')

        if master_dark_path.exists():
            master_dark_hdul = fits.open(master_dark_path)
        else:
            n_dark = 5 
            dark_stack = []
            for i in range(n_dark):
                self.metis["skycalc_atmosphere"].include = False  # no atmosphere
                self.metis.observe(sim_tp.darkness()) 
                hdul_dark = self.metis.readout(exptime=self.exptime.value)[0]
                dark_stack.append([hdul_dark[j].data.astype(float) for j in range(1, 5)]) # all 4 detector planes

            # combine per-detector (median across exposures)
            master_dark_per_det = [np.median([stack[i] for stack in dark_stack], axis=0) for i in range(4)]
            
            # build HDUList: primary + 4 detectors
            hdrs = [hdul_dark[j].header.copy() for j in range(1, 5)]
            master_dark_hdul = fits.HDUList([hdul_dark[0]])  # primary
            for i in range(4):
                master_dark_hdul.append(fits.ImageHDU(master_dark_per_det[i], header=hdrs[i], name=f"DET{i+1}.DATA"))

            master_dark_hdul.writeto(master_dark_path, overwrite=True)

        # dark current
        master_dark_path = pathlib.Path(f'{self.output_dir}/master_dark.fits')

        if master_dark_path.exists():
            master_dark_hdul = fits.open(master_dark_path)
        else:
            n_dark = 5 
            dark_stack = []
            for i in range(n_dark):
                self.metis["skycalc_atmosphere"].include = False  # no atmosphere
                self.metis.observe(sim_tp.darkness()) 
                hdul_dark = self.metis.readout(exptime=self.exptime.value)[0]
                dark_stack.append([hdul_dark[j].data.astype(float) for j in range(1, 5)]) # all 4 detector planes

            # combine per-detector (median across exposures)
            master_dark_per_det = [np.median([stack[i] for stack in dark_stack], axis=0) for i in range(4)]
            
            # build HDUList: primary + 4 detectors
            hdrs = [hdul_dark[j].header.copy() for j in range(1, 5)]
            master_dark_hdul = fits.HDUList([hdul_dark[0]])  # primary
            for i in range(4):
                master_dark_hdul.append(fits.ImageHDU(master_dark_per_det[i], header=hdrs[i], name=f"DET{i+1}.DATA"))

            master_dark_hdul.writeto(master_dark_path, overwrite=True)
            
        # flat-field
        master_flat_path = pathlib.Path(f'{self.output_dir}/master_flat.fits')

        if master_flat_path.exists():
            master_flat_hdul = fits.open(master_flat_path)
        else:
            n_flat = 5
            flat_stack = []
            for i in range(n_flat):
                self.metis["skycalc_atmosphere"].include = False  # no atmosphere
                self.metis.observe(sim_tp.flatlamp())
                hdul_flat = self.metis.readout(exptime=10)[0]
                flat_stack.append([hdul_flat[j].data.astype(float) for j in range(1, 5)]) # all 4 detectors

            master_flat_per_det = []
            for i in range(4):
                # combine per-detector (median across exposures)
                flat_stack_i = np.stack([stack[i] for stack in flat_stack], axis=0)  # shape (nflat, ny, nx)
                flat = np.median(flat_stack_i, axis=0)  # median combine exposures

                # normalize per wavelength (column-wise)
                med_per_col = np.median(flat, axis=0)  # axis=0 = spatial
                flat /= med_per_col  # broadcast division
                master_flat_per_det.append(flat)

            # build HDUList: primary + 4 detectors
            hdrs = [hdul_flat[j].header.copy() for j in range(1, 5)] # 4-detector HDUList
            master_flat_hdul = fits.HDUList([hdul_flat[0]])  # primary
            for i in range(4):
                master_flat_hdul.append(fits.ImageHDU(master_flat_per_det[i], header=hdrs[i], name=f"DET{i+1}.DATA"))

            master_flat_hdul.writeto(master_flat_path, overwrite=True)
            
        # empty sky
        hdul_sky_path = pathlib.Path(f'{self.output_dir}/hdul_sky.fits')

        if hdul_sky_path.exists():
            hdul_sky = fits.open(hdul_sky_path)
        else:
            self.metis["skycalc_atmosphere"].include = True # with atmosphere
            self.metis.observe(sim_tp.empty_sky())
            hdul_sky = self.metis.readout(exptime=self.exptime.value)[0]
            hdul_sky.writeto(hdul_sky_path, overwrite=True)

        if plot_exp is not None: # plot exposure
            self.plot_raw_whitelight(transit_simulated_frames[plot_exp],master_flat_hdul,
                                     master_dark_hdul,hdul_sky,plot_exp=plot_exp)

        return transit_simulated_frames, master_dark_hdul, master_flat_hdul, hdul_sky
    
    def rectify_calibrate_extract(self, 
                                  transit_simulated_frames, master_dark_hdul, 
                                  master_flat_hdul, hdul_sky, plot_exp=None):

        def get_rectified(target,hdul_raw):
            rect_path = pathlib.Path(f'{self.output_dir}/rectified_{target}.fits')
            if rect_path.exists():
                rect = fits.open(rect_path)[1]
            else:
                rect = self.metis["lms_spectral_traces"].rectify_cube(hdul_raw)
                rect.writeto(rect_path, overwrite=True)
            return rect
                
        rect_sky = get_rectified('sky',hdul_sky)
        rect_dark = get_rectified('dark',master_dark_hdul)
        rect_flat = get_rectified('flat',master_flat_hdul)

        rect_src_frames = []
        for i,hdul_src in enumerate(transit_simulated_frames):
            #print(i)
            rect_src = get_rectified(f'src_{i}',hdul_src)
            rect_src_frames.append(rect_src)

        # just take wavelength data from the first, is same for all
        self.hdr['CRVAL3'] = rect_src.header['CRVAL3']
        self.hdr['CRPIX3'] = rect_src.header['CRPIX3']
        self.hdr['CDELT3'] = rect_src.header['CDELT3']

        del transit_simulated_frames # memory issues

        # remove edge pixels (near-zero flux)
        clip_x = 2 
        sky = rect_sky.data.astype(float)[:, :, clip_x:-clip_x]
        dark = rect_dark.data.astype(float)[:, :, clip_x:-clip_x]
        flat = rect_flat.data.astype(float)[:, :, clip_x:-clip_x]

        def get_wavelegth_from_header(rect, header):
            crval = float(header['CRVAL3'])
            crpix = float(header['CRPIX3'])
            cdelt = float(header['CDELT3'])
            nwave = rect.shape[0]
            wl = crval + (np.arange(1, nwave+1) - crpix) * cdelt
            return wl, nwave

        def get_quantum_efficiency(metis, wave_array):
            wave_array = np.array(wave_array)
            qe_file_rel = metis.cmds["!DET.qe_curve"]["file_name"]
            qe_file = find_file(qe_file_rel)
            wl, qe = np.loadtxt(qe_file, unpack=True, comments="#",skiprows=15)
            qe_interp = np.interp(wave_array, wl, qe)
            return qe_interp

        def compute_variance_from_rectified_cube(metis, rectified, wavelengths):

            data_adu = rectified.data  # shape (nwave, ny, nx)
            exptime = metis.cmds["!OBS.exptime"]
            ndit = metis.cmds.get("!OBS.ndit", 1)
            gain = 2.0  # e-/ADU
            dark_current = metis.cmds.get("!DET.dark_current", 0.1)  # e-/s
            read_noise = metis.cmds.get("!DET.readout_noise", 70.0)  # e-/DIT
            
            qe = get_quantum_efficiency(metis,wavelengths)  # shape: (nwave,)
            qe_cube = qe[:, np.newaxis, np.newaxis] # reshape for broadcasting: (nwave, 1, 1)
            data_e = data_adu * gain * ndit # photon noise in electrons
            photon_var_e = data_e * qe_cube  # poisson processes (photon arrival): variance = expected number of counts
            dark_var_e = dark_current * exptime * ndit # dark noise
            read_var_e = (read_noise**2) * ndit # readout noise
            total_var_e = photon_var_e + dark_var_e + read_var_e # total variance in electrons
            var_cube = total_var_e / (gain**2) # convert back to ADU^2
            
            return var_cube
    
        # remove edge pixels (near-zero flux)
        clip_x = 2 
        sky = rect_sky.data.astype(float)[:, :, clip_x:-clip_x]
        dark = rect_dark.data.astype(float)[:, :, clip_x:-clip_x]
        flat = rect_flat.data.astype(float)[:, :, clip_x:-clip_x]

        def get_quantum_efficiency(metis, wave_array):
            wave_array = np.array(wave_array)
            qe_file_rel = metis.cmds["!DET.qe_curve"]["file_name"]
            qe_file = find_file(qe_file_rel)
            wl, qe = np.loadtxt(qe_file, unpack=True, comments="#",skiprows=15)
            qe_interp = np.interp(wave_array, wl, qe)
            return qe_interp

        def compute_variance_from_rectified_cube(metis, rectified,wavelengths):

            data_adu = rectified.data  # shape (nwave, ny, nx)
            exptime = metis.cmds["!OBS.exptime"]
            ndit = metis.cmds.get("!OBS.ndit", 1)
            gain = 2.0  # e-/ADU
            dark_current = metis.cmds.get("!DET.dark_current", 0.1)  # e-/s
            read_noise = metis.cmds.get("!DET.readout_noise", 70.0)  # e-/DIT
            
            qe = get_quantum_efficiency(metis,wavelengths)  # shape: (nwave,)
            qe_cube = qe[:, np.newaxis, np.newaxis] # reshape for broadcasting: (nwave, 1, 1)
            data_e = data_adu * gain * ndit # photon noise in electrons
            photon_var_e = data_e * qe_cube  # poisson processes (photon arrival): variance = expected number of counts
            dark_var_e = dark_current * exptime * ndit # dark noise
            read_var_e = (read_noise**2) * ndit # readout noise
            total_var_e = photon_var_e + dark_var_e + read_var_e # total variance in electrons
            var_cube = total_var_e / (gain**2) # convert back to ADU^2
            
            return var_cube

        calib_frames = []
        var_cubes = []
        for i,rect_src in enumerate(rect_src_frames):
            if i==0:
                wavelengths, nwave = get_wavelegth_from_header(rect_src, self.hdr)
            src = rect_src.data.astype(float)[:, :, clip_x:-clip_x]
            src_dark = src - dark
            sky_dark = sky - dark
            src_dark_flat =  src_dark / flat
            sky_dark_flat =  sky_dark / flat
            src_minus_sky = src_dark_flat - sky_dark_flat

            if plot_exp is not None:
                if i==plot_exp:
                    self.plot_rectified(src,dark,src_dark_flat,flat,src_minus_sky,sky_dark_flat,plot_exp)

            calib_frames.append(src_minus_sky)
            var_cube = compute_variance_from_rectified_cube(self.metis, rect_src, wavelengths)[:, :, clip_x:-clip_x]
            var_cubes.append(var_cube)

        def find_and_extract_trace(cube, var_cube=None, wavelengths=None,
                                aperture_halfwidth=3, 
                                smooth_sigma=2.0, signal_frac_thresh=0.3, 
                                poly_order=3, plot_diagnostics=False, plot_exp=None):
            """
            Find spectral trace in a rectified cube and extract a 1D spectrum.
            cube shape: (nwave, ny, nx) where axis0 is wavelength.
            var_cube: optional variance cube (same shape as cube) from simulation.
            header: rectified.header (used to build wavelength array if desired).
            Returns: wl (µm array or None), spec (1D), spec_err (1D), trace_y (centroid per wl)
            """

            wl = wavelengths
            white = np.nanmedian(cube, axis=0) # white-light image
            ny, nx = cube.shape[1], cube.shape[2]

            # centroid & signal detection
            frac_nonzero = np.zeros(nwave)
            peak_y = np.zeros(nwave, dtype=float)
            maxvals = np.zeros(nwave)
            for i in range(nwave):
                profile_y = np.nansum(cube[i,:,:], axis=1)
                # negative values from background subtraction or noise -> just normalize
                profile_y -= np.nanmin(profile_y)   # shift baseline to zero
                if np.nanmax(profile_y) > 0: # avoid division by 0
                    profile_y /= np.nanmax(profile_y)

                # --- after normalization ---
                maxvals[i] = np.nanmax(profile_y)   # will always be 1.0 if not flat
                peak_y[i] = np.nanargmax(profile_y)
                frac_nonzero[i] = np.sum(profile_y > 0.3) / float(len(profile_y))  

            sig_mask = (maxvals > 0.5) | (frac_nonzero > signal_frac_thresh)

            # 3) fit trace
            good_idxs = np.where(sig_mask)[0]
            if len(good_idxs) == 0:
                raise RuntimeError("No trace slices found by threshold heuristics. Lower thresholds or inspect white image.")
            centroids_smooth = gaussian_filter(peak_y[good_idxs], sigma=smooth_sigma)
            p = np.polyfit(good_idxs, centroids_smooth, deg=poly_order)
            trace_y = np.polyval(p, np.arange(nwave))
                
            if plot_exp is not None and plot_diagnostics:
                fig = plt.figure(figsize=(5,2))
                im = plt.imshow(white, origin='lower', aspect='auto', cmap='inferno')
                # overlay predicted trace
                xs = np.linspace(0, nx-1, nx)
                ys = trace_y.mean()  # just to set scale
                # trace drawn as points for each wavelength projected onto (ny,nx) image
                # we need to map wavelength index to x coordinate for plotting; we use linear mapping:
                wl_idx_to_x = np.linspace(0, nx-1, nwave)
                yplot = trace_y
                plt.scatter(wl_idx_to_x, yplot, s=1, c='cyan', label='trace (approx)')
                plt.title('White light with fitted trace (approx mapping)')
                plt.colorbar(im)
                plt.legend()
                fig.savefig(f'{self.project_path}/figures/METIS_output/extract_trace_{plot_exp}.pdf', bbox_inches='tight')
                plt.close()

            # 4) extract spectrum (updated)
            spec = np.zeros(nwave)
            spec_err = np.zeros(nwave)
            min_flux_frac = 0.6  # fraction of peak to include in aperture

            for i in range(nwave):
                ycen = trace_y[i]
                if not np.isfinite(ycen):
                    spec[i] = np.nan
                    spec_err[i] = np.nan
                    continue

                y0 = int(np.round(ycen))
                ylo = max(0, y0 - aperture_halfwidth)
                yhi = min(ny, y0 + aperture_halfwidth + 1)
                slice_ = cube[i, ylo:yhi, :]

                # robust: only include pixels above a fraction of the peak
                peak_val = np.nanmax(slice_)
                mask = slice_ >= (peak_val * min_flux_frac)
                valid_pixels = slice_[mask]

                if valid_pixels.size == 0:
                    spec[i] = np.nan
                    spec_err[i] = np.nan
                else:
                    spec[i] = np.nansum(valid_pixels)
                    if var_cube is not None:
                        spec_err[i] = np.sqrt(np.nansum(var_cube[i, ylo:yhi, :][mask]))
                    else:
                        spec_err[i] = np.sqrt(np.nansum((valid_pixels - np.nanmedian(valid_pixels))**2))

            sig_mask &= (spec>0.0) # mask negative values
            spec[~sig_mask] = np.nan
            spec_err[~sig_mask] = np.nan

            return wl, spec, spec_err, trace_y

        fluxes_obs = []
        err_obs = []
        for i,calib in enumerate(calib_frames):

            if i==plot_exp:
                plot_diagnostics = True
            else:
                plot_diagnostics = False
            wl_obs, flx_obs, flx_obs_err, _ = find_and_extract_trace(calib, 
                                                            var_cube=var_cubes[i], 
                                                            wavelengths=wavelengths,
                                                            plot_diagnostics=plot_diagnostics,
                                                            plot_exp=plot_exp)
            fluxes_obs.append(flx_obs)
            err_obs.append(flx_obs_err)

        wl_obs = np.array(wl_obs)
        fluxes_obs = np.array(fluxes_obs)
        err_obs = np.array(err_obs)

        if plot_exp is not None:
            self.plot_extracted_spetrum(wl_obs,fluxes_obs[plot_exp],
                                        err_obs[plot_exp],plot_exp=plot_exp)

        return wl_obs, fluxes_obs, err_obs

    def plot_raw_whitelight(self,hdul_src,master_flat_hdul,master_dark_hdul,hdul_sky,plot_exp=''):
        # Compare white-light images before rectification
        x = 1
        src_white_raw  = hdul_src[x].data.astype(float)
        flat_raw= master_flat_hdul[x].data.astype(float)
        dark_raw= master_dark_hdul[x].data.astype(float)
        sky_white_raw= hdul_sky[x].data.astype(float)

        fig, ax = plt.subplots(2,2,figsize=(7,6))
        im0 = ax[0,0].imshow(src_white_raw, vmin=0, vmax=np.percentile(src_white_raw, 99), aspect='auto')
        ax[0,0].set_title("Science (raw white-light)")
        plt.colorbar(im0,ax=ax[0,0])
        im1 = ax[1,0].imshow(flat_raw, vmin=0,vmax=np.nanpercentile(flat_raw,99), aspect='auto')
        ax[1,0].set_title("Flat (raw)")
        plt.colorbar(im1,ax=ax[1,0])
        im2 = ax[0,1].imshow(sky_white_raw, vmin=0, vmax=np.percentile(sky_white_raw, 99), aspect='auto')
        ax[0,1].set_title("Sky (raw white-light)")
        plt.colorbar(im2,ax=ax[0,1])
        im3 = ax[1,1].imshow(dark_raw, vmin=0, vmax=np.percentile(dark_raw, 99), aspect='auto')
        ax[1,1].set_title("Dark (raw)")
        plt.colorbar(im3,ax=ax[1,1])
        fig.tight_layout()
        fig.savefig(f'{self.project_path}/figures/METIS_output/raw_whitelight_{plot_exp}.pdf', bbox_inches='tight')
        plt.close()

    def plot_rectified(self,src,dark,src_dark_flat,flat,src_minus_sky,sky_dark_flat,plot_exp):
        fig, ax = plt.subplots(3, 2, figsize=(7,7), sharex=True, sharey=True)
        im = ax[0,0].imshow(np.nanmedian(src, axis=0), aspect='auto')
        fig.colorbar(im, ax=ax[0,0]); ax[0,0].set_title("Science (rectified)")
        im = ax[0,1].imshow(np.nanmedian(dark, axis=0), aspect='auto')
        fig.colorbar(im, ax=ax[0,1]); ax[0,1].set_title("Master dark (rectified)")
        im = ax[1,0].imshow(np.nanmedian(src_dark_flat, axis=0), aspect='auto')
        fig.colorbar(im, ax=ax[1,0]); ax[1,0].set_title("Dark+flat corrected")
        im = ax[1,1].imshow(np.nanmedian(flat, axis=0), aspect='auto')
        fig.colorbar(im, ax=ax[1,1]); ax[1,1].set_title("Flat (rectified)")
        im = ax[2,0].imshow(np.nanmedian(src_minus_sky, axis=0), aspect='auto')
        fig.colorbar(im, ax=ax[2,0]); ax[2,0].set_title("Source - sky (calibrated)")
        im = ax[2,1].imshow(np.nanmedian(sky_dark_flat, axis=0), aspect='auto')
        fig.colorbar(im, ax=ax[2,1]); ax[2,1].set_title("Sky (calibrated)")
        plt.tight_layout()
        fig.savefig(f'{self.project_path}/figures/METIS_output/rectified_{plot_exp}.pdf', bbox_inches='tight')
        plt.close()

    def plot_extracted_spetrum(self,wl,fl,err,plot_exp=''):
        fig = plt.figure(figsize=(6,2.5),dpi=150)
        plt.plot(wl,fl,lw=0.8)
        plt.fill_between(wl, fl - err, fl+err, color='gray', alpha=0.3)
        plt.xlabel('Wavelength [µm]')
        plt.ylabel('Extracted flux')
        fig.savefig(f'{self.project_path}/figures/METIS_output/extracted_spectrum_{plot_exp}.pdf', bbox_inches='tight')
        plt.close()