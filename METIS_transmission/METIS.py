import os
os.environ["SCOPESIM_INSTRUMENTS"] = '/net/lem/data1/grasser/ScopeSim_Templates'
import scopesim as sim
import scopesim_templates as sim_tp
import numpy as np
import astropy.units as u
import pathlib
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.ticker import ScalarFormatter
from scipy.ndimage import gaussian_filter1d
from astropy.io import fits
from scopesim.utils import find_file
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

class METIS:

    def __init__(self,central_wavelength,system_obj,order,n_transit):

        self.project_path = system_obj.project_path
        self.exptime= system_obj.exptime
        self.planet_wl_obs_range = system_obj.planet_wl_obs_range
        self.cmd = sim.UserCommands(use_instrument="METIS",
                       set_modes=["lms"],
                       properties={"!OBS.wavelen": central_wavelength})
        if 'pwv' in system_obj.parameters:
            self.cmd['!ATMO']['pwv'] = system_obj.parameters['pwv'] # default: 2.5
        if 'airmass' in system_obj.parameters:
            self.cmd['!OBS']['airmass'] = system_obj.parameters['airmass'] # default: 1.2

        # default seeing: 'PSF_LM_9mag_06seeing.fits'
        # PSF model with a 0.6 arcsecond seeing for a 9th magnitude star
        #print("cmd['OBS!']['psf_file']",self.cmd['OBS!']['psf_file'])

        self.metis = sim.OpticalTrain(self.cmd)
        self.transit_flux_array = system_obj.transit_flux_array

        self.order = order
        self.output_dir = pathlib.Path(f'{self.project_path}/METIS_data/transit{n_transit+1}/order{order}')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figs_dir = pathlib.Path(f'{self.project_path}/figures/METIS_output/transit{n_transit+1}')
        self.figs_dir.mkdir(parents=True, exist_ok=True)

    def get_observations(self,plot_exp=None,delete_tmp_files=True,delete_all=True):

        timeseries_extracted = pathlib.Path(f'{self.output_dir}/timeseries_{self.order}.npz')
        if timeseries_extracted.exists():
            data = np.load(timeseries_extracted)
            wl  = data["wl"]
            fl  = data["fl"]
            err = data["err"]
        else:
            transit, dark, flat, sky = self.observe_transit_calib(self.transit_flux_array,plot_exp=plot_exp)
            wl, fl, err = self.rectify_calibrate_extract(transit, dark, flat, sky, plot_exp=plot_exp)
            np.savez(timeseries_extracted, wl=wl, fl=fl, err=err)
        print('Observation-time series is ready.')

        # delete temporary files (observations & rectified), take up too much space
        if delete_tmp_files and timeseries_extracted.exists(): 
    
            transit_frames = [f for f in self.output_dir.iterdir() if f.name.startswith('transit_frame_')]
            
            if len(transit_frames)>1:
                print('Deleting raw transit frames')
                for file in transit_frames:
                    if file.name != 'transit_frame_0.fits' and delete_all:
                        file.unlink(missing_ok=True) # delete
            if delete_all:
                for fname in ['hdul_sky.fits', 'master_dark.fits', 'master_flat.fits']:
                    (self.output_dir / fname).unlink(missing_ok=True)
            if delete_all:
                keep = []
            else:
                keep = ['src_0','sky','dark','flat'] # keep for debugging/plotting
            rect_files = [f for f in self.output_dir.iterdir() if f.name.startswith('rectified_')]
            if len(rect_files)>4:
                print('Deleting rectified files')
                for file in rect_files:
                    core_name = file.stem.replace('rectified_', '')  # remove prefix
                    if core_name not in keep:
                        file.unlink(missing_ok=True) # delete

        return wl, fl, err
        
    def observe_transit_calib(self,transit_flux_array,plot_exp=None):

        print(f'\n *** Generating observations... *** \n')

        transit_simulated_frames = []
        hdul_src_0 = pathlib.Path(f'{self.output_dir}/transit_frame_0.fits')
        for i,fl in enumerate(transit_flux_array):
            
            hdul_src_path = pathlib.Path(f'{self.output_dir}/transit_frame_{int(i)}.fits')
            rect_path = pathlib.Path(f'{self.output_dir}/rectified_src_{int(i)}.fits')
            
            if hdul_src_path.exists():
                hdul_src = fits.open(hdul_src_path)
            elif hdul_src_path.exists()==False and rect_path.exists():
                # rectified bservation already generated, no need to make new obs
                hdul_src = fits.open(hdul_src_0) # append dummy to list, will not be used
            else:
                src = sim.Source(x=[0],y=[0],ref=[0],lam=self.planet_wl_obs_range, spectra=fl) 
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
            self.plot_raw_obs(transit_simulated_frames[plot_exp],master_flat_hdul,
                                     master_dark_hdul,hdul_sky,plot_exp=plot_exp)

        print(f'\n *** Observations ready. *** \n')
        return transit_simulated_frames, master_dark_hdul, master_flat_hdul, hdul_sky
    
    def rectify_calibrate_extract(self, 
                                  transit_simulated_frames, master_dark_hdul, 
                                  master_flat_hdul, hdul_sky, 
                                  aperture_halfwidth=3, plot_exp=None):
        
        print(f'\n *** Calibrating observations... *** \n')

        def get_rectified(target,hdul_raw,delete_raw_hduls=True):
            rect_path = pathlib.Path(f'{self.output_dir}/rectified_{target}.fits')
            if rect_path.exists():
                rect = fits.open(rect_path)[1]
            else:
                rect = self.metis["lms_spectral_traces"].rectify_cube(hdul_raw)
                rect.writeto(rect_path, overwrite=True)

            keep = ['src_0','sky','dark','flat'] # keep for investigating/plotting
            if delete_raw_hduls and target not in keep: # take up too much space
                num = int(target.split('_')[1])
                hdul_src_path = pathlib.Path(f'{self.output_dir}/transit_frame_{int(num)}.fits')
                if hdul_src_path.exists():
                    print(f'Created rectified_{target}, deleting transit_frame_{int(num)}')
                    hdul_src_path.unlink(missing_ok=True)  # delete file
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
        
        def estimate_variance_from_calib_data(data):

            nwave, ny, nx = data.shape

            # define margins
            edge_margin = 3 # pixels to exclude at edges (often noisy, to be safe)
            source_center = (ny//2, nx//2) # approximate PSF center
            source_radius = 5 # pixels radius to exclude source + margin

            # build mask
            y, x = np.indices((ny, nx))
            mask_edges = (y >= edge_margin) & (y < ny - edge_margin) & (x >= edge_margin) & (x < nx - edge_margin)
            mask_source = ((y - source_center[0])**2 + (x - source_center[1])**2) > source_radius**2
            mask = mask_edges & mask_source  # True = pixel included for background

            # compute std per wavelength
            bg_std = np.std(data[:, mask], axis=1)  # shape (nwave,)

            # replace zeros with wavelength-dependent median (some edge wavelengths...)
            bg_std_corrected = np.where(bg_std==0, np.median(bg_std[bg_std>0]), bg_std)
            
            # broadcast to full cube
            var_cube = np.ones_like(data) * (bg_std_corrected[:, np.newaxis, np.newaxis]**2)
            return var_cube
    
        # remove edge pixels (near-zero flux)
        clip_x = 2 
        sky = rect_sky.data.astype(float)[:, :, clip_x:-clip_x]
        dark = rect_dark.data.astype(float)[:, :, clip_x:-clip_x]
        flat = rect_flat.data.astype(float)[:, :, clip_x:-clip_x]

        calib_frames = []
        var_cubes = []
        for i,rect_src in enumerate(rect_src_frames):
            if i==0:
                wavelengths, nwave = get_wavelegth_from_header(rect_src, self.hdr)
            try:
                src = rect_src.data.astype(float)[:, :, clip_x:-clip_x]
            except:
                print('Corrupted:',i)
            src_dark = src - dark
            sky_dark = sky - dark
            src_dark_flat =  src_dark / flat
            sky_dark_flat =  sky_dark / flat
            src_minus_sky = src_dark_flat - sky_dark_flat

            if plot_exp is not None:
                if i==plot_exp:
                    self.plot_rectified(src,dark,src_dark_flat,flat,src_minus_sky,sky_dark_flat,plot_exp)

            calib_frames.append(src_minus_sky)
            var_cube = estimate_variance_from_calib_data(src_minus_sky)[:, :, clip_x:-clip_x]
            var_cubes.append(var_cube)           

        def find_trace_and_optimally_extract(cube, var_cube=None, wavelengths=None,
                                            aperture_halfwidth=3, smooth_sigma=5, 
                                            signal_frac_thresh=0.15, poly_order=3,
                                            plot_diagnostics=False, plot_exp=None,
                                            fixed_trace=True,fix_source_center=True):
            """
            Find the spectral trace and extract a 1D spectrum from a 3D (λ, y, x) cube,
            using a wavelength-dependent poly threshold for robust slice selection.
            Optimal extraction of flux using PSF weighting.
            """

            # --- initialize ---
            wl = wavelengths
            nwave, ny, nx = cube.shape

            # Arrays to store centroid and flux info per slice
            centroid_y = np.full(nwave, np.nan)
            centroid_x = np.full(nwave, np.nan)
            maxvals = np.zeros(nwave)
            frac_nonzero = np.zeros(nwave)

            # --- loop over all wavelength slices ---
            for i in range(nwave):
                slice_ = cube[i, :, :]

                # skip slices that are fully empty or NaN
                if not np.any(np.isfinite(slice_)):
                    continue

                # sum over x and y to get 1D profiles
                slice_ = slice_ - np.nanmedian(slice_) # remove per-slice background offset
                profile_y = np.nansum(slice_, axis=1)  # sum over columns -> row profile
                profile_x = np.nansum(slice_, axis=0)  # sum over rows -> column profile
                profile_y = gaussian_filter1d(profile_y, sigma=1)
                profile_x = gaussian_filter1d(profile_x, sigma=1)

                # record maximum flux and fraction of pixels that are finite
                maxvals[i] = np.nanmax(profile_y)
                frac_nonzero[i] = np.sum(np.isfinite(profile_y)) / ny

                # skip if profile is fully NaN
                if np.all(np.isnan(profile_y)) or np.all(np.isnan(profile_x)):
                    continue

                # --- compute centroid using flux-weighted mean ---
                y_indices = np.arange(ny)
                x_indices = np.arange(nx)
                
                # only if slice has signal
                if np.nansum(profile_y) > 0:
                    centroid_y[i] = np.nansum(y_indices * profile_y) / np.nansum(profile_y)
                    centroid_x[i] = np.nansum(x_indices * profile_x) / np.nansum(profile_x)
                else:
                    centroid_y[i] = np.nan
                    centroid_x[i] = np.nan

            # --- wavelength-dependent threshold ---
            valid = np.isfinite(maxvals)
            p_thresh = np.polyfit(wl[valid], maxvals[valid], deg=2)
            threshold = np.polyval(p_thresh, wl)
            sig_mask = (maxvals > threshold) | (frac_nonzero > signal_frac_thresh)
            sig_mask &= np.isfinite(centroid_y)

            good_idxs = np.where(sig_mask)[0]

            if len(good_idxs) == 0:
                sorted_idx = np.argsort(maxvals)[::-1]
                topn = max(10, int(0.1 * nwave))
                good_idxs = sorted_idx[:topn]
                print(f"[WARN] No slices passed threshold — fallback to top {len(good_idxs)} by flux")

            # --- reject centroid outliers using median absolute deviation ---
            y_med = np.nanmedian(centroid_y[good_idxs])
            y_mad = 1.4826 * np.nanmedian(np.abs(centroid_y[good_idxs] - y_med))
            valid = np.abs(centroid_y[good_idxs] - y_med) < 6 * y_mad
            good_idxs = good_idxs[valid]

            # Ensure enough slices remain to fit polynomial (only needed if not fixed trace)
            if not fixed_trace and len(good_idxs) < poly_order + 2:
                raise RuntimeError(
                    f"Not enough valid slices ({len(good_idxs)}) to fit a {poly_order}-order trace.")

            # --- trace model ---
            if fix_source_center:
                trace_y = np.full(nwave, ny//2)
                trace_x = np.full(nwave, (nx-1)//2)

            elif fixed_trace:
                # use global median centroid across all good slices
                trace_y_mean = np.nanmedian(centroid_y[good_idxs])
                trace_x_mean = np.nanmedian(centroid_x[good_idxs])
                trace_y = np.full(nwave, trace_y_mean)
                trace_x = np.full(nwave, trace_x_mean)
                #print(f"[INFO] Using fixed centroid: y = {trace_y_mean:.2f}, x = {trace_x_mean:.2f}")

            else:
                # smooth centroids and fit polynomial to model moving trace
                centroids_y_smooth = gaussian_filter(centroid_y[good_idxs], sigma=smooth_sigma)
                centroids_x_smooth = gaussian_filter(centroid_x[good_idxs], sigma=smooth_sigma)

                p_y = np.polyfit(good_idxs, centroids_y_smooth, deg=poly_order)
                p_x = np.polyfit(good_idxs, centroids_x_smooth, deg=poly_order)

                trace_y = np.polyval(p_y, np.arange(nwave))
                trace_x = np.polyval(p_x, np.arange(nwave))

            # --- extract 1D spectrum along the trace using PSF weighting (optimal extraction) ---
            spec = np.zeros(nwave)
            spec_err = np.zeros(nwave)

            for i in range(nwave):
                y0, x0 = trace_y[i], trace_x[i]

                # define integer bounding box for aperture
                ylo = int(np.clip(y0 - aperture_halfwidth, 0, ny - 1))
                yhi = int(np.clip(y0 + aperture_halfwidth + 1, 0, ny))
                xlo = int(np.clip(x0 - aperture_halfwidth, 0, nx - 1))
                xhi = int(np.clip(x0 + aperture_halfwidth + 1, 0, nx))

                # extract sub-image inside aperture
                subim = cube[i, ylo:yhi, xlo:xhi]

                # create a 2D Gaussian weight centered on centroid
                y_inds = np.arange(ylo, yhi)
                x_inds = np.arange(xlo, xhi)
                Y, X = np.meshgrid(y_inds, x_inds, indexing='ij')
                sigma = aperture_halfwidth / 2  # Gaussian width, adjust if needed
                weights = np.exp(-0.5 * ((Y - y0)**2 + (X - x0)**2) / sigma**2)
                weights /= np.sum(weights)  # normalize

                # weighted flux sum
                subim_masked = np.where(subim < 0, 0, subim)
                spec[i] = np.nansum(subim_masked * weights)

                # weighted error
                if var_cube is not None:
                    subvar = var_cube[i, ylo:yhi, xlo:xhi]
                    spec_err[i] = np.sqrt(np.nansum(subvar * weights**2))

            # --- optional diagnostics plot ---
            if plot_exp is not None and plot_diagnostics:
                fig, ax = plt.subplots(2, 1, figsize=(6, 3), sharex=True)
                ax[0].plot(wl, maxvals, color="k", lw=1)
                ax[0].plot(wl, threshold, ls="--", color="r", label="Threshold")
                ax[0].set_ylabel("Max slice flux")
                ax[0].legend()
                ax[1].plot(wl, centroid_y, ".", color="gray", alpha=0.4)
                ax[1].plot(wl, trace_y, "r-", label="Fitted trace" if not fixed_trace else "Fixed trace")
                ax[1].set_ylabel("Centroid Y [pix]")
                ax[1].set_xlabel("Wavelength [\u03BCm]")
                ax[1].set_ylim(0,ny)
                ax[1].legend()
                plt.tight_layout()
                plt.subplots_adjust(hspace=0)
                fig.savefig(f'{self.figs_dir}/extracted_trace_{self.order}{plot_exp}.pdf', bbox_inches='tight')
                plt.close()

            return wl, spec, spec_err, trace_y, trace_x
        
        def animate_trace(cube,trace_y,trace_x,wl_obs,flx_obs,
                        aperture_halfwidth=3,interval=50,plot_exp=0):
            """
            Animate a 2D rectified cube showing the source trace with a circular aperture,
            and a moving vertical line over the 1D extracted spectrum.
            """

            step=10 # use only every 10th frame
            nwave, ny, nx = cube.shape
            vmin = np.nanmin(cube[::step])
            vmax = np.nanmax(cube[::step])

            fig, (ax_img, ax_spec) = plt.subplots(2, 1, figsize=(6, 4), dpi=100, 
                                                  gridspec_kw={'height_ratios': [2, 1]},
                                                  constrained_layout=True)

            # --- top: image with aperture circle ---
            im = ax_img.imshow(cube[0], origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
            circ = Circle((trace_x[0], trace_y[0]), aperture_halfwidth,
                        edgecolor='cyan', facecolor='none', lw=1.5)
            ax_img.add_patch(circ)
            frame_str = f"{0:04d}"
            wl_str = f"{wl_obs[0]:.5f}"
            ax_img.set_title(f"Wavelength slice {frame_str}, {wl_str} µm")

            # --- bottom: 1D extracted spectrum ---
            ax_spec.plot(wl_obs, flx_obs, color='k', lw=0.8)
            line = ax_spec.axvline(wl_obs[0], color='r', lw=1.2)
            ax_spec.set_xlabel("Wavelength [µm]")
            ax_spec.set_ylabel("Flux")
            ax_spec.set_xlim(np.nanmin(wl_obs), np.nanmax(wl_obs))
            ax_spec.set_ylim(0, np.nanmax(flx_obs) * 1.1)
            ax_spec.yaxis.set_major_formatter(ScalarFormatter())
            ax_spec.yaxis.get_major_formatter().set_scientific(True)
            ax_spec.yaxis.get_major_formatter().set_powerlimits((0, 0))

            # --- update function for animation ---
            def update(frame):
                im.set_data(cube[frame])
                circ.center = (trace_x[frame], trace_y[frame])
                frame_str = f"{frame:04d}"
                wl_str = f"{wl_obs[frame]:.5f}"
                ax_img.set_title(f"Wavelength slice {frame_str}, {wl_str} µm")
                line.set_xdata([wl_obs[frame], wl_obs[frame]])
                return [im, circ, line]

            anim = FuncAnimation(fig, update, frames=np.arange(0, nwave, step), interval=interval, blit=True)
            savepath = f'{self.figs_dir}/trace_animation_{self.order}{plot_exp}.gif'
            writer = PillowWriter(fps=50)
            anim.save(savepath, writer=writer)
            plt.close()
            return

        fluxes_obs = []
        err_obs = []
        for i,calib in enumerate(calib_frames):

            if i==plot_exp:
                plot_diagnostics = True
            else:
                plot_diagnostics = False
            wl_obs, flx_obs, flx_obs_err, trace_y, trace_x = find_trace_and_optimally_extract(calib, 
                                                                var_cube=var_cubes[i], 
                                                                wavelengths=wavelengths,
                                                                aperture_halfwidth=aperture_halfwidth,
                                                                plot_diagnostics=plot_diagnostics,
                                                                plot_exp=plot_exp)
            if i==plot_exp:
                # Use the same integer bounding-box center as in extraction
                def get_integer_aperture_centers(trace_y, trace_x, ny, nx, aperture_halfwidth):
                    trace_y_int = np.zeros_like(trace_y)
                    trace_x_int = np.zeros_like(trace_x)
                    for i in range(len(trace_y)):
                        y0, x0 = trace_y[i], trace_x[i]
                        ylo = int(np.clip(y0 - aperture_halfwidth, 0, ny - 1))
                        yhi = int(np.clip(y0 + aperture_halfwidth + 1, 0, ny))
                        xlo = int(np.clip(x0 - aperture_halfwidth, 0, nx - 1))
                        xhi = int(np.clip(x0 + aperture_halfwidth + 1, 0, nx))
                        # approximate the extraction aperture center (integer midpoint)
                        trace_y_int[i] = (ylo + yhi - 1) / 2
                        trace_x_int[i] = (xlo + xhi - 1) / 2
                    return trace_y_int, trace_x_int
                
                trace_y_int, trace_x_int = get_integer_aperture_centers(trace_y, trace_x, calib.shape[1], calib.shape[2], aperture_halfwidth)
                animate_trace(calib, trace_y_int, trace_x_int, wl_obs, flx_obs,
                            aperture_halfwidth=aperture_halfwidth, plot_exp=plot_exp)
            fluxes_obs.append(flx_obs)
            err_obs.append(flx_obs_err)

        wl_obs = np.array(wl_obs)
        fluxes_obs = np.array(fluxes_obs)
        err_obs = np.array(err_obs)

        if plot_exp is not None:
            self.plot_extracted_spetrum(wl_obs,fluxes_obs[plot_exp],
                                        err_obs[plot_exp],plot_exp=plot_exp)

        print(f'\n *** Observations calibrated. *** \n')
        return wl_obs, fluxes_obs, err_obs

    def plot_raw_obs(self,hdul_src,master_flat_hdul,master_dark_hdul,hdul_sky,plot_exp=''):
        # Compare raw observations before rectification
        x = 1
        src_raw  = hdul_src[x].data.astype(float)
        flat_raw= master_flat_hdul[x].data.astype(float)
        dark_raw= master_dark_hdul[x].data.astype(float)
        sky_raw= hdul_sky[x].data.astype(float)

        fig, ax = plt.subplots(2,2,figsize=(7,6))
        im0 = ax[0,0].imshow(src_raw, vmin=0, vmax=np.percentile(src_raw, 99), aspect='auto')
        ax[0,0].set_title("Science (raw)")
        plt.colorbar(im0,ax=ax[0,0])
        im1 = ax[1,0].imshow(flat_raw, vmin=0,vmax=np.nanpercentile(flat_raw,99), aspect='auto')
        ax[1,0].set_title("Flat (raw)")
        plt.colorbar(im1,ax=ax[1,0])
        im2 = ax[0,1].imshow(sky_raw, vmin=0, vmax=np.percentile(sky_raw, 99), aspect='auto')
        ax[0,1].set_title("Sky (raw)")
        plt.colorbar(im2,ax=ax[0,1])
        im3 = ax[1,1].imshow(dark_raw, vmin=0, vmax=np.percentile(dark_raw, 99), aspect='auto')
        ax[1,1].set_title("Dark (raw)")
        plt.colorbar(im3,ax=ax[1,1])
        fig.tight_layout()
        fig.savefig(f'{self.figs_dir}/raw_obs_{self.order}{plot_exp}.pdf', bbox_inches='tight')
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
        fig.savefig(f'{self.figs_dir}/rectified_{self.order}{plot_exp}.pdf', bbox_inches='tight')
        plt.close()

    def plot_extracted_spetrum(self,wl,fl,err,plot_exp=''):
        fig = plt.figure(figsize=(6,2.5),dpi=150)
        plt.plot(wl,fl,lw=0.8)
        plt.fill_between(wl, fl - err, fl+err, color='gray', alpha=0.3)
        plt.xlabel('Wavelength [µm]')
        plt.ylabel('Extracted flux')
        fig.savefig(f'{self.figs_dir}/extracted_spectrum_{self.order}{plot_exp}.pdf', bbox_inches='tight')
        plt.close()