import numpy as np
import astropy.units as u
import wget
import os
import pathlib
from astropy.io import fits
from astropy.table import QTable
from utils import *
import matplotlib.pyplot as plt
from petitRADTRANS import physical_constants as cst
from pRT_model import *

class System:

    def __init__(self,parameters,plot=False):

        self.parameters = parameters
        self.project_path = parameters['project_path'] 
        self.exptime = parameters['exptime_per_frame']
        figs_path = pathlib.Path(f'{self.project_path}/figures')
        figs_path.mkdir(parents=True, exist_ok=True)

        self.R_star = parameters['R_star']
        self.d_star = parameters['d_star']
        sma = parameters['sma']
        period = parameters['period']
        incl = parameters['inclination']
        e = parameters['e']
        self.Kp = ((2*np.pi*sma*np.sin(incl))/(period*np.sqrt(1-e**2))).to(u.km/u.s) # RV semi-amplitude
        #print(f"Kp = {self.Kp:.3f}")

        transit_duration = 2.67*u.h
        num_exp_in_transit = int(transit_duration.to(u.s)/self.exptime) # number of exposures during transit
        delta_phase = (transit_duration.to(u.day)/2/period).value # half duration in phase
        self.phase_transit = np.linspace(-delta_phase,delta_phase,num_exp_in_transit) # phase during transit
        self.rv_transit = self.Kp*np.sin(2*np.pi*self.phase_transit)
        #print(f"RV at ingress/egress: +/- {np.max(self.rv_transit):.3f}")

        self.overhead = int(10) # total -> make even number
        self.num_exp = num_exp_in_transit + self.overhead # total number of exposures
        idx = np.arange(self.num_exp)
        t_offsets = (idx - (self.num_exp - 1)/2.0) * self.exptime # time offset (in u.s) symmetric around mid-transit t=0
        t_offsets_days = t_offsets.to(u.day)
        self.phase_obs = (t_offsets_days / period).decompose().value # phase during observation (transit + overhead)
        self.rv_obs = self.Kp * np.sin(2.0 * np.pi * self.phase_obs) 
        #print(f"RV at obs start/end: +/- {np.max(rv_obs):.3f}")

        self.planet_wl_um, transit_radii_um  = self.get_planet_spectrum()
        self.transit_radii_cm = (transit_radii_um.to(u.cm))
        self.delta_lambda = ((self.transit_radii_cm / self.R_star.to(u.cm))**2).value  # wavelength-dependent transit depth
        star_wl, star_flx = self.get_stellar_spectrum()

        # interpolate stellar flux onto planet wavelength grid
        star_wl_um = star_wl.to(u.um) # from AA to um
        star_flx_um = star_flx.to(u.erg / (u.s * u.cm**2 * u.um), equivalencies=u.spectral_density(star_wl))
        star_flx_um = instrumental_broadening(star_wl_um.value, star_flx_um.value, resolution=1e5)*star_flx_um.unit
        self.star_flx_interp = np.interp(self.planet_wl_um.value, star_wl_um.value, star_flx_um.value)*star_flx_um.unit

        if plot:
            figs_dir = pathlib.Path(f'{self.project_path}/figures/input')
            figs_dir.mkdir(parents=True, exist_ok=True)
            self.plot_stellar_spectrum()
            self.plot_planet_transmission()

    def get_planet_spectrum(self):
        planet_spectrum = pathlib.Path(f'{self.project_path}/pRT_spectra/planet_spectrum.fits')

        if planet_spectrum.exists():
            tbl = QTable.read(planet_spectrum)
            planet_wl_um = tbl['wavelength'] # in um
            transit_radii_um = tbl['flux'] # in um
        else:
            planet_wl_um, transit_radii_um = pRT_spectrum(self.parameters).make_spectrum(save_as='planet_spectrum')

        return planet_wl_um, transit_radii_um

    def get_stellar_spectrum(self):

        def get_PHOENIX_model(temperature, log_g, fe_h=0, wl_lims=[0.5*u.micron, 3*u.micron]):
            if isinstance(temperature, u.Quantity):
                temperature = temperature.to(u.K).value

            sign_specifier = '+' if fe_h > 0 else '-'
            t_val = int(200*np.round(temperature/200))
            log_g_val = 0.5*np.round(log_g/0.5)
            fe_h_val = 0.5*np.round(fe_h/0.5)
            
            fname = f'lte{t_val:05d}-{log_g_val:.2f}{sign_specifier}{np.abs(fe_h_val):.1f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
            fpath = f'ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/{fname}'
            savepath = os.path.join(self.project_path, fname)
            wave_path = 'ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'
            wave_savepath = os.path.join(self.project_path, 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')

            # Download wavelength grid if missing
            if not os.path.exists(wave_savepath):
                wget.download(wave_path, wave_savepath)

            # Download spectrum if missing
            if not os.path.exists(savepath):
                wget.download(fpath, savepath)

            # Load spectrum
            hdu_flux = fits.open(savepath)
            #flux_density = hdu_flux[0].data.astype(float) * u.erg / (u.s * u.cm**2 * u.AA)
            flux_density = hdu_flux[0].data.astype(float) * u.erg/ (u.cm**3 * u.s)
            hdu_wave = fits.open(wave_savepath)
            wavelengths = hdu_wave[0].data.astype(float) * u.AA

            # Mask to chosen wavelength range
            mask = (wavelengths >= wl_lims[0]) & (wavelengths <= wl_lims[1])
            return wavelengths[mask], flux_density[mask]

        star_template = pathlib.Path(f'{self.project_path}/M2_star.fits')
        if star_template.exists():
            tbl = QTable.read(star_template)
            star_wl = tbl['wavelength']
            star_flx = tbl['flux']
        else:
            star_wl, star_flx = get_PHOENIX_model(temperature=3500, log_g=5, fe_h=0, wl_lims=[2.8*u.micron, 4.2*u.micron])
            tbl = QTable([star_wl, star_flx], names=['wavelength', 'flux'])
            tbl.write(star_template, overwrite=True)

        return star_wl, star_flx
    
    def get_transit_array(self, wl_obs_range, plot=False):
        """
        Build time-series transit spectra interpolated onto a chosen observed wavelength range.
        """

        # Ensure wl_obs_range is a Quantity in same unit as planet_wl_um
        wl_min, wl_max = wl_obs_range
        wl_min = wl_min * self.planet_wl_um.unit if not isinstance(wl_min, u.Quantity) else wl_min
        wl_max = wl_max * self.planet_wl_um.unit if not isinstance(wl_max, u.Quantity) else wl_max

        # Select observed wavelength range
        wl_mask = (self.planet_wl_um > wl_min) & (self.planet_wl_um < wl_max)
        planet_wl_obs_range = self.planet_wl_um[wl_mask]

        # Indices of exposures that are in transit
        self.in_transit = np.linspace(self.overhead // 2, self.num_exp - self.overhead // 2 - 1,
                                      self.num_exp - self.overhead, dtype=int)

        # Initialize with units
        transit_flux_array = np.empty((self.num_exp, len(planet_wl_obs_range))) * self.star_flx_interp.unit

        # Loop over exposures
        for i in range(self.num_exp):
            if i in self.in_transit:
                # Radial velocity shift
                rv = self.rv_obs[i]
                rv = rv.to(u.km / u.s) if isinstance(rv, u.Quantity) else rv * u.km / u.s
                beta = (1.0 + rv / const.c.to("km/s")).value
                wl_shifted = beta * self.planet_wl_um

                # Interpolate transit depth and stellar flux
                delta_lambda_interp = np.interp(planet_wl_obs_range.value, wl_shifted.value, self.delta_lambda)
                star_flux_interp = np.interp(planet_wl_obs_range.value, self.planet_wl_um.value,
                                            self.star_flx_interp.value) * self.star_flx_interp.unit

                # Apply transit depth
                transit_flux = star_flux_interp * (1.0 - delta_lambda_interp)

                # Scale to Earth
                transit_flux_earth = (transit_flux * (self.R_star / self.d_star) ** 2).to(star_flux_interp.unit)
                transit_flux_array[i] = transit_flux_earth

            else:
                # Star-only flux (no transit)
                star_flux_only = np.interp(planet_wl_obs_range.value, self.planet_wl_um.value,
                                            self.star_flx_interp.value) * self.star_flx_interp.unit

                star_flux_only_earth = (star_flux_only * (self.R_star / self.d_star) ** 2).to(star_flux_only.unit)
                transit_flux_array[i] = star_flux_only_earth

        # Save to object for later use
        self.planet_wl_obs_range = planet_wl_obs_range
        self.transit_flux_array = transit_flux_array

        if plot:
            figs_dir = pathlib.Path(f'{self.project_path}/figures/input')
            figs_dir.mkdir(parents=True, exist_ok=True)
            self.plot_transit_timeseries()

        return transit_flux_array
    
    def plot_transit_timeseries(self):
        fig,ax=plt.subplots(1,1,figsize=(7,3),dpi=100)
        im = ax.imshow(self.transit_flux_array.value,aspect='auto',
                       extent=[np.min(self.planet_wl_um.value),np.max(self.planet_wl_um.value),
                        0,self.num_exp])
        plt.colorbar(im)
        ax.set_ylabel(f'Exposures')
        ax.set_xlabel(f'Wavelength [{self.planet_wl_um.unit}]')
        fig.savefig(f'{self.project_path}/figures/input/transit_timeseries.pdf', bbox_inches='tight')
        plt.close()

    def plot_stellar_spectrum(self):
        fig,ax=plt.subplots(1,1,figsize=(7,3),dpi=100)
        ax.plot(self.planet_wl_um.value,self.star_flx_interp, color='black',lw=0.5,label='Planet host')
        plt.legend()
        ax.set_ylabel(f'Flux [{self.star_flx_interp.unit}]')
        ax.set_xlabel(f'Wavelength [{self.planet_wl_um.unit}]')
        fig.savefig(f'{self.project_path}/figures/input/stellar_spectrum.pdf', bbox_inches='tight')
        plt.close()

    def plot_planet_transmission(self):
        fig,ax=plt.subplots(1,1,figsize=(7,3),dpi=100)
        ax.plot(self.planet_wl_um, self.transit_radii_cm / cst.r_jup_mean,lw=0.5)
        ax.set_xlabel(f'Wavelength [{self.planet_wl_um.unit}]')
        ax.set_ylabel(r'Transit radius [$\rm R_{Jup}$]')
        fig.savefig(f'{self.project_path}/figures/input/planet_spectrum.pdf', bbox_inches='tight')
        plt.close()