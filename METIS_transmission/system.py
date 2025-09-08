import numpy as np
import astropy.units as u
import wget
import os
import pathlib
from astropy.io import fits
from astropy.table import QTable
from utils import *
from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS import physical_constants as cst

class System:

    def __init__(self,name,exptime):

        self.name=name
        self.data_path = os.path.join(os.getcwd(),name)
        self.exptime = exptime

        self.R_star = 0.5 * u.R_sun # stellar radius
        self.d_star = 38 * u.pc # distance to K2-18
        sma = 0.15*u.au
        #period = 32.9*u.day
        period = 10.0*u.day # USE SHORTER PERIOD FOR TESTING -> larger Kp
        incl = 89.57*u.deg # inclination
        e = 0.0
        Kp = ((2*np.pi*sma*np.sin(incl))/(period*np.sqrt(1-e**2))).to(u.km/u.s) # RV semi-amplitude
        #print(f"Kp = {Kp:.3f}")

        transit_duration = 2.67*u.h
        num_exp_in_transit = int(transit_duration.to(u.s)/exptime) # number of exposures during transit
        delta_phase = (transit_duration.to(u.day)/2/period).value # half duration in phase
        phase_transit = np.linspace(-delta_phase,delta_phase,num_exp_in_transit) # phase during transit
        rv_transit = Kp*np.sin(2*np.pi*phase_transit)
        #print(f"RV at ingress/egress: +/- {np.max(rv_transit):.3f}")

        self.overhead = int(10) # total -> make even number
        self.num_exp = num_exp_in_transit + self.overhead # total number of exposures
        idx = np.arange(self.num_exp)
        t_offsets = (idx - (self.num_exp - 1)/2.0) * exptime # time offset (in u.s) symmetric around mid-transit t=0
        t_offsets_days = t_offsets.to(u.day)
        phase_obs = (t_offsets_days / period).decompose().value # phase during observation (transit + overhead)
        self.rv_obs = Kp * np.sin(2.0 * np.pi * phase_obs) 
        #print(f"RV at obs start/end: +/- {np.max(rv_obs):.3f}")

        self.planet_wl_um, transit_radii_um  = self.get_planet_spectrum()
        self.delta_lambda = ((transit_radii_um.to(u.cm) / self.R_star.to(u.cm))**2).value  # wavelength-dependent transit depth
        star_wl, star_flx = self.get_stellar_spectrum()

        # interpolate stellar flux onto planet wavelength grid
        star_wl_um = star_wl.to(u.um) # from AA to um
        star_flx_um = star_flx.to(u.erg / (u.s * u.cm**2 * u.um), equivalencies=u.spectral_density(star_wl))
        star_flx_um = instrumental_broadening(star_wl_um.value, star_flx_um.value, resolution=1e5)*star_flx_um.unit
        self.star_flx_interp = np.interp(self.planet_wl_um.value, star_wl_um.value, star_flx_um.value)*star_flx_um.unit

    def get_planet_spectrum(self):
        planet_spectrum = pathlib.Path(f'{self.data_path}/planet_spectrum.fits')

        if planet_spectrum.exists():
            tbl = QTable.read(planet_spectrum)
            planet_wl_um = tbl['wavelength'] # in um
            transit_radii_um = tbl['flux'] # in um
            transit_radii_cm = (transit_radii_um.to(u.cm)).value
        else:
            
            radtrans = Radtrans(
                pressures=np.logspace(-6, 2, 100),
                line_species=['H2O','CH4'],
                rayleigh_species=['H2', 'He'],
                gas_continuum_contributors=['H2--H2', 'H2--He'],
                wavelength_boundaries=[2.8,4.2], # microns (L-band)
                line_opacity_mode='lbl')

            temperatures = 300 * np.ones_like(radtrans.pressures) # radtrans.pressures is in cgs units

            mass_fractions = {
                'H2': 0.74 * np.ones(temperatures.size),
                'He': 0.24 * np.ones(temperatures.size),
                'H2O': 1e-2 * np.ones(temperatures.size),
                'CH4': 1e-2 * np.ones(temperatures.size)}

            #  2.33 is a typical value for H2-He dominated atmospheres
            mean_molar_masses = 2.33 * np.ones(temperatures.size)
            
            planet_radius = 1.0 * cst.r_jup_mean # FIX PLANET RADIUS
            reference_gravity = 10 ** 3.5
            reference_pressure = 0.01

            planet_wl_cm, transit_radii_cm, _ = radtrans.calculate_transit_radii(
                                                                        temperatures=temperatures,
                                                                        mass_fractions=mass_fractions,
                                                                        mean_molar_masses=mean_molar_masses,
                                                                        reference_gravity=reference_gravity,
                                                                        planet_radius=planet_radius,
                                                                        reference_pressure=reference_pressure)
            
            transit_radii_cm = instrumental_broadening(planet_wl_cm, transit_radii_cm, resolution=1e5)
            transit_radii_um = (transit_radii_cm*1e4)*u.um
            planet_wl_um = (planet_wl_cm*1e4)*u.um

            planet_spectrum = pathlib.Path(f'{self.data_path}/planet_spectrum.fits')
            tbl = QTable([planet_wl_um, transit_radii_um], names=['wavelength', 'flux'])
            tbl.write(planet_spectrum, overwrite=True)

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

            data_path = os.getcwd()
            savepath = os.path.join(data_path, fname)
            
            wave_path = 'ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'
            wave_savepath = os.path.join(data_path, 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')

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

        star_template = pathlib.Path(f'M2_star.fits')
        if star_template.exists():
            tbl = QTable.read(star_template)
            star_wl = tbl['wavelength']
            star_flx = tbl['flux']
        else:
            star_wl, star_flx = get_PHOENIX_model(temperature=3500, log_g=5, fe_h=0, wl_lims=[2.8*u.micron, 4.2*u.micron])
            tbl = QTable([star_wl, star_flx], names=['wavelength', 'flux'])
            tbl.write(star_template, overwrite=True)

        return star_wl, star_flx
    
    def get_transit_array(self,wl_obs_range):

        wl_mask = (self.planet_wl_um>np.min(wl_obs_range)) & (self.planet_wl_um<np.max(wl_obs_range[-1]))
        planet_wl_obs_range = self.planet_wl_um[wl_mask]
        self.in_transit = np.linspace(self.overhead//2,self.num_exp-self.overhead//2-1,self.num_exp-self.overhead,dtype=int)
        transit_flux_array = np.empty(shape=(self.num_exp,len(planet_wl_obs_range.value)))

        for i in range(self.num_exp):
            if i in self.in_transit:

                    rv = self.rv_obs[i]
                    rv = rv.to(u.km/u.s) if isinstance(rv, u.Quantity) else rv * u.km/u.s
                    beta = (1+rv/const.c.to('km/s')).value
                    wl_shifted = beta*self.planet_wl_um
                    
                    # convert to transit depth
                    delta_lambda_interp = np.interp(planet_wl_obs_range, wl_shifted, self.delta_lambda)
                    star_flux_interp = np.interp(planet_wl_obs_range, self.planet_wl_um, self.star_flx_interp)
                    transit_flux = star_flux_interp * (1.0 - delta_lambda_interp)
                    transit_flux_earth = (transit_flux * (self.R_star / self.d_star)**2).to(transit_flux.unit) # scale by distance
                    transit_flux_array[i] = transit_flux_earth.value
                    
            else:
                star_flux_only = np.interp(planet_wl_obs_range,self.planet_wl_um,self.star_flx_interp)
                star_flux_only_earth = (star_flux_only * (self.R_star / self.d_star)**2).to(star_flux_only.unit) # scale by distance
                transit_flux_array[i] = star_flux_only_earth
                        
            transit_flux_array *= self.star_flx_interp.unit  
            self.planet_wl_obs_range = planet_wl_obs_range
            return transit_flux_array




