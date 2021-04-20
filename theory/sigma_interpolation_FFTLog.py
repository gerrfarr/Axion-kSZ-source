from typing import List

from ..axion_camb_wrappers.power_interpolation import LinearPowerInterpolation
from ..axion_camb_wrappers.growth_interpolation import GrowthInterpolation
from ..auxiliary.window_functions import WindowFunctions
from ..auxiliary.integration_helper import IntegrationHelper
from ..auxiliary.sharp_k_sq_transform import SharpKVar
from .cosmology import Cosmology
import numpy as np
from scipy.interpolate import interp1d, RectBivariateSpline
import mcfit

class SigmaInterpolatorFFTLog(object):

    def __init__(self, cosmo, power, growth, z_vals, kMin, kMax, Nr=1024, window_function='top_hat'):
        """

        :type integrationHelper: IntegrationHelper
        :type growth: GrowthInterpolation
        :type power: LinearPowerInterpolation
        :type cosmo: Cosmology
        """
        self.__cosmo = cosmo
        self.__power = power
        self.__growth = growth
        self.__window_function = window_function

        self.window = None
        self.dwindow = None
        self.radius_of_mass = None
        self.mass_of_radius = None

        #assert (window_function == 'top_hat' or window_function == 'gaussian')

        WindowFunctions.set_window_functions(self, window_function, self.__cosmo)

        self.__z_vals = z_vals
        self.__k_vals = np.logspace(np.log10(kMin), np.log10(kMax), Nr)
        self.__r_vals = np.full(Nr, np.nan)
        self.__r_vals_deriv = np.full(Nr, np.nan)

        if self.__window_function == 'top_hat':
            self.__transform = mcfit.TophatVar(self.__k_vals, lowring=False)
            self.__transform_deriv = mcfit.TophatVar(self.__k_vals, lowring=False, deriv=1j)
        elif self.__window_function == 'gaussian':
            self.__transform = mcfit.GaussVar(self.__k_vals, lowring=False)
            self.__transform_deriv = mcfit.GaussVar(self.__k_vals, lowring=False, deriv=1j)
        elif self.__window_function == 'sharp_k':
            self.__transform = SharpKVar(self.__k_vals, lowring=False)
            self.__transform_deriv = SharpKVar(self.__k_vals, lowring=False, deriv=1j)

        self.__sigma_sq_vals = None
        self.__dsigma_sq_dr_vals = None
        self.__dsigma_sq_dloga_vals = None

        self.__interpolator = None
        self.__interpolator_dr = None
        self.__interpolator_dloga = None

    def compute(self, do_dr=True, do_dloga=True):
        kMesh, zMesh = np.meshgrid(self.__k_vals, self.__z_vals)

        integrand_base = self.__growth(kMesh, zMesh)**2*self.__power(kMesh)
        self.__r_vals, self.__sigma_sq_vals = self.__transform(integrand_base, extrap=True)

        self.__interpolator = RectBivariateSpline(np.log10(self.__r_vals), self.__z_vals, self.__sigma_sq_vals.T, bbox=[np.min(np.log10(self.__r_vals)), np.max(np.log10(self.__r_vals)), np.min(self.__z_vals), np.max(self.__z_vals)], kx=3, ky=1, s=0)

        if do_dr:
            self.__r_vals_deriv,  vals = self.__transform_deriv(integrand_base, extrap=True)
            self.__dsigma_sq_dr_vals = np.einsum('ij, j->ij', vals, 1/self.__r_vals_deriv)

            self.__interpolator_dr = RectBivariateSpline(np.log10(self.__r_vals_deriv), self.__z_vals, self.__dsigma_sq_dr_vals.T, bbox=[np.min(np.log10(self.__r_vals_deriv)), np.max(np.log10(self.__r_vals_deriv)), np.min(self.__z_vals), np.max(self.__z_vals)], kx=3, ky=1, s=0)

        if do_dloga:
            integrand_dloga = 2 * integrand_base * self.__growth.f(kMesh, zMesh)
            dump, self.__dsigma_sq_dloga_vals = self.__transform(integrand_dloga, extrap=True)

            self.__interpolator_dloga = RectBivariateSpline(np.log10(self.__r_vals), self.__z_vals, self.__dsigma_sq_dloga_vals.T, bbox=[np.min(np.log10(self.__r_vals)), np.max(np.log10(self.__r_vals)), np.min(self.__z_vals), np.max(self.__z_vals)], kx=3, ky=1, s=0)

    def __sigma_sq_interpolation(self, r, z):
        return np.squeeze(self.__interpolator.ev(np.log10(r), z))

    def __dsigma_sq_dr_interpolation(self, r, z):
        return np.squeeze(self.__interpolator_dr.ev(np.log10(r), z))

    def __dsigma_sq_dloga_interpolation(self, r, z):
        return np.squeeze(self.__interpolator_dloga.ev(np.log10(r), z))

    def __call__(self, m, z):
        return np.sqrt(self.__sigma_sq_interpolation(self.radius_of_mass(m), z))

    def sigma_of_r(self, r, z):
        return np.sqrt(self.__sigma_sq_interpolation(r, z))

    def dsigma_dloga_of_r(self, r, z):
        return self.__dsigma_sq_dloga_interpolation(r, z)/(2*self.sigma_of_r(r, z))

    def dsigma_dloga_of_m(self, m, z):
        return self.__dsigma_sq_dloga_interpolation(self.radius_of_mass(m), z) / (2 * self(m, z))

    def dlogSigma_sq_dloga_of_m(self, m, z):
        return self.__dsigma_sq_dloga_interpolation(self.radius_of_mass(m), z) / self.__sigma_sq_interpolation(self.radius_of_mass(m), z)

    def dlogSigma_dloga_of_m(self, m, z):
        return self.dlogSigma_sq_dloga_of_m(m,z)/2.0

    def dsigma_dr_of_r(self, r, z):
        return self.__dsigma_sq_dr_interpolation(r, z) / (2 * self.sigma_of_r(r, z))

    def dsigma_dr_of_m(self, m, z):
        return self.__dsigma_sq_dr_interpolation(self.radius_of_mass(m), z) / (2 * self(m, z))

    def dsigma_dm(self, m, z):
        return self.dsigma_dr_of_m(m,z) * self.radius_of_mass(m) / m / 3.0

    def dlogSigma_dlogm(self, m, z):
        return self.dsigma_dm(m, z)*m/self(m,z)
