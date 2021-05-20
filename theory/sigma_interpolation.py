from typing import List

from ..axion_camb_wrappers.power_interpolation import LinearPowerInterpolation
from ..axion_camb_wrappers.growth_interpolation import GrowthInterpolation
from ..auxiliary.window_functions import WindowFunctions
from ..auxiliary.integration_helper import IntegrationHelper
from .cosmology import Cosmology
import numpy as np
from scipy.interpolate import interp1d, RectBivariateSpline

class SigmaInterpolator(object):

    def __init__(self, cosmo, power, growth, mMin, mMax, z_vals, integrationHelper, Nr=100, window_function='top_hat'):
        """

        :type integrationHelper: IntegrationHelper
        :type growth: GrowthInterpolation
        :type power: LinearPowerInterpolation
        :type cosmo: Cosmology
        """
        self.__cosmo = cosmo
        self.__power = power
        self.__growth = growth
        self.__intHelper = integrationHelper
        self.__window_function = window_function


        self.window = None
        self.dwindow = None
        self.radius_of_mass = None
        self.mass_of_radius = None

        WindowFunctions.set_window_functions(self, window_function, self.__cosmo)

        self.__r_vals = np.logspace(np.log10(self.radius_of_mass(mMin)), np.log10(self.radius_of_mass(mMax)), Nr)
        self.__z_vals = z_vals
        self.__sigma_sq_vals = None
        self.__dsigma_sq_dr_vals = None
        self.__dsigma_sq_dloga_vals = None

        self.__interpolator = None
        self.__interpolator_dr = None
        self.__interpolator_dloga = None

    def compute(self, kmin, kmax, smart=True, do_dr=True, do_dloga=True):
        if self.__window_function == 'sharp_k' and smart:
            eval_vals, weights = self.__intHelper.get_points_weights()
            kmax_vals = np.clip(1/self.__r_vals, kmin, kmax)
            rMesh, zMesh, dump = np.meshgrid(self.__r_vals, self.__z_vals, weights)

            k_eval_vals = np.outer((kmax_vals - kmin) / 2, eval_vals) + np.outer((kmax_vals + kmin) / 2, np.ones(np.shape(eval_vals)))
            k_eval_weights = np.outer((kmax_vals - kmin) / 2, weights)

            kMesh = np.einsum('i, jk->ijk', np.ones(np.shape(self.__z_vals)), k_eval_vals)
            weightMesh = np.einsum('i, jk->ijk', np.ones(np.shape(self.__z_vals)), k_eval_weights)

        else:
            k_eval_vals, k_eval_weights = self.__intHelper.get_points_weights(kmin, kmax)
            rMesh, zMesh, kMesh = np.meshgrid(self.__r_vals, self.__z_vals, k_eval_vals)
            rMesh, zMesh, weightMesh = np.meshgrid(self.__r_vals, self.__z_vals, k_eval_weights)

        integrand_base = kMesh**2*self.__growth(kMesh, zMesh)**2*self.__power(kMesh)
        integrand_sigma = integrand_base * self.window(kMesh * rMesh)**2

        sigma_sq_vals = 1/(2*np.pi**2)*np.sum(integrand_sigma*weightMesh, axis=-1)
        self.__sigma_sq_vals = sigma_sq_vals

        self.__interpolator = RectBivariateSpline(np.log10(self.__r_vals), self.__z_vals, self.__sigma_sq_vals.T, bbox=[np.min(np.log10(self.__r_vals)), np.max(np.log10(self.__r_vals)), np.min(self.__z_vals), np.max(self.__z_vals)], kx=3, ky=1, s=0)

        if do_dr and self.__window_function != 'sharp_k':
            integrand_dr = integrand_base * kMesh * self.dwindow(kMesh * rMesh) * self.window(kMesh * rMesh)
            dsigma_sq_dr = 2 / (2 * np.pi**2) * np.sum(integrand_dr * weightMesh, axis=-1)
            self.__dsigma_sq_dr_vals = dsigma_sq_dr

            self.__interpolator_dr = RectBivariateSpline(np.log10(self.__r_vals), self.__z_vals, self.__dsigma_sq_dr_vals.T, bbox=[np.min(np.log10(self.__r_vals)), np.max(np.log10(self.__r_vals)), np.min(self.__z_vals), np.max(self.__z_vals)], kx=3, ky=1, s=0)
        elif do_dr and self.__window_function == 'sharp_k':
            k_vals = np.clip(1/self.__r_vals, kmin, kmax)
            kMesh2, zMesh2 = np.meshgrid(k_vals, self.__z_vals)
            dsigma_sq_dr = - 1 / (2 * np.pi**2) * kMesh2**4 * self.__growth(kMesh2, zMesh2)**2 * self.__power(kMesh2)
            dsigma_sq_dr[:,(1/self.__r_vals < kmin) + (1/self.__r_vals > kmax)] = np.nan
            self.__dsigma_sq_dr_vals = dsigma_sq_dr

            self.__interpolator_dr = RectBivariateSpline(np.log10(self.__r_vals), self.__z_vals, self.__dsigma_sq_dr_vals.T, bbox=[np.min(np.log10(self.__r_vals)), np.max(np.log10(self.__r_vals)), np.min(self.__z_vals), np.max(self.__z_vals)], kx=3, ky=1, s=0)
        else:
            pass

        if do_dloga:
            integrand_dloga = integrand_sigma * self.__growth.f(kMesh, zMesh)
            dsigma_sq_dloga = np.sum(integrand_dloga*weightMesh, axis=-1)/(np.pi**2)
            self.__dsigma_sq_dloga_vals = dsigma_sq_dloga
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

    def dsigma_dr_of_r(self, r, z):
        return self.__dsigma_sq_dr_interpolation(r, z) / (2 * self.sigma_of_r(r, z))

    def dsigma_dr_of_m(self, m, z):
        return self.__dsigma_sq_dr_interpolation(self.radius_of_mass(m), z) / (2 * self(m, z))

    def dsigma_dm(self, m, z):
        return self.dsigma_dr_of_m(m,z) * self.radius_of_mass(m) / m / 3.0

    def dlogSigma_dlogm(self, m, z):
        return self.dsigma_dm(m, z)*m/self(m,z)

    def dlogSigma_sq_dloga_of_m(self, m, z):
        return self.__dsigma_sq_dloga_interpolation(self.radius_of_mass(m), z) / self.__sigma_sq_interpolation(self.radius_of_mass(m), z)

    def dlogSigma_dloga_of_m(self, m, z):
        return self.dlogSigma_sq_dloga_of_m(m,z)/2.0
