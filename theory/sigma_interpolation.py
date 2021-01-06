from typing import List

from ..axion_camb_wrappers.power_interpolation import LinearPowerInterpolation
from ..axion_camb_wrappers.growth_interpolation import GrowthInterpolation
from ..auxiliary.window_functions import WindowFunctions
from ..auxiliary.integration_helper import IntegrationHelper
from .cosmology import Cosmology
import numpy as np
from scipy.interpolate import interp1d

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
        self.__window_function=window_function

        if window_function=='top_hat':
            self.__window=WindowFunctions.top_hat
            self.__radius_of_mass = lambda m: WindowFunctions.radius_of_mass_top_hat(m)/self.__cosmo.rho_mean**(1/3)
            self.__mass_of_radius = lambda r: WindowFunctions.mass_of_radius_top_hat(r)*self.__cosmo.rho_mean

        elif window_function=='gaussian':
            self.__window = WindowFunctions.gaussian
            self.__radius_of_mass = lambda m: WindowFunctions.radius_of_mass_gaussian(m) / self.__cosmo.rho_mean**(1 / 3)
            self.__mass_of_radius = lambda r: WindowFunctions.mass_of_radius_gaussian(r) * self.__cosmo.rho_mean

        elif window_function=='sharp_k':
            self.__window = WindowFunctions.sharp_k
            self.__radius_of_mass = lambda m: WindowFunctions.radius_of_mass_sharp_k(m) / self.__cosmo.rho_mean**(1 / 3)
            self.__mass_of_radius = lambda r: WindowFunctions.mass_of_radius_sharp_k(r) * self.__cosmo.rho_mean

        elif window_function=='none':
            self.__window = WindowFunctions.no_window
            self.__radius_of_mass = lambda m: WindowFunctions.radius_of_mass_top_hat(m) / self.__cosmo.rho_mean**(1 / 3)
            self.__mass_of_radius = lambda r: WindowFunctions.mass_of_radius_top_hat(r) * self.__cosmo.rho_mean

        else:
            raise Exception("Unknown window function: " +str(window_function))

        self.__r_vals = np.logspace(np.log10(self.__radius_of_mass(mMin)), np.log10(self.__radius_of_mass(mMax)), Nr)
        self.__z_vals = z_vals
        self.__sigma_sq_vals = np.zeros((len(self.__r_vals), len(self.__z_vals)))

        self.__interpolators=[None for i in range(len(self.__z_vals))]

    def compute(self, kmin, kmax, smart=True):
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

        print(np.shape(weightMesh))
        integrand = kMesh**2*self.__growth(kMesh, zMesh)**2*self.__power(kMesh)*self.__window(kMesh*rMesh)**2

        sigma_sq_vals = 1/(2*np.pi**2)*np.sum(integrand*weightMesh, axis=-1)
        self.__sigma_sq_vals = sigma_sq_vals

        for z_i, z in enumerate(self.__z_vals):
            self.__interpolators[z_i] = interp1d(np.log10(self.__r_vals), np.sqrt(self.__sigma_sq_vals[z_i]))

    def __sigma_interpolation(self, r, z):
        assert(np.min(np.fabs(self.__z_vals-z))<1.0e-5)
        return self.__interpolators[np.argmin(np.fabs(self.__z_vals-z))](np.log10(r))

    def __call__(self, m, z):
        return self.__sigma_interpolation(self.__radius_of_mass(m), z)

    def sigma_of_r(self, r, z):
        return self.__sigma_interpolation(r, z)

