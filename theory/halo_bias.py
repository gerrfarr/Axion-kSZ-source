import numpy as np
from .cosmology import Cosmology
from ..auxiliary.window_functions import WindowFunctions
from ..auxiliary.integration_helper import IntegrationHelper

class HaloBias(object):
    def __init__(self, cosmo, sigma_func, integrationHelper, window_function='top_hat'):
        """

        :type integrationHelper: IntegrationHelper
        :type cosmo: Cosmology
        """
        self.__cosmo = cosmo
        self.__sigma = sigma_func
        self.__intHelper = integrationHelper

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


    def __simple_bias(self, m_vals, z_vals):
        return 1+(self.__cosmo.delta_crit**2 - self.__sigma(0.0, m_vals)**2)/(self.__sigma(0.0, m_vals)*self.__sigma(z_vals, m_vals)*self.__cosmo.delta_crit)

    def __mass_averaged_bias(self, k_vals, z, mMin, mMax, n, q):
        mass_eval_vals, mass_eval_weights = self.__intHelper.get_points_weights(mMin, mMax)
        evalMesh_M, evalMesh_k = np.meshgrid(mass_eval_vals, k_vals)
        integrand = evalMesh_M*n(evalMesh_M)*self.__simple_bias(evalMesh_M, z)**q*self.__window(evalMesh_k * evalMesh_M)**2
        return np.sum(integrand*mass_eval_weights, axis=-1)