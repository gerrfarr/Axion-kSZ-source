from ..axion_camb_wrappers.power_interpolation import LinearPowerInterpolation
from ..axion_camb_wrappers.growth_interpolation import GrowthInterpolation
from ..auxiliary.integration_helper import IntegrationHelper
from .cosmology import Cosmology
from .halo_bias_base import HaloBiasBase

import numpy as np
from scipy.interpolate import interp1d

class CorrelationFunctions(object):

    def __init(self, cosmo, linear_power, growth, halo_bias, kMin, kMax, z_vals, rMin, rMax, integrationHelper, Nr=1000):
        """

        :type integrationHelper: IntegrationHelper
        :type halo_bias: HaloBiasBase
        :type growth: GrowthInterpolation
        :type linear_power: LinearPowerInterpolation
        :type cosmo: Cosmology
        """

        self.__cosmo = cosmo
        self.__linear_power=linear_power
        self.__growth = growth
        self.__halo_bias = halo_bias

        self.__kMin = kMin
        self.__kMax = kMax

        self.__rMin = rMin

        self.__z_vals = z_vals
        self.__r_vals = np.linspace(rMin, rMax, Nr)

        self.__intHelper = integrationHelper

    def compute(self, unbiased=False):
        if unbiased:
            xi_unbiased, dxi_dloga_unbiased, xi, dxi_dloga = self.compute_xi(self.__r_vals, self.__z_vals, deriv=True, unbiased=True)

            dbarxi_dloga = self.compute_dbarxi_dloga(self.__r_vals, self.__z_vals, dxi_dloga)
            dbarxi_dloga_unbiased = self.compute_dbarxi_dloga(self.__r_vals, self.__z_vals, dxi_dloga_unbiased)

            return xi_unbiased, xi, dbarxi_dloga_unbiased, dbarxi_dloga
        else:
            xi, dxi_dloga = self.compute_xi(self.__r_vals, self.__z_vals, deriv=True, unbiased=False)

            dbarxi_dloga = self.compute_dbarxi_dloga(self.__r_vals, self.__z_vals, dxi_dloga)

            return xi, dbarxi_dloga

    def compute_xi(self, r, z, deriv=True, unbiased=False):
        eval_vals, weights = self.__intHelper.get_points_weights()
        xmin_vals, xmax_vals = self.__kMin*r, self.__kMax*r

        rMesh, zMesh, dump = np.meshgrid(r, z, weights)

        x_eval_vals = np.outer((xmax_vals - xmin_vals) / 2, eval_vals) + np.outer((xmax_vals + xmin_vals) / 2, np.ones(np.shape(eval_vals)))
        eval_weights = np.outer((xmax_vals - xmin_vals) / 2, weights)

        xMesh = np.einsum('i, jk->ijk', np.ones(np.shape(z)), x_eval_vals)
        weightMesh = np.einsum('i, jk->ijk', np.ones(np.shape(z)), eval_weights)

        kMesh = xMesh/rMesh

        integrand_unbiased = xMesh*np.sin(xMesh)*self.__linear_power(kMesh)*self.__growth(kMesh, zMesh)**2*weightMesh
        integrand_biased = integrand_unbiased * self.__halo_bias(kMesh, zMesh, 1)**2

        if deriv:
            integrand_deriv_unbiased = integrand_unbiased * self.__growth.f(kMesh, zMesh)
            integrand_deriv_biased = integrand_unbiased * self.__growth.f(kMesh, zMesh) * self.__halo_bias(kMesh, zMesh, 1) * self.__halo_bias(kMesh, zMesh, 0)

        if unbiased:
            if deriv:
                return np.sum(integrand_unbiased, axis=-1) / (2 * np.pi**2 * r**3), np.sum(integrand_deriv_unbiased, axis=-1) / (np.pi**2 * r**3), np.sum(integrand_biased, axis=-1) / (2 * np.pi**2 * r**3), np.sum(integrand_deriv_biased, axis=-1) / (np.pi**2 * r**3)
            else:
                return np.sum(integrand_unbiased, axis=-1) / (2*np.pi**2*r**3), np.sum(integrand_biased, axis=-1) / (2*np.pi**2*r**3)
        else:
            if deriv:
                return np.sum(integrand_biased, axis=-1) / (2 * np.pi**2 * r**3), np.sum(integrand_deriv_biased, axis=-1) / (np.pi**2 * r**3)
            else:
                return np.sum(integrand_biased, axis=-1) / (2 * np.pi**2 * r**3)

    def compute_dbarxi_dloga(self, r, z, dxi_dloga_input):
        assert(dxi_dloga_input.shape == (len(z), len(r)))
        eval_vals, weights = self.__intHelper.get_points_weights()

        r_eval_vals = np.outer((r - self.__rMin) / 2, eval_vals) + np.outer((r + self.__rMin) / 2, np.ones(np.shape(eval_vals)))
        eval_weights = np.outer((r - self.__rMin) / 2, weights)

        rMesh = np.einsum('i, jk->ijk', np.ones(np.shape(z)), r_eval_vals)
        weightMesh = np.einsum('i, jk->ijk', np.ones(np.shape(z)), eval_weights)

        dxi_dloga_vals = interp1d(r, dxi_dloga_input)(r_eval_vals)

        integrand = rMesh**2*dxi_dloga_vals*weightMesh

        return 3*np.sum(integrand, axis=-1)/r**3


