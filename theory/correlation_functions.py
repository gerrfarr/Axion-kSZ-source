from ..axion_camb_wrappers.power_interpolation import LinearPowerInterpolation
from ..axion_camb_wrappers.growth_interpolation import GrowthInterpolation
from ..auxiliary.integration_helper import IntegrationHelper
from .cosmology import Cosmology
from .halo_bias_base import HaloBiasBase

import numpy as np
from scipy.interpolate import interp1d

class CorrelationFunctions(object):

    def __init__(self, cosmo, linear_power, growth, halo_bias, kMin, kMax, z_vals, rMin, r_vals, integrationHelper, Nr=1024):
        """

        :type integrationHelper: IntegrationHelper
        :type halo_bias: HaloBiasBase
        :type growth: GrowthInterpolation
        :type linear_power: LinearPowerInterpolation
        :type cosmo: Cosmology
        """

        assert (rMin >= 1 / kMax)

        self.__cosmo = cosmo
        self.__linear_power=linear_power
        self.__growth = growth
        self.__halo_bias = halo_bias

        self.__kMin = kMin
        self.__kMax = kMax

        self.__rMin = rMin

        self.__z_vals = z_vals
        self.__r_vals = np.unique(np.concatenate((np.linspace(rMin, np.max(r_vals), Nr), r_vals)))
        self.__r_selection = np.where(self.__r_vals[:,np.newaxis] == r_vals)[0]

        self.__intHelper = integrationHelper

        self.__xi_unbiased = None
        self.__xi = None
        self.__dbarxi_dloga_unbiased = None
        self.__dbarxi_dloga = None

        self.__computed=False
        self.__computed_unbiased = False

    def compute(self, unbiased=False, old_bias=False):
        if unbiased:
            self.__xi_unbiased, dxi_dloga_unbiased, self.__xi, dxi_dloga = self.compute_xi(self.__r_vals, self.__z_vals, deriv=True, unbiased=True, old_bias=old_bias)

            self.__dbarxi_dloga = self.compute_dbarxi_dloga(self.__r_vals, self.__z_vals, dxi_dloga)
            self.__dbarxi_dloga_unbiased = self.compute_dbarxi_dloga(self.__r_vals, self.__z_vals, dxi_dloga_unbiased)

            self.__computed = True
            self.__computed_unbiased = True

            return self.__xi_unbiased[:, self.__r_selection], self.__xi[:, self.__r_selection], self.__dbarxi_dloga_unbiased[:, self.__r_selection], self.__dbarxi_dloga[:, self.__r_selection]
        else:
            self.__xi, dxi_dloga = self.compute_xi(self.__r_vals, self.__z_vals, deriv=True, unbiased=False, old_bias=old_bias)

            self.__dbarxi_dloga = self.compute_dbarxi_dloga(self.__r_vals, self.__z_vals, dxi_dloga)

            self.__computed = True

            return self.__xi[:, self.__r_selection], self.__dbarxi_dloga[:, self.__r_selection]

    def get_correlation_functions(self, r_vals, z_vals, unbiased=False):
        z_vals = np.asarray(z_vals)
        r_vals = np.asarray(r_vals)
        assert(np.all(np.isin(z_vals, self.__z_vals)) and self.__computed and np.max(r_vals)<=np.max(self.__r_vals) and np.min(r_vals)>=np.min(self.__r_vals))
        assert(z_vals.shape == r_vals.shape or z_vals.shape == () or r_vals.shape == ())
        if r_vals.shape == ():
            shape = z_vals.shape
        else:
            shape = r_vals.shape

        r_flat = r_vals.flatten()
        z_flat = z_vals.flatten()

        rindexa, rindexb = np.where(r_flat == self.__r_vals[:, np.newaxis])
        zindexa, zindexb = np.where(z_flat == self.__z_vals[:, np.newaxis])

        if np.all(np.isin(r_vals, self.__r_vals)):
            out_xi = self.__xi[zindexa[zindexb.argsort()], rindexa[rindexb.argsort()]].reshape(shape)
            out_dbarxi_dloga = self.__dbarxi_dloga[zindexa[zindexb.argsort()], rindexa[rindexb.argsort()]].reshape(shape)

            if unbiased:
                out_xi_unbiased = self.__xi_unbiased[zindexa[zindexb.argsort()], rindexa[rindexb.argsort()]].reshape(shape)
                out_dbarxi_dloga_unbiased = self.__dbarxi_dloga_unbiased[zindexa[zindexb.argsort()], rindexa[rindexb.argsort()]].reshape(shape)
                return out_xi_unbiased, out_xi, out_dbarxi_dloga_unbiased, out_dbarxi_dloga
            else:
                return out_xi, out_dbarxi_dloga
        else:
            rind = np.arange(0, len(r_flat), 1, dtype=np.int)
            out_xi = interp1d(self.__r_vals, self.__xi)(r_flat)[zindexa[zindexb.argsort()], rind].reshape(shape)
            out_dbarxi_dloga = interp1d(self.__r_vals, self.__dbarxi_dloga)(r_flat)[zindexa[zindexb.argsort()], rind].reshape(shape)

            if unbiased:
                out_xi_unbiased = interp1d(self.__r_vals, self.__xi_unbiased)(r_flat)[zindexa[zindexb.argsort()], rind].reshape(shape)
                out_dbarxi_dloga_unbiased = interp1d(self.__r_vals, self.__dbarxi_dloga_unbiased)(r_flat)[zindexa[zindexb.argsort()], rind].reshape(shape)
                return out_xi_unbiased, out_xi, out_dbarxi_dloga_unbiased, out_dbarxi_dloga
            else:
                return out_xi, out_dbarxi_dloga

    def compute_xi(self, r, z, deriv=True, unbiased=False, old_bias=False):
        eval_vals, weights = self.__intHelper.get_points_weights()
        xmin_vals, xmax_vals = self.__kMin*r, self.__kMax*r

        rMesh, zMesh, dump = np.meshgrid(r, z, weights)

        x_eval_vals = np.outer((xmax_vals - xmin_vals) / 2, eval_vals) + np.outer((xmax_vals + xmin_vals) / 2, np.ones(np.shape(eval_vals)))
        eval_weights = np.outer((xmax_vals - xmin_vals) / 2, weights)

        xMesh = np.einsum('i, jk->ijk', np.ones(np.shape(z)), x_eval_vals)
        weightMesh = np.einsum('i, jk->ijk', np.ones(np.shape(z)), eval_weights)

        kMesh = xMesh/rMesh

        integrand_unbiased = xMesh*np.sin(xMesh)*self.__linear_power(kMesh)*self.__growth(kMesh, zMesh)**2*weightMesh
        if old_bias:
            integrand_biased = integrand_unbiased * self.__halo_bias(kMesh, zMesh, 2)
        else:
            integrand_biased = integrand_unbiased * self.__halo_bias(kMesh, zMesh, 1)**2

        if deriv:
            integrand_deriv_unbiased = integrand_unbiased * self.__growth.f(kMesh, zMesh)
            if old_bias:
                integrand_deriv_biased = integrand_unbiased * self.__growth.f(kMesh, zMesh) * self.__halo_bias(kMesh, zMesh, 1)
            else:
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




