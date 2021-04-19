from ..axion_camb_wrappers.power_interpolation import LinearPowerInterpolation
from ..axion_camb_wrappers.growth_interpolation import GrowthInterpolation
from ..auxiliary.integration_helper import IntegrationHelper
from .cosmology import Cosmology
from .halo_bias_base import HaloBiasBase
import mcfit

import numpy as np
from scipy.interpolate import interp1d

class CorrelationFunctions(object):

    def __init__(self, cosmo, linear_power, growth, halo_bias, kMin, kMax, z_vals, rMin, integrationHelper, Nk=1024):
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
        self.__k_vals = np.logspace(np.log10(self.__kMin), np.log10(self.__kMax), Nk)
        self.__p2xi = mcfit.P2xi(self.__k_vals, lowring=True)

        self.__rMin = rMin

        self.__z_vals = z_vals
        self.__r_vals = None

        self.__intHelper = integrationHelper

        self.__xi_unbiased = None
        self.__xi = None
        self.__dbarxi_dloga_unbiased = None
        self.__dbarxi_dloga = None

        self.__xi_unbiased_interp = None
        self.__xi_interp = None
        self.__dbarxi_dloga_unbiased_interp = None
        self.__dbarxi_dloga_interp = None

        self.__computed=False
        self.__computed_unbiased = False

    def compute(self, unbiased=False, old_bias=False):
        if unbiased:
            self.__r_vals, self.__xi_unbiased, dxi_dloga_unbiased, self.__xi, dxi_dloga = self.compute_xi(self.__z_vals, deriv=True, unbiased=True, old_bias=old_bias)

            self.__dbarxi_dloga = self.compute_dbarxi_dloga(self.__r_vals, self.__z_vals, dxi_dloga)
            self.__dbarxi_dloga_unbiased = self.compute_dbarxi_dloga(self.__r_vals, self.__z_vals, dxi_dloga_unbiased)

            self.__computed = True
            self.__computed_unbiased = True

            self.__xi_unbiased_interp = interp1d(self.__r_vals, self.__xi_unbiased)
            self.__dbarxi_dloga_unbiased_interp = interp1d(self.__r_vals, self.__dbarxi_dloga_unbiased)

        else:
            self.__r_vals, self.__xi, dxi_dloga = self.compute_xi(self.__z_vals, deriv=True, unbiased=False, old_bias=old_bias)

            self.__dbarxi_dloga = self.compute_dbarxi_dloga(self.__r_vals, self.__z_vals, dxi_dloga)

            self.__computed = True

        self.__xi_interp = interp1d(self.__r_vals, self.__xi)
        self.__dbarxi_dloga_interp = interp1d(self.__r_vals, self.__dbarxi_dloga)

    def get_correlation_functions(self, r_vals, z_vals, unbiased=False):
        z_vals = np.asarray(z_vals)
        r_vals = np.asarray(r_vals)
        assert(np.all(np.isin(z_vals, self.__z_vals)))
        assert(self.__computed)
        assert(np.max(r_vals)<=np.max(self.__r_vals) and np.min(r_vals)>=np.min(self.__r_vals))
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
            out_xi = self.__xi_interp(r_flat)[zindexa[zindexb.argsort()], rind].reshape(shape)
            out_dbarxi_dloga = self.__dbarxi_dloga_interp(r_flat)[zindexa[zindexb.argsort()], rind].reshape(shape)

            if unbiased:
                out_xi_unbiased = self.__xi_unbiased_interp(r_flat)[zindexa[zindexb.argsort()], rind].reshape(shape)
                out_dbarxi_dloga_unbiased = self.__dbarxi_dloga_unbiased_interp(r_flat)[zindexa[zindexb.argsort()], rind].reshape(shape)
                return out_xi_unbiased, out_xi, out_dbarxi_dloga_unbiased, out_dbarxi_dloga
            else:
                return out_xi, out_dbarxi_dloga

    def compute_xi(self, z, deriv=True, unbiased=False, old_bias=False):
        kMesh, zMesh = np.meshgrid(self.__k_vals, z)

        linear_power = self.__linear_power(kMesh)*self.__growth(kMesh, zMesh)**2
        if old_bias:
            non_linear_power = linear_power * self.__halo_bias(kMesh, zMesh, 2)
        else:
            non_linear_power = linear_power * self.__halo_bias(kMesh, zMesh, 1)**2

        if deriv:
            linear_deriv = 2 * linear_power * self.__growth.f(kMesh, zMesh)
            if old_bias:
                non_linear_deriv = linear_deriv * self.__halo_bias(kMesh, zMesh, 1)
            else:
                non_linear_deriv = linear_deriv * self.__halo_bias(kMesh, zMesh, 1) * self.__halo_bias(kMesh, zMesh, 0)

        r, nonlin_xi = self.__p2xi(non_linear_power)
        if unbiased:
            r, lin_xi = self.__p2xi(linear_power)
            if deriv:
                r, nonlin_dxi = self.__p2xi(non_linear_deriv)
                r, lin_dxi = self.__p2xi(linear_deriv)
                return r, lin_xi, lin_dxi, nonlin_xi, nonlin_dxi
            else:
                return r, lin_xi, nonlin_xi
        else:
            if deriv:
                r, nonlin_dxi = self.__p2xi(non_linear_deriv)
                return r, nonlin_xi, nonlin_dxi
            else:
                return r, nonlin_xi

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




