import numpy as np
from .cosmology import Cosmology
from.halo_bias_base import HaloBiasBase
from ..auxiliary.window_functions import WindowFunctions
from ..auxiliary.integration_helper import IntegrationHelper

from scipy.interpolate import interp2d, RectBivariateSpline
import warnings
import mcfit
from scipy.integrate import quad as spIntegrate

class HaloBias(HaloBiasBase):
    def __init__(self, cosmo, sigma, mass_function, mMin, mMax, kMin, kMax, z_vals, integrationHelper, Nk=1024, window_function='top_hat'):
        """

        :type integrationHelper: IntegrationHelper
        :type cosmo: Cosmology
        """
        super().__init__(cosmo, sigma, mass_function, mMin, mMax, kMin, kMax, z_vals, Nk=Nk, window_function=window_function)
        self.__intHelper = integrationHelper

        self.__q1Bias_vals = None
        self.__q2Bias_vals = None

        self.__interpolator_q1 = None
        self.__interpolator_q2 = None

        self._bsqm_mean = None
        self._bm_mean = None
        self._m_mean = None

        self._bsqm_assign_func = lambda z: np.nan
        self._bm_assign_func = lambda z: np.nan
        self._m_assign_func = lambda z: np.nan

        if self._window_function == "top_hat":
            self._bsq_TH = None
            self._b_TH = None
            self._n_TH = None

            self._bsq_TH_assign_func = lambda z: np.nan
            self._b_TH_assign_func = lambda z: np.nan
            self._n_TH_assign_func = lambda z: np.nan

    def mass_averaged_bias(self, k_vals, z_vals, mMin, mMax):
        rmin, rmax = self.radius_of_mass(mMin), self.radius_of_mass(mMax)

        eval_vals, weights = self.__intHelper.get_points_weights()
        kMesh, zMesh, dump = np.meshgrid(k_vals, z_vals, weights)

        if self._window_function == 'sharp_k':
            xmax = np.clip(rmax*k_vals, rmin*k_vals, 1.0)
        else:
            xmax = rmax*k_vals

        xmin = rmin*k_vals

        x_eval_vals = np.outer((xmax - xmin) / 2, eval_vals) + np.outer((xmax + xmin) / 2, np.ones(np.shape(eval_vals)))
        eval_weights = np.outer((xmax - xmin) / 2, weights)

        xMesh = np.einsum('i, jk->ijk', np.ones(np.shape(z_vals)), x_eval_vals)
        weightMesh = np.einsum('i, jk->ijk', np.ones(np.shape(z_vals)), eval_weights)

        mMesh = self.mass_of_radius(xMesh/kMesh)

        integrand_denom = 3*mMesh/xMesh * self._mass_function(mMesh, zMesh) * self.window(xMesh)**2 * weightMesh
        simpleBiasMesh = self.simple_bias(mMesh, zMesh)
        integrand_num1 = integrand_denom * simpleBiasMesh
        integrand_num2 = integrand_denom * simpleBiasMesh**2

        num1, num2, denom = np.sum(integrand_num1, axis=-1), np.sum(integrand_num2, axis=-1), np.sum(integrand_denom, axis=-1)

        self.num1,self.num2,self.denom = num1, num2, denom

        return num1/denom, num2/denom

    def compute_asymptotic(self):
        self._nbar = np.array([self.__intHelper.integrate(lambda logM: self._mass_function(np.exp(logM), z), np.log(self._mMin), np.log(self._mMax)) for z in self._z_vals])

        self._bsqm_mean = np.array([self.__intHelper.integrate(lambda logM: np.exp(logM) * self.simple_bias(np.exp(logM), z)**2 * self._mass_function(np.exp(logM), z), np.log(self._mMin), np.log(self._mMax)) for z in self._z_vals]) / self._nbar
        self._bm_mean = np.array([self.__intHelper.integrate(lambda logM: np.exp(logM) * self.simple_bias(np.exp(logM), z) * self._mass_function(np.exp(logM), z), np.log(self._mMin), np.log(self._mMax)) for z in self._z_vals]) / self._nbar
        self._m_mean = np.array([self.__intHelper.integrate(lambda logM: np.exp(logM) * self._mass_function(np.exp(logM), z), np.log(self._mMin), np.log(self._mMax)) for z in self._z_vals]) / self._nbar

        self._bsqm_assign_func = lambda z: self._bsqm_mean[np.where(z == self._z_vals[:, None])[0]].reshape(np.array(z).shape)
        self._bm_assign_func = lambda z: self._bm_mean[np.where(z == self._z_vals[:, None])[0]].reshape(np.array(z).shape)
        self._m_assign_func = lambda z: self._m_mean[np.where(z == self._z_vals[:, None])[0]].reshape(np.array(z).shape)

        if self._window_function == "top_hat":
            self._bsq_TH = np.array([self.__intHelper.integrate(lambda x: x / self.mass_of_radius(x / self._kMax) * self._mass_function(self.mass_of_radius(x / self._kMax), z) * self.simple_bias(self.mass_of_radius(x / self._kMax), z)**2, self.radius_of_mass(self._mMin) * self._kMax, self.radius_of_mass(self._mMax) * self._kMax) for z in self._z_vals])
            self._b_TH = np.array([self.__intHelper.integrate(lambda x: x / self.mass_of_radius(x / self._kMax) * self._mass_function(self.mass_of_radius(x / self._kMax), z) * self.simple_bias(self.mass_of_radius(x / self._kMax), z), self.radius_of_mass(self._mMin) * self._kMax, self.radius_of_mass(self._mMax) * self._kMax) for z in self._z_vals])

            self._n_TH = np.array([self.__intHelper.integrate(lambda x: x / self.mass_of_radius(x / self._kMax) * self._mass_function(self.mass_of_radius(x / self._kMax), z), self.radius_of_mass(self._mMin) * self._kMax, self.radius_of_mass(self._mMax) * self._kMax) for z in self._z_vals])

            self._bsq_TH_assign_func = lambda z: self._bsq_TH[np.where(z == self._z_vals[:, None])[0]].reshape(np.array(z).shape)
            self._b_TH_assign_func = lambda z: self._b_TH[np.where(z == self._z_vals[:, None])[0]].reshape(np.array(z).shape)
            self._n_TH_assign_func = lambda z: self._n_TH[np.where(z == self._z_vals[:, None])[0]].reshape(np.array(z).shape)

    def compute(self):
        self.compute_asymptotic()

        q1Bias, q2Bias = self.mass_averaged_bias(self._k_vals, self._z_vals, self._mMin, self._mMax)

        if self._window_function == 'sharp_k':
            q1Bias[:, self._k_vals >= 1.0 / self.radius_of_mass(self._mMin)] = self.simple_bias(self._mMin, self._z_vals)[:, np.newaxis]
            q2Bias[:, self._k_vals >= 1.0 / self.radius_of_mass(self._mMin)] = self.simple_bias(self._mMin, self._z_vals)[:, np.newaxis]**2
        elif self._window_function == 'gaussian':
            a, b = np.where(np.isnan(q1Bias) + np.isinf(q1Bias))  # get index of elements where bias is nan or infinity
            q1Bias[a, b] = self.simple_bias(self._mMin, self._z_vals[a])  # replace nan elements with asymptotic limit
            q2Bias[a, b] = self.simple_bias(self._mMin, self._z_vals[a])**2
        elif self._window_function == 'top_hat':
            a,b = np.where(np.isnan(q1Bias) + np.isinf(q1Bias)) #get index of elements where bias is nan or infinity
            q1Bias[a,b] = self._b_TH_assign_func(self._z_vals[a]) / self._n_TH_assign_func(self._z_vals[a]) #replace nan elements with asymptotic limit
            q2Bias[a,b] = self._b_TH_assign_func(self._z_vals[a]) / self._n_TH_assign_func(self._z_vals[a])

        self.__q1Bias_vals = q1Bias
        self.__q2Bias_vals = q2Bias

        self.__interpolator_q1 = RectBivariateSpline(np.log10(self._k_vals), self._z_vals, self.__q1Bias_vals.T, bbox=[np.min(np.log10(self._k_vals)), np.max(np.log10(self._k_vals)), np.min(self._z_vals), np.max(self._z_vals)], kx=3, ky=1, s=0)

        self.__interpolator_q2 = RectBivariateSpline(np.log10(self._k_vals), self._z_vals, self.__q2Bias_vals.T, bbox=[np.min(np.log10(self._k_vals)), np.max(np.log10(self._k_vals)), np.min(self._z_vals), np.max(self._z_vals)], kx=3, ky=1, s=0)

    def __call__(self, k, z, q):
        if q == 1:
            vals = np.squeeze(self.__interpolator_q1.ev(np.log10(k), z))

            try:
                if self._window_function == "top_hat":
                    vals[k > self._kMax] = self._b_TH_assign_func(z[k > self._kMax]) / self._n_TH_assign_func(z[k > self._kMax])
                elif self._window_function == "gaussian":
                    vals[k > self._kMax] = self.simple_bias(self._mMin, z[k > self._kMax])
                elif self._window_function == "sharp_k":
                    vals[k > self._kMax] = np.nan

                vals[k < self._kMin] = self._bm_assign_func(z[k < np.min(self._k_vals)]) / self._m_assign_func(z[k < self._kMin])

            except (IndexError, TypeError):
                if self._window_function == "top_hat":
                    vals[k > self._kMax] = self._b_TH_assign_func(z) / self._n_TH_assign_func(z)
                elif self._window_function == "gaussian":
                    vals[k > self._kMax] = self.simple_bias(self._mMin, z)
                elif self._window_function == "sharp_k":
                    vals[k > self._kMax] = np.nan

                vals[k < self._kMin] = self._bm_assign_func(z) / self._m_assign_func(z)
        elif q == 2:
            vals = np.squeeze(self.__interpolator_q2.ev(np.log10(k), z))
            try:
                if self._window_function == "top_hat":
                    vals[k > self._kMax] = self._bsq_TH_assign_func(z[k > self._kMax]) / self._n_TH_assign_func(z[k > self._kMax])
                elif self._window_function == "gaussian":
                    vals[k > self._kMax] = self.simple_bias(self._mMin, z[k > self._kMax])**2
                elif self._window_function == "sharp_k":
                    vals[k > self._kMax] = np.nan

                vals[k < self._kMin] = self._bsqm_assign_func(z[k < np.min(self._k_vals)]) / self._m_assign_func(z[k < np.min(self._k_vals)])
            except (IndexError, TypeError):
                if self._window_function == "top_hat":
                    vals[k > self._kMax] = self._bsq_TH_assign_func(z) / self._n_TH_assign_func(z)
                elif self._window_function == "gaussian":
                    vals[k > self._kMax] = self.simple_bias(self._mMin, z)**2
                elif self._window_function == "sharp_k":
                    vals[k > self._kMax] = np.nan

                vals[k < self._kMin] = self._bsqm_assign_func(z) / self._m_assign_func(z)
        else:
            raise Exception("q={} is not defined.".format(q))

        return vals


class HaloBiasFFTLog(HaloBias):
    def __init__(self, cosmo, sigma, mass_function, mMin, mMax, kMin, kMax, z_vals, Nk=1024, window_function='top_hat'):
        """

        :type integrationHelper: IntegrationHelper
        :type cosmo: Cosmology
        """
        super().__init__(cosmo, sigma, mass_function, mMin, mMax, kMin, kMax, z_vals, None, Nk=Nk, window_function=window_function)

        assert (window_function == 'top_hat' or window_function == 'gaussian')

        self.__r_vals = np.logspace(np.log10(1/kMax), np.log10(1/kMin), Nk)

        if self._window_function == 'top_hat':
            self.__transform = mcfit.TophatVar(self.__r_vals, lowring=True)
        elif self._window_function == 'gaussian':
            self.__transform = mcfit.GaussVar(self.__r_vals, lowring=True)

    def mass_averaged_bias(self, *args):
        rMesh, zMesh = np.meshgrid(self.__r_vals, self._z_vals)
        mMesh = self.mass_of_radius(rMesh)

        integrand_denom = 6*np.pi**2*mMesh/rMesh**3 * self._mass_function(mMesh, zMesh)
        integrand_denom[np.where((mMesh > self._mMax) + (mMesh < self._mMin))] = 0.0
        simpleBiasMesh = self.simple_bias(mMesh, zMesh)
        integrand_num1 = integrand_denom * simpleBiasMesh
        integrand_num2 = integrand_denom * simpleBiasMesh**2

        self._k_vals, num1 = self.__transform(integrand_num1, extrap=False)
        self._k_vals, num2 = self.__transform(integrand_num2, extrap=False)
        self._k_vals, denom = self.__transform(integrand_denom, extrap=False)

        self.num1, self.num2, self.denom = num1, num2, denom

        return num1/denom, num2/denom

