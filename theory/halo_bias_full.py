import numpy as np
from .cosmology import Cosmology
from.halo_bias_base import HaloBiasBase
from ..auxiliary.window_functions import WindowFunctions
from ..auxiliary.integration_helper import IntegrationHelper
from ..axion_camb_wrappers.growth_interpolation import GrowthInterpolation

from scipy.interpolate import interp2d, RectBivariateSpline
import warnings
import mcfit
from scipy.integrate import quad as spIntegrate
from scipy.integrate import cumtrapz as spCumIntegrate


class HaloBias(HaloBiasBase):
    def __init__(self, cosmo, growth, sigma, mass_function, mMin, mMax, kMin, kMax, z_vals, integrationHelper, Nk=1024, window_function='top_hat'):
        """

        :type integrationHelper: IntegrationHelper
        :type cosmo: Cosmology
        :type growth: GrowthInterpolation
        """
        super().__init__(cosmo, sigma, mass_function, mMin, mMax, kMin, kMax, z_vals, Nk=Nk, window_function=window_function)
        self.__intHelper = integrationHelper
        self.__growth = growth

        self.__bias_vals = None
        self.__n_vals = None

        self.__interpolator_B = None
        self.__interpolator_N = None

        self._b_mean = None
        self._m_mean = None

        self._b_assign_func = lambda z: np.nan
        self._m_assign_func = lambda z: np.nan

    def mass_averaged_bias(self, k_vals, z_vals, mMin, mMax):
        rmin, rmax = self.radius_of_mass(mMin), self.radius_of_mass(mMax)

        eval_vals, weights = self.__intHelper.get_points_weights()
        kMesh, zMesh, dump = np.meshgrid(k_vals, z_vals, weights)
        dump, nBarMesh = np.meshgrid(k_vals, self._nbar)

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

        base_integrand = 3/xMesh * self._mass_function(mMesh, zMesh) * weightMesh * self.window(xMesh)
        integrandN = base_integrand * (1-self.simple_bias(mMesh, zMesh)) * self._sigma.dlogSigma_dloga_of_m(mMesh, zMesh)
        integrandB = base_integrand * self.simple_bias(mMesh, zMesh)

        valsN, valsB = np.sum(integrandN, axis=-1), np.sum(integrandB, axis=-1)

        return valsB / nBarMesh, valsN / nBarMesh

    def compute_asymptotic(self):
        self._nbar = np.array([self.__intHelper.integrate(lambda logM: self._mass_function(np.exp(logM), z), np.log(self._mMin), np.log(self._mMax)) for z in self._z_vals])

        self._b_mean = np.array([self.__intHelper.integrate(lambda logM: self.simple_bias(np.exp(logM), z) * self._mass_function(np.exp(logM), z), np.log(self._mMin), np.log(self._mMax)) for z in self._z_vals]) / self._nbar

        self._n_mean = np.array([self.__intHelper.integrate(lambda logM: (1-self.simple_bias(np.exp(logM), z)) * self._sigma.dlogSigma_dloga_of_m(np.exp(logM), z) * self._mass_function(np.exp(logM), z), np.log(self._mMin), np.log(self._mMax)) for z in self._z_vals]) / self._nbar

        self._m_mean = np.array([self.__intHelper.integrate(lambda logM: np.exp(logM) * self._mass_function(np.exp(logM), z), np.log(self._mMin), np.log(self._mMax)) for z in self._z_vals]) / self._nbar

        self._b_assign_func = lambda z: self._b_mean[np.where(z.flatten() == self._z_vals[:, None])[0]].reshape(np.array(z).shape)
        self._n_assign_func = lambda z: self._n_mean[np.where(z.flatten() == self._z_vals[:, None])[0]].reshape(np.array(z).shape)
        self._m_assign_func = lambda z: self._m_mean[np.where(z.flatten() == self._z_vals[:, None])[0]].reshape(np.array(z).shape)

    def compute(self):
        self.compute_asymptotic()

        self.__bias_vals, self.__n_vals = self.mass_averaged_bias(self._k_vals, self._z_vals, self._mMin, self._mMax)

        self.__interpolator_B = RectBivariateSpline(np.log10(self._k_vals), self._z_vals, self.__bias_vals.T, bbox=[np.min(np.log10(self._k_vals)), np.max(np.log10(self._k_vals)), np.min(self._z_vals), np.max(self._z_vals)], kx=3, ky=1, s=0)
        self.__interpolator_N = RectBivariateSpline(np.log10(self._k_vals), self._z_vals, self.__n_vals.T, bbox=[np.min(np.log10(self._k_vals)), np.max(np.log10(self._k_vals)), np.min(self._z_vals), np.max(self._z_vals)], kx=3, ky=1, s=0)

    """def compute_approximation(self, N=50000):
        self.compute_asymptotic()

        if self._window_function=="top_hat" or self._window_function=="gaussian":
            self.__interpolator_B = interp_emulator(lambda logk, z: self._b_assign_func(z)*self.window(10**logk*self.radius_of_mass(self._m_assign_func(z))))
            self.__interpolator_N = interp_emulator(lambda logk, z: self.window(10**logk * self.radius_of_mass(self._m_assign_func(z))))

        elif self._window_function=="sharp_k":
            ms = np.logspace(np.log10(self._mMin), np.log10(self._mMax), N)
            mMesh,zMesh = np.meshgrid(ms, self._z_vals)
            dump,nBarMesh = np.meshgrid(ms, self._nbar)

            self.__bias_vals = np.flip(spCumIntegrate(self.simple_bias(mMesh, zMesh) * self._mass_function(mMesh, zMesh), np.log(ms), initial=0.0) / nBarMesh, axis=-1)
            self.__n_vals = np.flip(spCumIntegrate(self._mass_function(mMesh, zMesh), np.log(ms), initial=0.0) / nBarMesh, axis=-1)

            self._k_vals = np.flip(1/self.radius_of_mass(ms))
            self.__interpolator_B = RectBivariateSpline(np.log10(self._k_vals), self._z_vals, self.__bias_vals.T, bbox=[np.min(np.log10(self._k_vals)), np.max(np.log10(self._k_vals)), np.min(self._z_vals), np.max(self._z_vals)], kx=3, ky=1, s=0)
            self.__interpolator_N = RectBivariateSpline(np.log10(self._k_vals), self._z_vals, self.__n_vals.T, bbox=[np.min(np.log10(self._k_vals)), np.max(np.log10(self._k_vals)), np.min(self._z_vals), np.max(self._z_vals)], kx=3, ky=1, s=0)
        else:
            raise Exception("Unknown Window function: "+str(self._window_function))"""

    def __call__(self, k, z, q, approximate=False):
        if q == 0:
            vals = np.squeeze(self.__interpolator_N.ev(np.log10(k), z)/self.__growth.f(k,z))
            vals[k > np.max(self._k_vals)] = 0.0
            try:
                vals[k < np.min(self._k_vals)] = self._n_assign_func(z[k < np.min(self._k_vals)])/self.__growth.f(k[k < np.min(self._k_vals)],z[k < np.min(self._k_vals)])
            except (IndexError, TypeError):
                vals[k < np.min(self._k_vals)] = self._n_assign_func(z)/self.__growth.f(k[k < np.min(self._k_vals)],z)

            vals+=self(k,z,1)
        elif q == 1:
            vals = np.squeeze(self.__interpolator_B.ev(np.log10(k), z))
            vals[k > np.max(self._k_vals)] = 0.0
            try:
                vals[k < np.min(self._k_vals)] = self._b_assign_func(z[k < np.min(self._k_vals)])
            except (IndexError, TypeError):
                vals[k < np.min(self._k_vals)] = self._b_assign_func(z)
        else:
            raise Exception("q={} is not defined.".format(q))

        return vals

    def mean_bias(self, z):
        return self._b_assign_func(z)

    def mean_mass(self, z):
        return self._m_assign_func(z)



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
            MK = mcfit.kernels.Mellin_Tophat(3, 0)
            self.__transform = mcfit.mcfit(self.__r_vals, MK, 1.5, lowring=True)
            self.__transform.prefac *= self.__transform.x**3
        elif self._window_function == 'gaussian':
            MK = mcfit.kernels.Mellin_Gauss(3, 0)
            self.__transform = mcfit.mcfit(self.__r_vals, MK, 1.5, lowring=True)
            self.__transform.prefac *= self.__transform.x**3

    def mass_averaged_bias(self, *args):
        rMesh, zMesh = np.meshgrid(self.__r_vals, self._z_vals)
        dump, nBarMesh = np.meshgrid(self.__r_vals, self._nbar)
        mMesh = self.mass_of_radius(rMesh)

        integrand1 = 3.0/rMesh**3 * self._mass_function(mMesh, zMesh)
        integrand1[np.where((mMesh > self._mMax) + (mMesh < self._mMin))] = 0.0
        integrand2 = integrand1 * self.simple_bias(mMesh, zMesh)

        self._k_vals, vals1 = self.__transform(integrand1, extrap=False)
        self._k_vals, vals2 = self.__transform(integrand2, extrap=False)

        return vals2 / nBarMesh, vals1 / nBarMesh


class interp_emulator:
    def __init__(self, func):
        self.__func = func

    def ev(self, *args):
        return self.__func(*args)