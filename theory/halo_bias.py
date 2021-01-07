import numpy as np
from .cosmology import Cosmology
from ..auxiliary.window_functions import WindowFunctions
from ..auxiliary.integration_helper import IntegrationHelper

from scipy.interpolate import interp2d, RectBivariateSpline
import warnings

class HaloBias(object):
    def __init__(self, cosmo, sigma, mass_function, mMin, mMax, kMin, kMax, z_vals, integrationHelper, Nk=1000, window_function='top_hat'):
        """

        :type integrationHelper: IntegrationHelper
        :type cosmo: Cosmology
        """
        self.__cosmo = cosmo
        self.__sigma = sigma
        self.__mass_function = mass_function
        self.__intHelper = integrationHelper
        self.__window_function = window_function

        self.window = None
        self.dwindow = None
        self.radius_of_mass = None
        self.mass_of_radius = None

        WindowFunctions.set_window_functions(self, self.__window_function, self.__cosmo)

        if window_function == 'sharp_k' and kMax>1/self.radius_of_mass(mMin):
            warnings.warn(f"The given value of k_max={kMax:.2E} is not feasible because you chose a sharp-k filter and a maximum mass of m_min={mMin:.2E}. k_max has instead be set to {1/self.radius_of_mass(mMin):.2E}", RuntimeWarning)
            kMax = 1.0/self.radius_of_mass(mMin)

        self.__k_vals = np.logspace(np.log10(kMin), np.log10(kMax), Nk)
        self.__z_vals = z_vals
        self.__mMin, self.__mMax = mMin, mMax

        self.__q1Bias_vals = None
        self.__q2Bias_vals = None

        self.__interpolator_q1 = None
        self.__interpolator_q2 = None

    def simple_bias(self, m_vals, z_vals):
        return 1+(self.__cosmo.delta_crit**2 - self.__sigma(m_vals, 0.0)**2)/(self.__sigma(m_vals, 0.0)*self.__sigma(m_vals, z_vals)*self.__cosmo.delta_crit)

    def mass_averaged_bias(self, k_vals, z_vals, mMin, mMax, smart=True):
        if self.__window_function == 'sharp_k' and smart:
            eval_vals, weights = self.__intHelper.get_points_weights()
            rmin, rmax = self.radius_of_mass(mMin), self.radius_of_mass(mMax)
            rmax_vals = np.clip(1 / k_vals, rmin, rmax)
            logMMax_vals = np.log(self.mass_of_radius(rmax_vals))
            kMesh, zMesh, dump = np.meshgrid(k_vals, z_vals, weights)

            logM_eval_vals = np.outer((logMMax_vals - np.log(mMin)) / 2, eval_vals) + np.outer((logMMax_vals + np.log(mMin)) / 2, np.ones(np.shape(eval_vals)))
            eval_weights = np.outer((logMMax_vals - np.log(mMin)) / 2, weights)

            logMMesh = np.einsum('i, jk->ijk', np.ones(np.shape(z_vals)), logM_eval_vals)
            weightMesh = np.einsum('i, jk->ijk', np.ones(np.shape(z_vals)), eval_weights)
        else:
            logM_eval_vals, eval_weights = self.__intHelper.get_points_weights(np.log(mMin), np.log(mMax))
            kMesh, zMesh, logMMesh = np.meshgrid(k_vals, z_vals, logM_eval_vals)
            kMesh, zMesh, weightMesh = np.meshgrid(k_vals, z_vals, eval_weights)

        mMesh = np.exp(logMMesh)

        integrand_denom_log = mMesh**2 * self.__mass_function(mMesh, zMesh) * self.window(kMesh * self.radius_of_mass(mMesh))**2
        simpleBiasMesh = self.simple_bias(mMesh, zMesh)
        integrand_num_log1 = integrand_denom_log * simpleBiasMesh
        integrand_num_log2 = integrand_denom_log * simpleBiasMesh**2

        num1, num2, denom = np.sum(integrand_num_log1*weightMesh, axis=-1), np.sum(integrand_num_log2*weightMesh, axis=-1), np.sum(integrand_denom_log*weightMesh, axis=-1)
        return num1/denom, num2/denom

    def compute(self):
        q1Bias, q2Bias = self.mass_averaged_bias(self.__k_vals, self.__z_vals, self.__mMin, self.__mMax)

        if self.__window_function == 'sharp_k':
            q1Bias[:, self.__k_vals>=1.0/self.radius_of_mass(self.__mMin)] = self.simple_bias(self.__mMin, self.__z_vals)[:,np.newaxis]
            q2Bias[:, self.__k_vals>=1.0/self.radius_of_mass(self.__mMin)] = self.simple_bias(self.__mMin, self.__z_vals)[:,np.newaxis]**2

        self.__q1Bias_vals = q1Bias
        self.__q2Bias_vals = q2Bias

        self.__interpolator_q1 =  RectBivariateSpline(np.log10(self.__k_vals), self.__z_vals, self.__q1Bias_vals.T, bbox=[np.min(np.log10(self.__k_vals)), np.max(np.log10(self.__k_vals)), np.min(self.__z_vals), np.max(self.__z_vals)], kx=3, ky=1, s=0)

        self.__interpolator_q2 = RectBivariateSpline(np.log10(self.__k_vals), self.__z_vals, self.__q2Bias_vals.T, bbox=[np.min(np.log10(self.__k_vals)), np.max(np.log10(self.__k_vals)), np.min(self.__z_vals), np.max(self.__z_vals)], kx=3, ky=1, s=0)

    def __call__(self, k, z, q):
        if q==1:
            vals = np.squeeze(self.__interpolator_q1.ev(np.log10(k), z))
        elif q==2:
            vals = np.squeeze(self.__interpolator_q2.ev(np.log10(k), z))
        else:
            raise Exception("q={} is not defined.".format(q))

        vals[k > np.max(self.__k_vals)] = np.nan
        vals[k < np.min(self.__k_vals)] = np.nan
        return vals

