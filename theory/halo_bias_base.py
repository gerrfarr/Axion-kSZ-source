import numpy as np
from .cosmology import Cosmology
from ..auxiliary.window_functions import WindowFunctions
from ..auxiliary.integration_helper import IntegrationHelper

from scipy.interpolate import interp2d, RectBivariateSpline
import warnings

class HaloBiasBase(object):
    def __init__(self, cosmo, sigma, mass_function, mMin, mMax, kMin, kMax, z_vals, Nk=1024, window_function='top_hat'):
        """

                :type integrationHelper: IntegrationHelper
                :type cosmo: Cosmology
                """
        self.__cosmo = cosmo
        self.__sigma = sigma
        self._mass_function = mass_function
        self._window_function = window_function

        self.window = None
        self.dwindow = None
        self.radius_of_mass = None
        self.mass_of_radius = None

        WindowFunctions.set_window_functions(self, self._window_function, self.__cosmo)

        if window_function == 'sharp_k' and kMax > 1 / self.radius_of_mass(mMin):
            warnings.warn(f"The given value of k_max={kMax:.2E} is not feasible because you chose a sharp-k filter and a maximum mass of m_min={mMin:.2E}. k_max has instead be set to {1 / self.radius_of_mass(mMin):.2E}", RuntimeWarning)
            kMax = 1.0 / self.radius_of_mass(mMin)

        self._k_vals = np.logspace(np.log10(kMin), np.log10(kMax), Nk)
        self._z_vals = z_vals
        self._mMin, self._mMax = mMin, mMax

    def simple_bias(self, m_vals, z_vals):
        return 1+(self.__cosmo.delta_crit**2 - self.__sigma(m_vals, 0.0)**2)/(self.__sigma(m_vals, 0.0)*self.__sigma(m_vals, z_vals)*self.__cosmo.delta_crit)

    def mass_averaged_bias(self, *args, **kwargs):
        raise NotImplementedError("Use sub-classes")

    def compute(self):
        raise NotImplementedError("Use sub-classes")

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Use sub-classes")