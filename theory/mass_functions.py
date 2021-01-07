from .sigma_interpolation import SigmaInterpolator
from .cosmology import Cosmology

import numpy as np

class PressSchechterMassFunction(object):
    def __init__(self, cosmo, sigmaInterpolator):
        """

        :type sigmaInterpolator: SigmaInterpolator
        :type cosmo: Cosmology
        """
        self.__cosmo = cosmo
        self.__sigmaInt = sigmaInterpolator

    def __call__(self, m, z):
        return np.sqrt(2 / np.pi) * (self.__cosmo.rho_mean * self.__cosmo.delta_crit / self.__sigmaInt(m, z) / m**2) * np.fabs(self.__sigmaInt.dlogSigma_dlogm(m, z)) * np.exp(-self.__cosmo.delta_crit**2 / (2 * self.__sigmaInt(m, z)**2))
