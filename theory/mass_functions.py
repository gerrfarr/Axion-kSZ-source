from .sigma_interpolation import SigmaInterpolator
from .cosmology import Cosmology

import numpy as np

class MassFunction(object):
    def __init__(self, cosmo, sigmaInterpolator):
        """

                :type sigmaInterpolator: SigmaInterpolator
                :type cosmo: Cosmology
                """
        self.cosmo = cosmo
        self.sigmaInt = sigmaInterpolator

class PressSchechterMassFunction(MassFunction):
    def __call__(self, m, z):
        #output is dN/d log m
        vals = np.sqrt(2 / np.pi) * (self.cosmo.rho_mean * self.cosmo.delta_crit / self.sigmaInt(m, z) / m) * np.fabs(self.sigmaInt.dlogSigma_dlogm(m, z)) * np.exp(-self.cosmo.delta_crit**2 / (2 * self.sigmaInt(m, z)**2))
        return vals

class JenkinsMassFunction(MassFunction):
    def __call__(self, m, z):
        raise NotImplementedError("This is not implemented yet!")
