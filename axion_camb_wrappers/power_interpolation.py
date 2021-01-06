import numpy as np
from scipy.interpolate import interp1d as interpolate

class LinearPowerInterpolation(object):

    def __init__(self, pk_vals_file):
        P_CAMB = np.loadtxt(pk_vals_file)

        self.__Pk0_interp = interpolate(np.log10(P_CAMB[:, 0]), P_CAMB[:, 1])

    def __call__(self, k):
        """

        :param k: k in units of h/Mpc
        :return: present day linear power spectrum in units of (h^{-1} Mpc)^3
        """
        return self.__Pk0_interp(np.log10(k))