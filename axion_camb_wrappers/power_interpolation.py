import numpy as np
from scipy.interpolate import interp1d as interpolate

class LinearPowerInterpolation(object):

    def __init__(self, pk_vals_file, extrap_kmax=1e3):
        P_CAMB = np.loadtxt(pk_vals_file)

        kmax = np.max(P_CAMB[:, 0])
        logk = np.log(P_CAMB[:, 0])
        logP = np.log(P_CAMB[:, 1])
        if extrap_kmax and extrap_kmax > kmax:
            logk = np.hstack([logk, np.log(kmax) * 0.1 + np.log(extrap_kmax) * 0.9,np.log(extrap_kmax)])
            logPnew = np.empty((len(P_CAMB) + 2))
            logPnew[:-2] = logP
            diff = (logPnew[-3] - logPnew[-4]) / (logk[-3] - logk[-4])
            if np.any(diff) < 0:
                raise ValueError("No log extrapolation possible! divergent behavior")

            delta = diff * (logk[-1] - logk[-3])
            logPnew[-1] = logPnew[-3] + delta
            logPnew[-2] = logPnew[-3] + delta * 0.9

            logP=logPnew

        self.__Pk0_interp = interpolate(logk, logP)

    def __call__(self, k):
        """

        :param k: k in units of h/Mpc
        :return: present day linear power spectrum in units of (h^{-1} Mpc)^3
        """
        return np.exp(self.__Pk0_interp(np.log(k)))