import numpy as np
from scipy.interpolate import interp1d as interpolate

class LinearPowerInterpolation(object):

    def __init__(self, pk_vals_file, extrap_kmax=1e3, extrap_kmin=1e-5):
        P_CAMB = np.loadtxt(pk_vals_file)

        kmax = np.max(P_CAMB[:, 0])
        kmin = np.min(P_CAMB[:, 0])
        self.__logk = np.log(P_CAMB[:, 0])
        self.__logP = np.log(P_CAMB[:, 1])
        if extrap_kmax and extrap_kmax > kmax:
            self.__logk = np.hstack([self.__logk, np.log(kmax) * 0.1 + np.log(extrap_kmax) * 0.9,np.log(extrap_kmax)])
            logPnew = np.empty((len(self.__logP) + 2))
            logPnew[:-2] = self.__logP
            diff = (logPnew[-3] - logPnew[-4]) / (self.__logk[-3] - self.__logk[-4])
            if np.any(diff) < 0:
                raise ValueError("No log extrapolation possible! divergent behavior")

            delta = diff * (self.__logk[-1] - self.__logk[-3])
            logPnew[-1] = logPnew[-3] + delta
            logPnew[-2] = logPnew[-3] + delta * 0.9

            self.__logP=logPnew

        if extrap_kmin and extrap_kmin < kmin:
            self.__logk = np.hstack([np.log(extrap_kmin), np.log(kmin) * 0.1 + np.log(extrap_kmin) * 0.9, self.__logk])

            logPnew = np.empty((len(self.__logP) + 2))
            logPnew[2:] = self.__logP
            diff = (logPnew[3] - logPnew[2]) / (self.__logk[3] - self.__logk[2])
            if np.any(diff) < 0:
                raise ValueError("No log extrapolation possible! divergent behavior")

            delta = diff * (self.__logk[0] - self.__logk[2])
            logPnew[0] = logPnew[2] + delta
            logPnew[1] = logPnew[2] + delta * 0.9

            self.__logP = logPnew

        self.__Pk0_interp = interpolate(self.__logk, self.__logP)

    def __call__(self, k):
        """

        :param k: k in units of h/Mpc
        :return: present day linear power spectrum in units of (h^{-1} Mpc)^3
        """
        try:
            return np.exp(self.__Pk0_interp(np.log(k)))
        except ValueError as ex:
            print(ex)
            print(np.min(k), np.max(k))
            raise ValueError("A value in x_new is outside the interpolation range")