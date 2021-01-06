import numpy as np
from scipy.interpolate import RectBivariateSpline

class GrowthInterpolation(object):

    def __init__(self, file_root, degree=3, smoothing=0.0):
        self.__f_data = np.loadtxt(file_root + '_devolution.dat')
        self.__delta_data = np.loadtxt(file_root + '_evolution.dat')
        a_data = np.loadtxt(file_root + '_a_vals.dat')
        self.__a_vals = a_data[:, 0]
        assert (a_data[-1, 0] == 1.0)
        self.__k_vals = np.loadtxt(file_root + '_matterpower_out.dat')[:, 0]

        normalized_dataset = self.__delta_data / np.meshgrid(self.__a_vals, self.__delta_data[:, -1])[1]

        G_interpolation = RectBivariateSpline(np.log10(self.__k_vals), np.log10(self.__a_vals), normalized_dataset, bbox=[min(np.log10(self.__k_vals)), max(np.log10(self.__k_vals)), min(np.log10(self.__a_vals)), max(np.log10(self.__a_vals))], kx=degree, ky=degree, s=smoothing)
        self.__G_func = lambda k, a: np.squeeze(G_interpolation.ev(np.log10(k), np.log10(a)))

        f_interpolation = RectBivariateSpline(np.log10(self.__k_vals), np.log10(self.__a_vals), self.__f_data, bbox=[min(np.log10(self.__k_vals)), max(np.log10(self.__k_vals)), min(np.log10(self.__a_vals)), max(np.log10(self.__a_vals))], kx=degree, ky=degree, s=smoothing)
        self.__f_func = lambda k, a: np.squeeze(f_interpolation.ev(np.log10(k), np.log10(a)))

    def __call__(self, k, z):
        return self.__G_func(k, 1/(1+z))

    def f(self, k, z):
        return self.__f_func(k, 1/(1+z))