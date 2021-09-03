import numpy as np
from scipy.interpolate import RectBivariateSpline

class GrowthInterpolation(object):

    def __init__(self, file_root, degree=3, smoothing=0.0, w=1e-4):
        self.__f_data = np.loadtxt(file_root + '_devolution.dat')

        a_data = np.loadtxt(file_root + '_a_vals.dat')
        self.__a_vals = a_data[:, 0]
        #assert (a_data[-1, 0] == 1.0)
        self.__k_vals = np.loadtxt(file_root + '_matterpower_out.dat')[:, 0]

        """
        self.__delta_data = np.loadtxt(file_root + '_evolution.dat')
        normalized_dataset = self.__delta_data / np.meshgrid(self.__a_vals, self.__delta_data[:, -1])[1]

        self.__G_interpolation = RectBivariateSpline(np.log(self.__k_vals), np.log(self.__a_vals), normalized_dataset, bbox=[min(np.log(self.__k_vals)), max(np.log(self.__k_vals)), min(np.log(self.__a_vals)), max(np.log(self.__a_vals))], kx=degree, ky=degree, s=smoothing)
        self.__G_func = lambda k, a: np.squeeze(self.__G_interpolation.ev(np.log(k), np.log(a)))"""

        self.__f_interpolation = RectBivariateSpline(np.log(self.__k_vals), np.log(self.__a_vals), self.__f_data, bbox=[min(np.log(self.__k_vals)), max(np.log(self.__k_vals)), min(np.log(self.__a_vals)), max(np.log(self.__a_vals))], kx=degree, ky=degree, s=smoothing)
        self.__f_func = lambda k, a: np.squeeze(self.__f_interpolation.ev(np.log(k), np.log(a)))

        self.__logD_norm_vals = np.zeros((len(self.__k_vals), len(self.__a_vals)))
        for i, k in enumerate(self.__k_vals):
            for j, a in enumerate(self.__a_vals):
                if i!=0 and i!=len(self.__k_vals)-1:
                    self.__logD_norm_vals[i, j] = self.__f_interpolation.integral(np.log(k) - w, np.log(k) + w, 0, np.log(a)) / (2 * w)
                elif i==0:
                    self.__logD_norm_vals[i, j] = self.__f_interpolation.integral(np.log(k), np.log(k) + w, 0, np.log(a)) / w
                elif i==len(self.__k_vals)-1:
                    self.__logD_norm_vals[i, j] = self.__f_interpolation.integral(np.log(k) - w, np.log(k), 0, np.log(a)) / w


        self.__G_interpolation = RectBivariateSpline(np.log(self.__k_vals), np.log(self.__a_vals), self.__logD_norm_vals, bbox=[min(np.log(self.__k_vals)), max(np.log(self.__k_vals)), min(np.log(self.__a_vals)), max(np.log(self.__a_vals))], kx=degree, ky=degree, s=smoothing)
        self.__G_func = lambda k, a: np.squeeze(np.exp(self.__G_interpolation.ev(np.log(k), np.log(a))))

    def __call__(self, k, z):
        return self.__G_func(k, 1/(1+z))

    def f(self, k, z):
        return self.__f_func(k, 1/(1+z))

    @property
    def kMin(self):
        return np.min(self.__k_vals)

    @property
    def kMax(self):
        return np.max(self.__k_vals)