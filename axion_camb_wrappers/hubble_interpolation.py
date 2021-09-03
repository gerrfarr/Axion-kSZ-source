import numpy as np
from scipy.interpolate import interp1d as interpolate

class HubbleInterpolation(object):

    def __init__(self, file_path):
        data = np.loadtxt(file_path)

        self.__H_interp = interpolate(data[:,0], data[:,1], fill_value='extrapolate')

    def __call__(self, z):
        return self.__H_interp(1/(1+z))
