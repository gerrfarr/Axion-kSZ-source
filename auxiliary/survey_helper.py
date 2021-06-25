import numpy as np

class SurveyType(object):
    def __init__(self, zMin, zMax, Nz, mMin, mMax, f_sky, sigma_v, delta_tau_sq=0.15):

        self.__zMin = zMin
        self.__zMax = zMax
        self.__Nz = Nz

        self.__z_edges = np.linspace(zMin, zMax, Nz + 1)
        self.__center_z = np.round((self.__z_edges[1:] + self.__z_edges[:-1]) / 2, 5)

        self.__mMin = mMin
        self.__mMax = mMax

        self.__f_sky = f_sky
        assert(len(sigma_v)==Nz)
        self.__sigma_v = np.sqrt(sigma_v**2 + (delta_tau_sq/0.15)*120**2)

    @property
    def zMin(self):
        return self.__zMin

    @property
    def zMax(self):
        return self.__zMax

    @property
    def Nz(self):
        return self.__Nz

    @property
    def z_edges(self):
        return self.__z_edges

    @property
    def center_z(self):
        return self.__center_z

    @property
    def mMin(self):
        return self.__mMin

    @property
    def mMax(self):
        return self.__mMax

    @property
    def f_sky(self):
        return self.__f_sky

    @property
    def sigma_v(self):
        return self.__sigma_v

    @staticmethod
    def overlap2f_sky(overlap_area):
        return overlap_area / 360.0**2 * np.pi

class StageSuper(SurveyType):
    def __init__(self, fid_cosmo, delta_tau_sq=0.15):
        sigma_v = np.array([15.0, 22.0, 27.0, 34.0, 42.0])  # km/s
        super().__init__(0.1, 0.6, 5, 1e13*fid_cosmo.h, 1e16*fid_cosmo.h, self.overlap2f_sky(1e4), sigma_v, delta_tau_sq=delta_tau_sq)

class StageIV(SurveyType):
    def __init__(self, fid_cosmo, delta_tau_sq=0.15):
        sigma_v=np.array([15.0, 22.0, 27.0, 34.0, 42.0]) #km/s
        super().__init__(0.1, 0.6, 5, 0.6e14*fid_cosmo.h, 1e16*fid_cosmo.h, self.overlap2f_sky(1e4), sigma_v, delta_tau_sq=delta_tau_sq)

class StageIII(SurveyType):
    def __init__(self, fid_cosmo, delta_tau_sq=0.15):
        sigma_v=np.array([100, 150, 190])#km/s
        super().__init__(0.1, 0.4, 3, 1e14*fid_cosmo.h, 1e16*fid_cosmo.h, self.overlap2f_sky(0.6e4), sigma_v, delta_tau_sq=delta_tau_sq)

class StageII(SurveyType):
    def __init__(self, fid_cosmo, delta_tau_sq=0.15):
        sigma_v=np.array([290, 440, 540])#km/s
        super().__init__(0.1, 0.4, 3, 1e14*fid_cosmo.h, 1e16*fid_cosmo.h, self.overlap2f_sky(0.4e4), sigma_v, delta_tau_sq=delta_tau_sq)