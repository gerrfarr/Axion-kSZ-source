import numpy as np

from ..theory.cosmology import Cosmology
from ..theory.halo_bias_base import HaloBiasBase
from ..theory.mass_functions import MassFunction
from ..theory.correlation_functions import CorrelationFunctions
from ..axion_camb_wrappers.power_interpolation import LinearPowerInterpolation
from ..axion_camb_wrappers.growth_interpolation import GrowthInterpolation

from ..auxiliary.integration_helper import IntegrationHelper
from ..auxiliary.window_functions import WindowFunctions

class Covariance(object):

    def __init__(self, cosmo, linear_power, growth, mass_function, bias, correlation_functions, zmin, zmax, Nz, r_vals, deltaR, f_sky, sigma_v_vals, integrationHelper, kmin=1.0e-3, kmax=1.0e2, mMin=1e14, mMax=1e16, old_bias=False):
        """

        :type correlation_functions: CorrelationFunctions
        :type bias: HaloBiasBase
        :type mass_function: MassFunction
        :type growth: GrowthInterpolation
        :type linear_power: LinearPowerInterpolation
        :type integrationHelper: IntegrationHelper
        :type cosmo: Cosmology
        """
        self.__intHelper = integrationHelper
        self.__cosmo = cosmo
        self.__linear_power = linear_power
        self.__growth = growth
        self.__mass_function = mass_function
        self.__bias = bias
        self.__correlations = correlation_functions

        self.__old_bias = old_bias

        self.__zmin = zmin
        self.__zmax = zmax
        self.__mMin = mMin
        self.__mMax = mMax
        self.__kmin = kmin
        self.__kmax = kmax
        self.__f_sky = f_sky
        self.__sigma_v_vals = sigma_v_vals # h km/s
        self.__deltaR = deltaR

        self.__r_vals = r_vals
        self.__Nz = Nz
        self.__z_edges = np.linspace(zmin, zmax, Nz + 1)
        self.__center_z = np.round((self.__z_edges[1:]+self.__z_edges[:-1])/2, 5)

        self.__number_density = np.empty(Nz)
        self.__bin_volumes = np.empty(Nz)
        self.__pair_numbers = np.empty((Nz, len(self.__r_vals)))

    def comoving_distance(self, z):
        return 3000.0 / self.__cosmo.h * self.__intHelper.integrate(lambda zp: 1 / (self.__cosmo.E(zp)), 0, z)

    def survey_volume(self, z_1, z_2):
        return 4/3 * np.pi * self.__f_sky * (self.comoving_distance(z_2) ** 3 - self.comoving_distance(z_1) ** 3)

    def number_density(self, z):
        return self.__intHelper.integrate(lambda logM: self.__mass_function(np.exp(logM), z), np.log(self.__mMin), np.log(self.__mMax))

    @staticmethod
    def number_of_pairs(r, deltaR, number_density, survey_volume, correlation):
        bin_volume = 4 / 3 * np.pi * ((r + deltaR)**3 - r**3)

        return number_density**2 * survey_volume * bin_volume * (1 + correlation) / 2
        #return number_density**2 * survey_volume * (bin_volume + 4*np.pi*r**2*correlation*deltaR)/2

    @staticmethod
    def binned_window_function(k, rmin, rmax):
        w_tilde_min = (2 * np.cos(k * rmin) + k * rmin * np.sin(k * rmin)) / (k * rmin)**3
        w_tilde_max = (2 * np.cos(k * rmax) + k * rmax * np.sin(k * rmax)) / (k * rmax)**3

        return 3 * (rmin**3 * w_tilde_min - rmax**3 * w_tilde_max) / (rmax**3 - rmin**3)

    def pre_compute(self):

        self.__number_density = np.array([self.number_density(z) for z in self.__center_z])
        self.__bin_volumes = np.array([self.survey_volume(self.__z_edges[i], self.__z_edges[i+1]) for i in range(self.__Nz)])
        rMesh, volumeMesh = np.meshgrid(self.__r_vals, self.__bin_volumes)
        dump, zMesh = np.meshgrid(self.__r_vals, self.__center_z)
        dump, nMesh = np.meshgrid(self.__r_vals, self.__number_density)

        self.__xi, self.__dbarxi_dloga = self.__correlations.get_correlation_functions(rMesh, zMesh, unbiased=False)

        self.__pair_numbers = self.number_of_pairs(rMesh, self.__deltaR, nMesh, volumeMesh, self.__xi)

    def compute(self, shot_noise=True, cosmic_variance=True):
        raw_pairs = np.array(np.meshgrid(self.__r_vals, self.__r_vals)).T.reshape((len(self.__r_vals)**2, 2))
        r_pairs = raw_pairs[np.where(raw_pairs[:,0]<=raw_pairs[:,1])[0]]

        r1Mesh, zMesh = np.meshgrid(r_pairs[:,0], self.__center_z)
        r2Mesh, dump = np.meshgrid(r_pairs[:,1], self.__center_z)
        volumeMesh = np.einsum('i,j->ij', self.__bin_volumes, np.ones(r_pairs.shape[0]))

        prefactors = 4/(np.pi**2*volumeMesh) * 100.0**2/(1+self.__correlations.get_correlation_functions(r1Mesh, zMesh)[0])/(1+self.__correlations.get_correlation_functions(r2Mesh, zMesh)[0])*self.__growth.f(1.0e-3, zMesh)**2

        eval_vals_logk, weights = self.__intHelper.get_points_weights(np.log(self.__kmin), np.log(self.__kmax))
        logkMesh = np.einsum('i, jk->jki', eval_vals_logk, np.ones(zMesh.shape))
        weightMesh = np.einsum('i, jk->jki', weights, np.ones(zMesh.shape))
        kMesh = np.exp(logkMesh)

        zIntMesh = np.einsum('i, jk->jki', np.ones(eval_vals_logk.shape), zMesh)
        r1Mesh = np.einsum('j, ijk->ijk', r_pairs[:, 0], np.ones(logkMesh.shape))
        r2Mesh = np.einsum('j, ijk->ijk', r_pairs[:, 1], np.ones(logkMesh.shape))
        number_density_mesh = np.einsum('i,ijk->ijk', self.__number_density, np.ones(logkMesh.shape))

        windowMesh1 = self.binned_window_function(kMesh, r1Mesh - self.__deltaR / 2, r1Mesh + self.__deltaR / 2)
        windowMesh2 = self.binned_window_function(kMesh, r2Mesh - self.__deltaR / 2, r2Mesh + self.__deltaR / 2)

        if not self.__old_bias:
            integrand = kMesh * (cosmic_variance*self.__growth(kMesh, zIntMesh)**2 * self.__linear_power(kMesh) * self.__bias(kMesh,zIntMesh, 1) * self.__bias(kMesh,zIntMesh, 0) + shot_noise/number_density_mesh)**2*windowMesh1*windowMesh2*weightMesh
            #integrand = kMesh * self.__growth.f(kMesh, zIntMesh)**2 * (1 / number_density_mesh)**2 * windowMesh1 * windowMesh2 * weightMesh
            #integrand = kMesh * self.__growth.f(kMesh, zIntMesh)**2 * (self.__growth(kMesh, zIntMesh)**2 * self.__linear_power(kMesh) * self.__bias(kMesh, zIntMesh, 1) * self.__bias(kMesh, zIntMesh, 0))**2 * windowMesh1 * windowMesh2 * weightMesh
        else:
            integrand = kMesh * (cosmic_variance*self.__growth(kMesh, zIntMesh)**2 * self.__linear_power(kMesh) * self.__bias(kMesh, zIntMesh, 1) + shot_noise / number_density_mesh)**2 * windowMesh1 * windowMesh2 * weightMesh
            #integrand = kMesh * self.__growth.f(kMesh, zIntMesh)**2 * (1 / number_density_mesh)**2 * windowMesh1 * windowMesh2 * weightMesh
            #integrand = kMesh * self.__growth.f(kMesh, zIntMesh)**2 * (self.__growth(kMesh, zIntMesh)**2 * self.__linear_power(kMesh) * self.__bias(kMesh, zIntMesh, 1))**2 * windowMesh1 * windowMesh2 * weightMesh
        integral = np.sum(integrand, axis=-1)

        output = prefactors*integral

        #reshaping into covariance matrix
        a = raw_pairs[:, 0] == r_pairs[:, 0, np.newaxis]
        b = raw_pairs[:, 1] == r_pairs[:, 1, np.newaxis]
        c = raw_pairs[:, 0] == r_pairs[:, 1, np.newaxis]
        d = raw_pairs[:, 1] == r_pairs[:, 0, np.newaxis]

        r_ia, r_ib = np.where(a*b+c*d)
        return output[:,r_ia[r_ib.argsort()]].reshape(len(self.__center_z), len(self.__r_vals), len(self.__r_vals))


    def noise_terms(self):
        dump, sigma_vMesh = np.meshgrid(self.__r_vals, self.__sigma_v_vals)
        return 2*sigma_vMesh**2/self.__pair_numbers