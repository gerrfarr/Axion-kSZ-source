import numpy as np

from ..theory.cosmology import Cosmology, CosmologyCustomH
from ..axion_camb_wrappers.growth_interpolation import GrowthInterpolation
from ..axion_camb_wrappers.power_interpolation import LinearPowerInterpolation
from ..axion_camb_wrappers.run_axion_camb import AxionCAMBWrapper

from ..auxiliary.integration_helper import IntegrationHelper
import camb
from scipy.interpolate import RectBivariateSpline, interp1d as interpolate
from scipy.integrate import cumtrapz

class OV_spectra:

    def __init__(self, cosmo, axionCAMBWrapper, integrationHelper=None, kMin=1.0e-4, kMax=1.0e2):
        """
        Parameters
        ----------
        integrationHelper : IntegrationHelper
        axionCAMBWrapper : AxionCAMBWrapper
        cosmo : Cosmology
        """

        self.__cosmo = cosmo
        self.__power = axionCAMBWrapper.get_linear_power()
        self.__growth = axionCAMBWrapper.get_growth()

        self.__intHelper = integrationHelper

        self.__kMin, self.__kMax = kMin, kMax

        self.__k_vals = None
        self.__z_vals = None
        self.__vish = None

    def compute_S(self, k_vals, z_vals, grid=True):

        vals, weights = self.__intHelper.get_points_weights(-1,1)

        t_vals = 2**(self.__cosmo.n_s/2)/2 * vals + 2**(self.__cosmo.n_s/2)/2
        t_weights = 2**(self.__cosmo.n_s/2)/2 * weights
        x_vals = 1-t_vals**(2/self.__cosmo.n_s)

        ymin_vals, ymax_vals = self.__kMin/k_vals, self.__kMax/k_vals
        logY_vals = np.empty((len(k_vals), 2*len(vals)))
        logY_weights = np.empty((len(k_vals), 2*len(vals)))

        half_way = np.zeros(ymin_vals.shape)
        half_way[np.where((ymax_vals<1.0) | (1.0<ymin_vals))] = ((np.log(ymax_vals)+np.log(ymin_vals))/2)[np.where((ymax_vals<1.0) | (1.0<ymin_vals))]

        logY_vals[:, :len(vals)] = np.outer((half_way - np.log(ymin_vals)) / 2, vals) + np.outer((half_way + np.log(ymin_vals)) / 2, np.ones_like(vals))
        logY_weights[:, :len(vals)] = np.outer((half_way - np.log(ymin_vals)) / 2, weights)
        logY_vals[:, len(vals):] = np.outer((np.log(ymax_vals) - half_way) / 2, vals) + np.outer((np.log(ymax_vals) + half_way) / 2, np.ones_like(vals))
        logY_weights[:, len(vals):] = np.outer((np.log(ymax_vals) - half_way) / 2, weights)

        """for i, k in enumerate(k_vals):
            if ymin_vals[i] < 1.0 < ymax_vals[i]:
                logY_vals[i, :len(vals)] = (0 - np.log(ymin_vals[i])) / 2 * vals + (0 + np.log(ymin_vals[i])) / 2
                logY_vals[i, len(vals):] = (np.log(ymax_vals[i]) - 0) / 2 * vals + (np.log(ymax_vals[i]) + 0) / 2

                logY_weights[i, :len(vals)] = (0 - np.log(ymin_vals[i])) / 2 * weights
                logY_weights[i, len(vals):] = (np.log(ymax_vals[i]) - 0) / 2 * weights
            else:
                half_way = (np.log(ymax_vals[i])+np.log(ymin_vals[i]))/2
                logY_vals[i, :len(vals)] = (half_way - np.log(ymin_vals[i])) / 2 * vals + (half_way + np.log(ymin_vals[i])) / 2
                logY_vals[i, len(vals):] = (np.log(ymax_vals[i]) - half_way) / 2 * vals + (np.log(ymax_vals[i]) + half_way) / 2

                logY_weights[i, :len(vals)] = (half_way - np.log(ymin_vals[i])) / 2 * weights
                logY_weights[i, len(vals):] = (np.log(ymax_vals[i]) - half_way) / 2 * weights"""


        """logY_vals = np.outer((np.log(ymax_vals)-np.log(ymin_vals))/2,vals) + np.outer((np.log(ymax_vals)+np.log(ymin_vals))/2,np.ones_like(vals))
        logY_weights = np.outer((np.log(ymax_vals)-np.log(ymin_vals))/2,weights)"""
        y_vals = np.exp(logY_vals)

        factor_A_sq = 1-2*x_vals[np.newaxis, np.newaxis, :]*y_vals[:, :, np.newaxis] + y_vals[:, :, np.newaxis]**2
        factor_A = np.sqrt(factor_A_sq)
        k_i,y_i,x_i = np.where((self.__kMin<=k_vals[:, np.newaxis, np.newaxis]*factor_A)&(k_vals[:, np.newaxis, np.newaxis]*factor_A<=self.__kMax))

        out = np.zeros((len(z_vals),*factor_A.shape))

        ky = (k_vals[k_i]*y_vals[k_i,y_i])
        kA = (k_vals[k_i]*factor_A[k_i,y_i,x_i])
        if not grid and k_vals.shape == z_vals.shape:
            zs = z_vals[k_i]
        else:
            zs = z_vals[:,np.newaxis]

        out[:,k_i,y_i,x_i] = t_vals[x_i]**(2/self.__cosmo.n_s-1) * y_vals[k_i, y_i] * self.__power(kA) * self.__power(ky) * self.__growth(kA, zs)**2 * self.__growth(ky, zs)**2 \
                             * (1-x_vals[x_i]**2)/factor_A_sq[k_i,y_i,x_i]**2 * (self.__cosmo.E(zs)/3000.0)**2 \
                             * (self.__growth.f(kA, zs) * y_vals[k_i,y_i]**2 - self.__growth.f(ky, zs) * factor_A_sq[k_i,y_i,x_i])**2

        self.__vish = 2/self.__cosmo.n_s * k_vals[np.newaxis, :] * np.sum(np.sum(out * t_weights[np.newaxis, np.newaxis, np.newaxis, :], axis=-1) * logY_weights[np.newaxis, :, :], axis=-1)
        self.__k_vals = k_vals
        self.__z_vals = z_vals
        return self.__vish

    def get_kernel_functions(self, zmax=20.0, N_z=5000):
        dw_dz = lambda z: 3000.0 / (self.__cosmo.E(z))
        z_vis = np.linspace(0, zmax, N_z)
        w_of_z_vals = cumtrapz(dw_dz(z_vis), z_vis, initial=0.0)
        w_of_z = interpolate(z_vis, w_of_z_vals)
        z_of_w = interpolate(w_of_z_vals, z_vis)

        pars = camb.set_params(H0=100 * self.__cosmo.h, ombh2=self.__cosmo.omegaB, omch2=self.__cosmo.omegaDM, ns=self.__cosmo.n_s, tau=self.__cosmo.tau, mnu=0.0, nnu=3.046, omk=0, num_massive_neutrinos=0)
        data = camb.get_background(pars)
        back_ev = data.get_background_redshift_evolution(z_vis, ['x_e', 'visibility'], format='array')
        g_interpolation = interpolate(z_vis, back_ev[:, 1])
        norm = self.__intHelper.integrate(lambda z: g_interpolation(z) * dw_dz(z), min(z_vis), max(z_vis))
        g_func = lambda z: (1 - np.exp(-self.__cosmo.tau)) / norm * g_interpolation(z)

        interp = RectBivariateSpline(1 / (1 + self.__z_vals), np.log(self.__k_vals), np.log(self.__vish), kx=1, ky=1)
        vish_func = lambda z, k: np.exp(interp.ev(1 / (1 + z), np.log(k)))

        return g_func, w_of_z, dw_dz, z_of_w, vish_func

    def compute_OV(self, ell_vals, zmax=20.0, N_z=5000):

        g_func, w_of_z, dw_dz, z_of_w, vish_func = self.get_kernel_functions(zmax=zmax, N_z=N_z)

        z_vals, z_weights = self.__intHelper.get_points_weights(0, zmax)

        ellMesh, zMesh = np.meshgrid(ell_vals, z_vals)
        kMesh = (ellMesh+1/2)/w_of_z(zMesh)
        aMesh = 1/(1+zMesh)
        cond = np.where((self.__kMin<=kMesh)&(kMesh<=self.__kMax))

        out = np.zeros(ellMesh.shape)

        out[cond] = g_func(zMesh[cond])**2/w_of_z(zMesh[cond])**2 * vish_func(zMesh[cond], kMesh[cond]) * dw_dz(zMesh[cond]) * aMesh[cond]**2

        return 1 / (16 * np.pi**2) * np.sum(out * z_weights[:, np.newaxis], axis=0)

