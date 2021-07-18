import numpy as np
from scipy.interpolate import pchip, RectBivariateSpline
import warnings
import matplotlib.pyplot as plt
from scipy.special import spherical_jn as bessel_j

from ..theory.cosmology import Cosmology, CosmologyCustomH
from ..axion_camb_wrappers.growth_interpolation import GrowthInterpolation
from ..axion_camb_wrappers.power_interpolation import LinearPowerInterpolation
from ..axion_camb_wrappers.run_axion_camb import AxionCAMBWrapper

from ..auxiliary.integration_helper import IntegrationHelper
import camb

from scipy.interpolate import interp1d as interpolate
from scipy.integrate import cumtrapz


def compute_ov_spectra(cosmo, axionCAMBWrapper, integrationHelper=None):
    """

    Parameters
    ----------
    integrationHelper : IntegrationHelper
    axionCAMBWrapper : AxionCAMBWrapper
    cosmo : Cosmology
    """
    ###SET COMPUTATION OPTIONS
    SINGULARITY = True  # use substitution to deal with singularity in S(k) integrand

    power = axionCAMBWrapper.get_linear_power(extrap_kmin=1e-5, extrap_kmax=1e6)
    growth = axionCAMBWrapper.get_growth()
    hubble = axionCAMBWrapper.get_hubble()

    dG_dt = lambda k, z: hubble(z)*growth(k, z)*growth.f(k,z)

    dw_dz = lambda z: 3000.0 / (cosmo.h * cosmo.E(z))
    z_vis = np.linspace(0, 20, 5000)
    w_of_z_vals = cumtrapz(dw_dz(z_vis), z_vis, initial=0.0)
    w_of_z = interpolate(z_vis, w_of_z_vals)

    pars = camb.set_params(H0=100 * cosmo.h, ombh2=cosmo.omegaB, omch2=cosmo.omegaCDM, ns=cosmo.n_s, tau=cosmo.tau, mnu=0.0, nnu=3.046, omk=0, num_massive_neutrinos=0)
    data = camb.get_background(pars)
    back_ev = data.get_background_redshift_evolution(z_vis, ['x_e', 'visibility'], format='array')
    g_interpolation = interpolate(z_vis, back_ev[:, 1])
    norm = integrationHelper.integrate(lambda z: g_interpolation(z) * dw_dz(z), min(z_vis), max(z_vis))
    g_func = lambda z: (1 - np.exp(-cosmo.tau)) / norm * g_interpolation(z)

    def S_integrand(k, X_T, U, z):
        def x_integrand(k, x, y, z):
            factorA = (1 - x**2)
            factorB = 1 + y**2 - 2 * x * y
            sqrt_factorB = np.sqrt(factorB)
            power_evals = power(k * y) * power(k * sqrt_factorB)

            numerator = factorA * (dG_dt(z, k * sqrt_factorB) * growth(z, k * y) * y**2 - dG_dt(z, k * y) * growth(z, k * sqrt_factorB) * factorB)**2
            denominator = factorB**2

            return power_evals * numerator / denominator

        if SINGULARITY:
            integrand = lambda t, u: 10**u * 2 / cosmo.n_s * t**(2 / cosmo.n_s - 1) * x_integrand(k, 1.0 - t**(2 / cosmo.n_s), 10**u, z)
            return np.log(10) * integrand(X_T, U)
        else:
            integrand2 = lambda x, u: 10**u * x_integrand(k, x, 10**u, z)
            return np.log(10) * integrand2(X_T, U)

    def vish(k, z):
        lk = np.log10(k)
        if SINGULARITY:
            return k * (integrationHelper.integrate2D(lambda t, u: S_integrand(k, t, u, z), 0, 2**(cosmo.n_s / 2), min([-7 - lk, -1]), 0) + integrationHelper.integrate2D(lambda t, u: S_integrand(k, t, u, z), 0, 2**(cosmo.n_s / 2), 0, max([3 - lk, 1])))
        else:
            return k * (integrationHelper.integrate2D(lambda x, u: S_integrand(k, x, u, z), -1, 1, min([-7 - lk, -1]), 0) + integrationHelper.integrate2D(lambda x, u: S_integrand(k, x, u, z), -1, 1, 0, max([3 - lk, 1])))

    def C_proj(ell, S, zmin=0, zmax=20):
        integrand_a = lambda a: g_func(1/a-1)**2 / w_of_z(1/a-1)**2 * S(a, (ell + 1 / 2) / w_of_z(1/a-1)) * dw_dz(1/a-1)
        integrand_loga = lambda loga: 10**loga * np.log(10) * integrand_a(10**loga)
        # integrand_z=lambda z:g(z)**2/w(z)**2*a(z)**2*S(1/(1+z), (ell+1/2)/w(z))*dw_dz(z)

        return 1 / (16 * np.pi**2) * integrationHelper.integrate(integrand_loga, np.log10(1 / (1 + zmax)), np.log10(1 / (1 + zmin)))

    a_vals = np.logspace(np.log10(1 / (1 + max(z_vis))), 0, 100)
    z_vals = (1 - a_vals) / a_vals
    k_vals = np.logspace(-2, 3, 300) * cosmo.h  # physical k

    vish_vals = np.full((len(z_vals), len(k_vals)), np.nan)
    for k_i,k in enumerate(k_vals):
        vish_vals[:,k_i] = vish(k, z_vals)

    interpolation = RectBivariateSpline(np.log10(a_vals), np.log10(k_vals), np.log10(vish_vals))
    vish_interp = lambda a, k: 10**np.squeeze(interpolation.ev(np.log10(np.array([a])), np.log10(np.array([k]))))

    l_vals = np.logspace(0, 5, 300)
    pp_vals = np.full(len(l_vals), np.nan)
    for l_i, l in l_vals:
        pp_vals[l_i] = C_proj(l, vish_interp)

    return pp_vals
