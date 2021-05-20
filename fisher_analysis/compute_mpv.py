from ..theory.cosmology import Cosmology,CosmologyCustomH
from ..theory.sigma_interpolation import SigmaInterpolator
from ..theory.sigma_interpolation_FFTLog import SigmaInterpolatorFFTLog
from ..theory.halo_bias import HaloBias
from ..theory.halo_bias_new import HaloBias as HaloBiasNew
from ..theory.halo_bias_full import HaloBias as HaloBiasFull
from ..theory.mass_functions import JenkinsMassFunction, PressSchechterMassFunction
from ..theory.correlation_functions import CorrelationFunctions
from ..theory.correlation_functions_FFTLog import CorrelationFunctions as CorrelationFunctionsFFTLog

from ..axion_camb_wrappers.growth_interpolation import GrowthInterpolation
from ..axion_camb_wrappers.power_interpolation import LinearPowerInterpolation
from ..axion_camb_wrappers.run_axion_camb import AxionCAMBWrapper

from ..auxiliary.integration_helper import IntegrationHelper
from ..auxiliary.survey_helper import SurveyType

import numpy as np


def compute_mean_pairwise_velocity(r_vals, rMin, cosmo, axionCAMB_wrapper, survey, window="gaussian", sigma_window=None, old_bias=False, full_bias=False, jenkins_mass=False, integrationHelper=None, kMin=1.0e-4, kMax=1.0e2, do_unbiased=False, get_correlation_functions=False, use_approximations=False, use_FFTLog=False):
    """

    Parameters
    ----------
    axionCAMB_wrapper : AxionCAMBWrapper
    survey : SurveyType
    cosmo : CosmologyCustomH
    integrationHelper : IntegrationHelper
    """
    lin_power = axionCAMB_wrapper.get_linear_power(extrap_kmin=1e-5, extrap_kmax=1e3)
    growth = axionCAMB_wrapper.get_growth()
    cosmo.set_H_interpolation(axionCAMB_wrapper.get_hubble())

    if integrationHelper is None:
        integrationHelper = IntegrationHelper(1024)

    if sigma_window is None:
        sigma_window=window

    if use_FFTLog:
        sigmaInt = SigmaInterpolatorFFTLog(cosmo, lin_power, growth, survey.center_z, kMin, kMax, Nr=1024, window_function=sigma_window)
        sigmaInt.compute(do_dr=True, do_dloga=full_bias)
    else:
        sigmaInt = SigmaInterpolator(cosmo, lin_power, growth, survey.mMin, survey.mMax, survey.center_z, integrationHelper, Nr=1024, window_function=sigma_window)
        sigmaInt.compute(kMin, kMax, do_dr=True, do_dloga=full_bias)

    if jenkins_mass:
        mass_function = JenkinsMassFunction(cosmo, sigmaInt)
    else:
        mass_function = PressSchechterMassFunction(cosmo, sigmaInt)

    if old_bias:
        halo_bias = HaloBias(cosmo, sigmaInt, mass_function, survey.mMin, survey.mMax, kMin, kMax, survey.center_z, integrationHelper, Nk=1024, window_function=window)
    elif full_bias:
        halo_bias = HaloBiasFull(cosmo, growth, sigmaInt, mass_function, survey.mMin, survey.mMax, kMin, kMax, survey.center_z, integrationHelper, Nk=1024, window_function=window)
    else:
        halo_bias = HaloBiasNew(cosmo, sigmaInt, mass_function, survey.mMin, survey.mMax, kMin, kMax, survey.center_z, integrationHelper, Nk=1024, window_function=window)

    if use_approximations and not old_bias:
        halo_bias.compute_approximation()
    else:
        halo_bias.compute()


    if use_FFTLog:
        corr = CorrelationFunctionsFFTLog(cosmo, lin_power, growth, halo_bias, kMin, halo_bias._kMax if window == "sharp_k" and old_bias and kMax > halo_bias._kMax else kMax, survey.center_z, rMin, integrationHelper)
    else:
        corr = CorrelationFunctions(cosmo, lin_power, growth, halo_bias, kMin, halo_bias._kMax if window == "sharp_k" and old_bias and kMax > halo_bias._kMax else kMax, survey.center_z, rMin, r_vals, integrationHelper)

    rMesh,zMesh = np.meshgrid(r_vals, survey.center_z)
    if do_unbiased:
        corr.compute(unbiased=True, old_bias=old_bias)
        xi_unbiased, xi, dbarxi_dloga_unbiased, dbarxi_dloga = corr.get_correlation_functions(rMesh, zMesh, unbiased=True)
        v = r_vals * 100 * cosmo.E(zMesh) * dbarxi_dloga / (3 * (1 + xi))
        v_dm = r_vals * 100 * cosmo.E(zMesh) * dbarxi_dloga_unbiased / (3 * (1 + xi_unbiased))
        if get_correlation_functions:
            return v, v_dm, xi_unbiased, xi, dbarxi_dloga_unbiased, dbarxi_dloga
        else:
            return v, v_dm
    else:
        corr.compute(unbiased=False, old_bias=old_bias)
        xi, dbarxi_dloga = corr.get_correlation_functions(rMesh, zMesh, unbiased=False)
        v = r_vals * 100 * cosmo.E(zMesh) * dbarxi_dloga / (3 * (1 + xi))
        if get_correlation_functions:
            return v, xi, dbarxi_dloga
        else:
            return v

