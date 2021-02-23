from ..theory.cosmology import Cosmology
from ..theory.sigma_interpolation import SigmaInterpolator
from ..theory.halo_bias import HaloBias
from ..theory.halo_bias_new import HaloBias as HaloBiasNew
from ..theory.mass_functions import JenkinsMassFunction, PressSchechterMassFunction
from ..theory.correlation_functions import CorrelationFunctions

from ..covariance.covariance_matrix import Covariance

from ..axion_camb_wrappers.growth_interpolation import GrowthInterpolation
from ..axion_camb_wrappers.power_interpolation import LinearPowerInterpolation

from ..auxiliary.integration_helper import IntegrationHelper
from ..auxiliary.survey_helper import SurveyType

def compute_covariance_matrix(r_vals, rMin, deltaR, cosmo, lin_power, growth, survey, window="gaussian", old_bias=False, jenkins_mass=False, integrationHelper=None, kMin=1.0e-4, kMax=1.0e2):

    if integrationHelper is None:
        integrationHelper = IntegrationHelper(1024)

    sigmaInt = SigmaInterpolator(cosmo, lin_power, growth, survey.mMin, survey.mMax, survey.center_z, integrationHelper, Nr=1024, window_function=window)
    sigmaInt.compute(kMin, kMax, do_dr=True, do_dloga=False)

    if jenkins_mass:
        mass_function = JenkinsMassFunction(cosmo, sigmaInt)
    else:
        mass_function = PressSchechterMassFunction(cosmo, sigmaInt)

    if old_bias:
        halo_bias = HaloBias(cosmo, sigmaInt, mass_function, survey.mMin, survey.mMax, kMin, kMax, survey.center_z, integrationHelper, Nk=1024, window_function=window)
    else:
        halo_bias = HaloBiasNew(cosmo, sigmaInt, mass_function, survey.mMin, survey.mMax, kMin, kMax, survey.center_z, integrationHelper, Nk=1024, window_function=window)

    halo_bias.compute()

    corr = CorrelationFunctions(cosmo, lin_power, growth, halo_bias, kMin, halo_bias._kMax if window == "sharp_k" and old_bias and kMax > halo_bias._kMax else kMax, survey.center_z, rMin, r_vals, integrationHelper)

    cov = Covariance(cosmo, lin_power, growth, mass_function, halo_bias, corr, survey.zMin, survey.zMax, survey.Nz, r_vals, deltaR, survey.f_sky, survey.sigma_v_vals, integrationHelper, kmin=kMin, kmax=kMax, mMin=survey.mMin, mMax=survey.mMax, old_bias=old_bias)
    cov.compute()

    return cov.full_covariance()