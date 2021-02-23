from mpi4py import MPI
import numpy as np
import os

try:
    from axion_kSZ_source.parallelization_helpers.parallelization_queue import ParallelizationQueue
except Exception:
    import sys

    sys.path.append("/global/homes/g/gfarren/axion kSZ/")
    from axion_kSZ_source.parallelization_helpers.parallelization_queue import ParallelizationQueue

from axion_kSZ_source.theory.cosmology import Cosmology,CosmologyCustomH
from axion_kSZ_source.fisher_analysis.get_parameter_derivative import ParamDerivatives
from axion_kSZ_source.fisher_analysis.compute_fisher_matrix import make_fisher_matrix
from axion_kSZ_source.auxiliary.cosmo_db import CosmoDB
from axion_kSZ_source.axion_camb_wrappers.run_axion_camb import AxionCAMBWrapper
from axion_kSZ_source.fisher_analysis.compute_mpv import compute_mean_pairwise_velocity
from axion_kSZ_source.auxiliary.survey_helper import StageII,StageIII,StageIV,SurveyType

from axion_kSZ_source.auxiliary.helper_functions import is_array
from axion_kSZ_source.auxiliary.integration_helper import IntegrationHelper

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

axion_masses = [1.0e-26]#np.logspace(-27, -23, 41)

if rank==0:
    delta_r = 2.0
    rMin=1.0e-3
    r_vals = np.arange(20.0, 180.0, delta_r)
    survey=StageIV(Cosmology.generate())
    window="gaussian"
    old_bias=True
    kMin,kMax=1.0e-4,1.0e2

    axion_abundances = [1.0e-2]#np.array([1.0e-04, 1.6e-04, 2.5e-04, 4.0e-04, 6.3e-04, 1.0e-03, 1.6e-03, 2.5e-03, 4.0e-03, 6.3e-03, 1.0e-02, 1.6e-02, 2.5e-02, 4.0e-02, 5.3e-02, 6.3e-02, 1.0e-01, 1.1e-01, 1.6e-01, 2.1e-01, 2.5e-01, 2.6e-01, 3.2e-01, 3.7e-01, 4.0e-01, 4.2e-01, 4.7e-01, 5.3e-01, 5.8e-01, 6.3e-01, 6.8e-01, 7.4e-01, 7.9e-01, 8.4e-01, 8.9e-01, 9.5e-01])

    axion_abundance_fractional_step_sizes = np.array([0.01, 0.05, 0.1, 0.2])

    parameters_numeric = ["h", "omegaCDM", "omegaB", "n_s", "A_s", "axion_frac"]
    parameter_fractional_step_sizes = {"h":0.05, "omegaCDM":0.05, "omegaB":0.05, "n_s":0.005, "A_s":0.005, "axion_frac":axion_abundance_fractional_step_sizes}
    parameters_analytic = ["b"]

    number_of_parameter_step_sizes = 0
    for step_size in parameter_fractional_step_sizes:
        if is_array(step_size):
            number_of_parameter_step_sizes +=len(step_size)
        else:
            number_of_parameter_step_sizes += 1

    stencil = np.array([-2, -1, 0, 1, 2])

    cosmoDB = CosmoDB()
    intHelper = IntegrationHelper(1024)

    def schedule_camb_run(cosmo):

        new, ID, ran_TF, successful_TF, out_path, log_path = cosmoDB.add(cosmo)
        file_root = os.path.basename(out_path)
        root_path = out_path[:-len(file_root)]

        wrapper = AxionCAMBWrapper(root_path, file_root, log_path)
        if new:
            p_pre_compute.add_job(wrapper, cosmo)

        return wrapper

    def mpv_eval_function(cosmo):
        ID, ran_TF, successful_TF, out_path, log_path = cosmoDB.get_by_cosmo(cosmo)
        file_root = os.path.basename(out_path)
        root_path = out_path[:-len(file_root)]
        wrapper = AxionCAMBWrapper(root_path, file_root, log_path)

        lin_power = wrapper.get_linear_power()
        growth = wrapper.get_growth()
        cosmo.set_H_interpolation(wrapper.get_hubble())

        id = p_eval.add_job(compute_mean_pairwise_velocity, r_vals, rMin, cosmo, lin_power, growth, survey, window=window, old_bias=old_bias, jenkins_mass=False, integrationHelper=intHelper, kMin=kMin, kMax=kMax, do_unbiased=False, get_correlation_functions=False)

        return id

    def b_deriv(cosmo, wrapper, r_vals, rMin, survey, window, old_bias, intHelper, kMin, kMax):

        lin_power = wrapper.get_linear_power()
        growth = wrapper.get_growth()
        cosmo.set_H_interpolation(wrapper.get_hubble())

        v, xi, dbarxi_dloga = compute_mean_pairwise_velocity(r_vals, rMin, cosmo, lin_power, growth, survey, window=window, old_bias=old_bias, jenkins_mass=False, integrationHelper=intHelper, kMin=kMin, kMax=kMax, do_unbiased=False, get_correlation_functions=True)

        return v*(1-xi)/(1+xi)


    parameters_analytic_deriv_functions = {"b": b_deriv}

for i_m, m in enumerate(axion_masses):

    p_pre_compute = ParallelizationQueue()
    p_eval = ParallelizationQueue()
    p_fisher = ParallelizationQueue()

    if rank==0:
        parameter_derivatives = []
        analytic_derivs_queue_ids = []
        for i_f, axion_frac in enumerate(axion_abundances):
            parameter_derivatives_tmp = {}
            analytic_derivs_queue_ids_tmp = {}

            fiducial_cosmo = Cosmology.generate(axion_frac=axion_frac, m_axion=m, read_H_from_file=True)
            fid_axion_camb = schedule_camb_run(fiducial_cosmo)

            for param in parameters_numeric:
                if is_array(parameter_fractional_step_sizes[param]):
                    parameter_derivatives_tmp[param] = []
                    for step_size in parameter_fractional_step_sizes[param]:
                        param_vals = getattr(fiducial_cosmo, param) * (1.0 + stencil * step_size)
                        if param == "axion_frac" and np.max(param_vals)>=1.0 or np.min(param_vals)<=0.0:
                            continue
                        ds = ParamDerivatives(fiducial_cosmo, param, param_vals, mpv_eval_function, eval_function_args=(), eval_function_kwargs={}, pre_computation_function=schedule_camb_run, pre_function_args=(), pre_function_kwargs={}, stencil=stencil)
                        ds.prep_parameters()
                        parameter_derivatives_tmp[param].append(ds)
                else:
                    param_vals = getattr(fiducial_cosmo, param)*(1.0 + stencil * parameter_fractional_step_sizes[param])
                    ds = ParamDerivatives(fiducial_cosmo, param, param_vals, mpv_eval_function, eval_function_args=(), eval_function_kwargs={}, pre_computation_function=schedule_camb_run, pre_function_args=(), pre_function_kwargs={}, stencil=stencil)
                    ds.prep_parameters()
                    parameter_derivatives_tmp[param] = ds

            for param in parameters_analytic:
                analytic_derivs_queue_ids_tmp[param] = p_eval.add_job(parameters_analytic_deriv_functions[param], fiducial_cosmo, fid_axion_camb, r_vals, rMin, survey, window, old_bias, intHelper, kMin, kMax)

            parameter_derivatives.append(parameter_derivatives_tmp)
            analytic_derivs_queue_ids.append(analytic_derivs_queue_ids_tmp)

        p_pre_compute.run()
        for i in range(len(p_pre_compute.outputs)):
            cosmoDB.set_run_by_cosmo(*p_pre_compute.jobs[i][1], p_pre_compute.outputs[i])

        for i_f, axion_frac in enumerate(axion_abundances):
            for ds in parameter_derivatives[i_f].values():
                if is_array(ds):
                    for d in ds:
                        d.prep_evaluation()
                else:
                    ds.prep_evaluation()

        p_eval.run()

        derivatives = np.full((len(axion_abundances), number_of_parameter_step_sizes + len(parameters_analytic), len(survey.center_z), len(r_vals)), np.nan)

        for i_f, axion_frac in enumerate(axion_abundances):
            i_param = 0
            for param in parameters_numeric:
                if is_array(parameter_fractional_step_sizes[param]):
                    for i_step_size, step_size in enumerate(parameter_fractional_step_sizes[param]):
                        param_vals = getattr(fiducial_cosmo, param) * (1.0 + stencil * step_size)
                        if param == "axion_frac" and np.max(param_vals)>=1.0 or np.min(param_vals)<=0.0:
                            i_param += 1
                            continue
                        i_param += 1
                        ds = parameter_derivatives[i_f][param][i_step_size]
                        derivatives[i_f, i_param] = ds.derivs(p_eval.outputs[ds.outputs])
                else:
                    i_param += 1
                    ds = parameter_derivatives[i_f][param]
                    derivatives[i_f, i_param, :, :] = ds.derivs(p_eval.outputs[ds.outputs])

            for param in parameters_analytic:
                i_param += 1
                derivatives[i_f, i_param, :, :] = p_eval.outputs[analytic_derivs_queue_ids[i_f][param]]

        np.save(f"./test_derivs_ma={m:.3E}.dat", derivatives)

        """n_combination = 1
        for key in parameter_fractional_step_sizes.keys():
            if is_array(parameter_fractional_step_sizes[key]):
                n_combination*=len(parameter_fractional_step_sizes[key])

        for i_f, axion_frac in enumerate(axion_abundances):
            deriv_sets = np.empty((n_combination, len(parameters_numeric)+len(parameters_analytic), len(survey.center_z), len(r_vals)))

            for i_param, param in enumerate(parameters_numeric):
                if is_array(parameter_fractional_step_sizes[param]):
                    for i_step_size, step_size in enumerate(parameter_fractional_step_sizes[param]):
                        param_vals = getattr(fiducial_cosmo, param) * (1.0 + stencil * step_size)
                        if param == "axion_frac" and np.max(param_vals)>=1.0 or np.min(param_vals)<=0.0:
                            continue
                        #TODO

                else:
                    param_vals = getattr(fiducial_cosmo, param)*(1.0 + stencil * parameter_fractional_step_sizes[param])
                    #TODO


            for param in parameters_analytic:
                #TODO
                pass"""

if rank==0:
    cosmoDB.save()
