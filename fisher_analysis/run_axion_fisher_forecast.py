import mpi4py.rc
mpi4py.rc.threads = False
from mpi4py import MPI


import numpy as np
import os

import argparse

try:
    from axion_kSZ_source.parallelization_helpers.parallelization_queue import ParallelizationQueue
except Exception:
    import sys
    #### THIS line may need editing ##### Include path to source code package here
    sys.path.append("/home/r/rbond/gfarren/axion kSZ/")
    from axion_kSZ_source.parallelization_helpers.parallelization_queue import ParallelizationQueue

from axion_kSZ_source.theory.cosmology import Cosmology,CosmologyCustomH
from axion_kSZ_source.fisher_analysis.get_parameter_derivative import ParamDerivatives
from axion_kSZ_source.auxiliary.cosmo_db import CosmoDB
from axion_kSZ_source.axion_camb_wrappers.run_axion_camb import AxionCAMBWrapper
from axion_kSZ_source.fisher_analysis.compute_mpv import compute_mean_pairwise_velocity
from axion_kSZ_source.fisher_analysis.compute_covariance import compute_covariance_matrix
from axion_kSZ_source.auxiliary.survey_helper import StageII,StageIII,StageIV,SurveyType,StageSuper

from axion_kSZ_source.auxiliary.helper_functions import is_array
from axion_kSZ_source.auxiliary.integration_helper import IntegrationHelper
from axion_kSZ_source.fisher_analysis.compute_fisher_matrix import make_fisher_matrix,get_deriv_sets

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

axion_masses = np.logspace(-27, -23, 41)#[5.0e-27, 5.0e-26, 1.0e-25, 1.0e-24]#[1.0e-27, 1.0e-26, 1.0e-25, 1.0e-24, 1.0e-23]#

if rank==0:
    parser = argparse.ArgumentParser(description='Process some inputs.')
    parser.add_argument("-o", "--outpath", dest="outpath", help="path for outputs", default=None)
    parser.add_argument("-s", "--stage", dest="stage", help="survey stage", default="IV")
    parser.add_argument("--deltaTau_sq", dest="delta_tau_sq", help="(delta tau/tau)^2", default=0.15, type=float)
    parser.add_argument("-w", "--window", dest="window", help="window function", default="sharp_k")
    parser.add_argument("-a", "--approximation", action='store_true', dest='use_approximation', help="use approximations for bias computation", default=False)
    parser.add_argument("-f", "--FFTLog", action='store_true', dest='use_FFTLog', help="use FFTLog for integral evaluation", default=False)

    args = parser.parse_args()

    delta_r = 2.0
    rMin=1.0e-2
    r_vals = np.arange(20.0, 180.0, delta_r)
    using_btau_param = False
    if args.stage == "IV":
        survey=StageIV(Cosmology.generate(), delta_tau_sq=args.delta_tau_sq)
    elif args.stage == "III":
        survey = StageIII(Cosmology.generate(), delta_tau_sq=args.delta_tau_sq)
    elif args.stage == "II":
        survey = StageII(Cosmology.generate(), delta_tau_sq=args.delta_tau_sq)
    elif args.stage == "SuperIV":
        survey = StageSuper(Cosmology.generate(), delta_tau_sq=args.delta_tau_sq)
    else:
        raise Exception("No stage {} known!".format(args.stage))

    window=args.window
    old_bias=False
    full_bias=False
    kMin,kMax=1.0e-4,1.0e2
    out_path= args.outpath if args.outpath is not None else "/scratch/r/rbond/gfarren/axion_kSZ/fisher_outputs/sharpK_final_Stage{}_deltaTau={:.2f}/".format(args.stage, args.delta_tau_sq)

    use_approximations=args.use_approximation
    use_FFTLog=args.use_FFTLog
    prefix = "sharpK_5point"
    if use_FFTLog:
        prefix+="_FFTLog"
    if use_approximations:
        prefix+="_approx"

    axion_abundances = np.array([1.0e-04, 1.6e-04, 2.5e-04, 4.0e-04, 6.3e-04, 1.0e-03, 1.6e-03, 2.5e-03, 4.0e-03, 6.3e-03, 1.0e-02, 1.6e-02, 2.5e-02, 4.0e-02, 5.3e-02, 6.3e-02, 1.0e-01, 1.1e-01, 1.6e-01, 2.1e-01, 2.5e-01, 2.6e-01, 3.2e-01, 3.7e-01, 4.0e-01, 4.2e-01, 4.7e-01, 5.3e-01, 5.8e-01, 6.3e-01, 6.8e-01, 7.4e-01, 7.9e-01, 8.4e-01, 8.9e-01, 9.5e-01])

    axion_abundance_fractional_step_sizes = np.array([0.01, 0.05, 0.1, 0.2])

    cosmo_params = ["h", "omegaDM", "omegaB", "n_s", "A_s", "axion_frac"]#"log_axion_frac"
    parameter_fractional_step_sizes = {"h":0.05, "omegaDM":0.05, "omegaB":0.05, "n_s":0.005, "A_s":0.005, "axion_frac":axion_abundance_fractional_step_sizes}
    parameter_absolute_step_sizes = {"h": 0.0, "omegaDM": 0.0, "omegaB": 0.0, "n_s": 0.0, "A_s": 0.0, "axion_frac": np.zeros(axion_abundance_fractional_step_sizes.shape)}
    parameter_bounds = {"h": None, "omegaDM": None, "omegaB": None, "n_s": None, "A_s": None, "axion_frac":(1.0e-5,1.0)}#"log_axion_frac": (-11,0),
    nuisance_params_zIndep = ["bt"] if using_btau_param else []
    nuisance_params_zDep = ["b"]
    nuisance_params = nuisance_params_zIndep + nuisance_params_zDep

    number_of_parameter_step_sizes = 0
    for step_size in parameter_fractional_step_sizes.values():
        if is_array(step_size):
            number_of_parameter_step_sizes += len(step_size)
        else:
            number_of_parameter_step_sizes += 1

    stencil = np.array([-2,-1, 0, 1,2])

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
        if successful_TF:
            file_root = os.path.basename(out_path)
            root_path = out_path[:-len(file_root)]
            wrapper = AxionCAMBWrapper(root_path, file_root, log_path)

            id = p_eval.add_job(compute_mean_pairwise_velocity, r_vals, rMin, cosmo, wrapper, survey, window=window, old_bias=old_bias, full_bias=full_bias, jenkins_mass=False, integrationHelper=intHelper, kMin=kMin, kMax=kMax, do_unbiased=False, get_correlation_functions=False, use_approximations=use_approximations, use_FFTLog=use_FFTLog)
        else:
            id = p_eval.add_job(lambda survey, r_vals: np.full((len(survey.center_z), len(r_vals)), np.nan), survey, r_vals)

        return id

    def cov_eval_function(cosmo):
        ID, ran_TF, successful_TF, out_path, log_path = cosmoDB.get_by_cosmo(cosmo)
        file_root = os.path.basename(out_path)
        root_path = out_path[:-len(file_root)]
        wrapper = AxionCAMBWrapper(root_path, file_root, log_path)

        lin_power = wrapper.get_linear_power()
        growth = wrapper.get_growth()
        cosmo.set_H_interpolation(wrapper.get_hubble())

        id = p_eval.add_job(compute_covariance_matrix, r_vals, rMin, delta_r, cosmo, wrapper, survey, window=window, old_bias=old_bias, full_bias=full_bias, jenkins_mass=False, integrationHelper=intHelper, kMin=kMin, kMax=kMax, use_approximations=use_approximations, use_FFTLog=use_FFTLog)

        return id

    def b_deriv(cosmo, wrapper, r_vals, rMin, survey, window, old_bias, full_bias, intHelper, kMin, kMax, use_approximations, use_FFTLog):
        try:
            v, xi, dbarxi_dloga = compute_mean_pairwise_velocity(r_vals, rMin, cosmo, wrapper, survey, window=window, old_bias=old_bias, full_bias=full_bias, jenkins_mass=False, integrationHelper=intHelper, kMin=kMin, kMax=kMax, do_unbiased=False, get_correlation_functions=True, use_approximations=use_approximations, use_FFTLog=use_FFTLog)

            return v*(1-xi)/(1+xi)
        except:
            return np.full((len(survey.center_z), len(r_vals)), np.nan)


    def bt_deriv(cosmo, wrapper, r_vals, rMin, survey, window, old_bias, full_bias, intHelper, kMin, kMax, use_approximations, use_FFTLog):
        try:
            v, xi, dbarxi_dloga = compute_mean_pairwise_velocity(r_vals, rMin, cosmo, wrapper, survey, window=window, old_bias=old_bias, full_bias=full_bias, jenkins_mass=False, integrationHelper=intHelper, kMin=kMin, kMax=kMax, do_unbiased=False, get_correlation_functions=True, use_approximations=use_approximations, use_FFTLog=use_FFTLog)

            return v
        except:
            return np.full((len(survey.center_z), len(r_vals)), np.nan)


    parameters_analytic_deriv_functions = {"b": b_deriv, "bt":bt_deriv}

    covariance = None

for i_m, m in enumerate(axion_masses):

    p_pre_compute = ParallelizationQueue()
    p_eval = ParallelizationQueue()
    p_fisher = ParallelizationQueue()

    if rank==0:
        fiducial_cosmologies = []
        parameter_derivatives = []
        analytic_derivs_queue_ids = []
        if i_m==0:
            covariance_eval_id = None
            covariance_cosmo = Cosmology.generate(omega_axion=1.0e-6, m_axion=1.0e-24, read_H_from_file=True)
            cov_camb = schedule_camb_run(covariance_cosmo)

        for i_f, axion_frac in enumerate(axion_abundances):
            parameter_derivatives_tmp = {}
            analytic_derivs_queue_ids_tmp = {}

            fiducial_cosmo = Cosmology.generate(axion_frac=axion_frac, m_axion=m, read_H_from_file=True)
            fiducial_cosmologies.append(fiducial_cosmo)
            fid_axion_camb = schedule_camb_run(fiducial_cosmo)

            for param in cosmo_params+nuisance_params:
                if param in parameter_fractional_step_sizes.keys():
                    if is_array(parameter_fractional_step_sizes[param]):
                        parameter_derivatives_tmp[param] = []
                        for step_size_frac, step_size_abs in zip(parameter_fractional_step_sizes[param], parameter_absolute_step_sizes[param]):
                            param_vals = getattr(fiducial_cosmo, param) * (1.0 + stencil * step_size_frac) + stencil * step_size_abs
                            if parameter_bounds[param] is not None and (np.max(param_vals)>=parameter_bounds[param][1] or np.min(param_vals)<=parameter_bounds[param][0]):
                                continue

                            ds = ParamDerivatives(fiducial_cosmo, param, param_vals, mpv_eval_function, eval_function_args=(), eval_function_kwargs={}, pre_computation_function=schedule_camb_run, pre_function_args=(), pre_function_kwargs={}, stencil=stencil)
                            ds.prep_parameters()
                            parameter_derivatives_tmp[param].append(ds)
                    else:
                        param_vals = getattr(fiducial_cosmo, param)*(1.0 + stencil * parameter_fractional_step_sizes[param]) + stencil * parameter_absolute_step_sizes[param]
                        ds = ParamDerivatives(fiducial_cosmo, param, param_vals, mpv_eval_function, eval_function_args=(), eval_function_kwargs={}, pre_computation_function=schedule_camb_run, pre_function_args=(), pre_function_kwargs={}, stencil=stencil)
                        ds.prep_parameters()
                        parameter_derivatives_tmp[param] = ds
                else:
                    analytic_derivs_queue_ids_tmp[param] = p_eval.add_job(parameters_analytic_deriv_functions[param], fiducial_cosmo, fid_axion_camb, r_vals, rMin, survey, window, old_bias, full_bias, intHelper, kMin, kMax, use_approximations, use_FFTLog)

            parameter_derivatives.append(parameter_derivatives_tmp)
            analytic_derivs_queue_ids.append(analytic_derivs_queue_ids_tmp)

        p_pre_compute.run()
        for i in range(len(p_pre_compute.outputs)):
            cosmoDB.set_run_by_cosmo(*p_pre_compute.jobs[i][1], p_pre_compute.outputs[i])

        cosmoDB.save()

        if i_m==0:
            covariance_eval_id = cov_eval_function(covariance_cosmo)

        for i_f, axion_frac in enumerate(axion_abundances):

            for ds in parameter_derivatives[i_f].values():
                if is_array(ds):
                    for d in ds:
                        d.prep_evaluation()
                else:
                    ds.prep_evaluation()

        p_eval.run()
        if i_m == 0:
            covariance = p_eval.outputs[covariance_eval_id]

        derivatives = np.full((len(axion_abundances), number_of_parameter_step_sizes + len(nuisance_params), len(survey.center_z), len(r_vals)), np.nan)

        for i_f, axion_frac in enumerate(axion_abundances):
            i_param = 0
            fiducial_cosmo = fiducial_cosmologies[i_f]
            for param in cosmo_params:
                if is_array(parameter_fractional_step_sizes[param]):
                    for i_step_size, step_size in enumerate(parameter_fractional_step_sizes[param]):
                        param_vals = getattr(fiducial_cosmo, param) * (1.0 + stencil * step_size)
                        if parameter_bounds[param] is not None and (np.max(param_vals)>=parameter_bounds[param][1] or np.min(param_vals)<=parameter_bounds[param][0]):
                            i_param += 1
                            continue
                        ds = parameter_derivatives[i_f][param][i_step_size]
                        derivatives[i_f, i_param] = ds.derivs(p_eval.outputs[ds.outputs])
                        i_param += 1
                else:
                    ds = parameter_derivatives[i_f][param]
                    derivatives[i_f, i_param, :, :] = ds.derivs(p_eval.outputs[ds.outputs])
                    i_param += 1

            for param in nuisance_params:
                derivatives[i_f, i_param, :, :] = p_eval.outputs[analytic_derivs_queue_ids[i_f][param]]
                i_param += 1

        np.save(f"{out_path}{prefix}_test_derivs_ma={m:.3E}", derivatives)
        if i_m == 0:
            np.save(f"{out_path}{prefix}_test_covariances", np.array(p_eval.outputs[covariance_eval_id]))

        for i_f, axion_frac in enumerate(axion_abundances):
            deriv_sets = get_deriv_sets(derivatives[i_f], cosmo_params + nuisance_params, parameter_fractional_step_sizes)
            for d_set in deriv_sets:
                d_set= np.array(d_set)
                p_fisher.add_job(make_fisher_matrix, d_set[0:-len(nuisance_params_zDep)], d_set[-len(nuisance_params_zDep):], survey.center_z, covariance)

        p_fisher.run()

        np.save(f"{out_path}{prefix}_test_fisher_matrices_ma={m:.3E}", np.array(p_fisher.outputs))

if rank==0:
    cosmoDB.save()
