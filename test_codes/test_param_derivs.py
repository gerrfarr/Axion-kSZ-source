from mpi4py import MPI
import numpy as np
import os

try:
    from axion_kSZ_source.parallelization_helpers.parallelization_queue import ParallelizationQueue
except Exception:
    import sys

    sys.path.append("/global/homes/g/gfarren/axion kSZ/")
    from axion_kSZ_source.parallelization_helpers.parallelization_queue import ParallelizationQueue

from axion_kSZ_source.theory.cosmology import Cosmology
from axion_kSZ_source.fisher_analysis.get_parameter_derivative import ParamDerivatives
from axion_kSZ_source.auxiliary.cosmo_db import CosmoDB
from axion_kSZ_source.axion_camb_wrappers.run_axion_camb import AxionCAMBWrapper

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

print(f"This is rank {rank}")

p_pre_compute=ParallelizationQueue()
p_eval=ParallelizationQueue()

if rank==0:

    stencil = np.array([-2,-1,0,1,2])
    cosmo = Cosmology.generate(omega_axion=1e-6)
    step_size = cosmo.A_s*0.001
    param_vals = cosmo.A_s+stencil*step_size

    cosmoDB= CosmoDB()

    def schedule_camb_run(cosmo):

        new, ID, ran_TF, successful_TF, out_path, log_path = cosmoDB.add(cosmo)
        file_root = os.path.basename(out_path)
        root_path = out_path[:-len(file_root)]
        print(root_path, file_root, log_path)

        wrapper = AxionCAMBWrapper(root_path, file_root, log_path)
        if new:
            p_pre_compute.add_job(wrapper, cosmo)

    def eval_function(cosmo):
        ID, ran_TF, successful_TF, out_path, log_path = cosmoDB.get_by_cosmo(cosmo)
        file_root = os.path.basename(out_path)
        root_path = out_path[:-len(file_root)]
        wrapper = AxionCAMBWrapper(root_path, file_root, log_path)

        lin_power = wrapper.get_linear_power()
        k_vals = np.logspace(-3,2, 100)
        
        p_eval.add_job(lin_power, k_vals)


    ds = ParamDerivatives(cosmo, 'A_s', param_vals, eval_function, eval_function_args=(), eval_function_kwargs={}, pre_computation_function=schedule_camb_run, pre_function_args=(), pre_function_kwargs={}, stencil=stencil)
    ds.prep_parameters()
    p_pre_compute.run()
    for i in range(len(p_pre_compute.outputs)):
        cosmoDB.set_run_by_cosmo(*p_pre_compute.jobs[i][1], p_pre_compute.outputs[i])

    ds.prep_evaluation()
    p_eval.run()
    
    np.savetxt("./test_derivs.dat", ds.derivs(p_eval.outputs))
    
    cosmoDB.save()

