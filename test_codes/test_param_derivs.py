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
p_evaluate=ParallelizationQueue()

if rank==0:

    stencil = np.array([-2,-1,0,1,2])
    cosmo = Cosmology.generate(omega_axion=1e-6)
    step_size = cosmo.h*0.01
    param_vals = cosmo.h+stencil*step_size

    cosmoDB= CosmoDB()

    def pre_compute_function(cosmo, out_path, log_path):
        file_root = os.path.basename(out_path)
        root_path = out_path[:-len(file_root)]
        print(root_path, file_root, log_path)

        wrapper = AxionCAMBWrapper(root_path, file_root, log_path)
        success = wrapper(cosmo)

        return success

    def eval_function(cosmo, out_path, log_path):
        file_root = os.path.basename(out_path)
        root_path = out_path[:-len(file_root)]
        wrapper = AxionCAMBWrapper(root_path, file_root, log_path)

        lin_power = wrapper.get_linear_power()
        k_vals = np.logspace(-2,2, 100)
        return lin_power(k_vals)

    ds = ParamDerivatives(cosmoDB, cosmo, 'h', param_vals, eval_function, eval_function_args=(), eval_function_kwargs={}, pre_computation_function=pre_compute_function, has_to_precompute=True, pre_function_args=(), pre_function_kwargs={}, eval_queue=p_evaluate, pre_compute_queue=p_pre_compute, stencil=stencil)
    ds.prep1()
    p_pre_compute.run()
    ds.prep2()
    p_evaluate.run()
    np.savetxt("./test_derivs.dat", ds.derivs())
    
    cosmoDB.save()

