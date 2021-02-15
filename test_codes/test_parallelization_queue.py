from mpi4py import MPI
import numpy as np

try:
    from axion_kSZ_source.parallelization_helpers.parallelization_queue import ParallelizationQueue
except Exception:
    import sys
    sys.path.append("/global/homes/g/gfarren/axion kSZ/")
    from axion_kSZ_source.parallelization_helpers.parallelization_queue import ParallelizationQueue
    
from axion_kSZ_source.theory.cosmology import Cosmology

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

print(f"This is rank {rank}")


def test_one():
    p=ParallelizationQueue()

    if rank==0:
        cosmo=Cosmology()
        f = lambda x, c: x**2*c.h
        x_vals = np.linspace(0, 100, 101)
        for x in x_vals:
            p.add_job(f, (x,cosmo), None)

        p.run()

        assert(p.outputs == x_vasl**2*cosmo.h)