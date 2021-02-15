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

        assert(np.all(p.outputs == x_vals**2*cosmo.h))

def test_two():
    p1=ParallelizationQueue()
    p2=ParallelizationQueue()

    if rank==0:
        cosmo = Cosmology()
        f = lambda x, c: x**2 * c.h
        x_vals = np.linspace(0, 100, 101)
        for x in x_vals:
            p1.add_job(f, (x, cosmo), None)
            p2.add_job(f, (x**2, cosmo), None)

        p1.run()
        p2.run()

        print(p1.outputs, p2.outputs)

        assert (np.all(p1.outputs == x_vals**2 * cosmo.h))
        assert (np.all(p2.outputs == x_vals**4 * cosmo.h))

test_one()
test_two()