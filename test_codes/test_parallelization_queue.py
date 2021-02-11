from mpi4py import MPI
import numpy as np

try:
    from axion_kSZ_source.parallelization_helpers.parallelization_queue import ParallelizationQueue
except Exception:
    import sys
    sys.path.append("/global/homes/g/gfarren/axion kSZ/")
    from axion_kSZ_source.parallelization_helpers.parallelization_queue import ParallelizationQueue

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

print(f"This is rank {rank}")

p=ParallelizationQueue()

if rank==0:
    f = lambda x: x**2
    x_vals = np.linspace(0, 100, 101)
    for x in x_vals:
        p.add_job(f, (x,), None)

    p.run()
    
    print(p.outputs)