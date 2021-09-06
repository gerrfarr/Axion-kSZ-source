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

from axion_kSZ_source.auxiliary.cosmo_db import CosmoDB
from axion_kSZ_source.axion_camb_wrappers.run_axion_camb import AxionCAMBWrapper

from axion_kSZ_source.theory.cosmology import Cosmology, CosmologyCustomH
from axion_kSZ_source.auxiliary.integration_helper import IntegrationHelper

from axion_kSZ_source.ostriker_vishniac.ov_new import OV_spectra
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

axion_masses = np.logspace(-27, -23, 41)#[5.0e-27, 5.0e-26, 1.0e-25, 1.0e-24]#[1.0e-27, 1.0e-26, 1.0e-25, 1.0e-24, 1.0e-23]#

axion_abundances = np.array([1.0e-04, 1.6e-04, 2.5e-04, 4.0e-04, 6.3e-04, 1.0e-03, 1.6e-03, 2.5e-03, 4.0e-03, 6.3e-03, 1.0e-02, 1.6e-02, 2.5e-02, 4.0e-02, 5.3e-02, 6.3e-02, 1.0e-01, 1.1e-01, 1.6e-01, 2.1e-01, 2.5e-01, 2.6e-01, 3.2e-01, 3.7e-01, 4.0e-01, 4.2e-01, 4.7e-01, 5.3e-01, 5.8e-01, 6.3e-01, 6.8e-01, 7.4e-01, 7.9e-01, 8.4e-01, 8.9e-01, 9.5e-01, 0.999])

p_pre_compute = ParallelizationQueue()
p_eval = ParallelizationQueue()

if rank==0:
    parser = argparse.ArgumentParser(description='Process some inputs.')
    parser.add_argument("-o", "--outpath", dest="outpath", help="path for outputs", default=None)

    args = parser.parse_args()

    out_path = args.outpath if args.outpath is not None else "/scratch/r/rbond/gfarren/axion_kSZ/"

    cosmoDB = CosmoDB()
    intHelper = IntegrationHelper(128)

    ell_vals = np.arange(2, 1e4 + 1, 1.0, dtype=int)
    kMin = 1e-4
    kMax = 1e2
    zmax = 12.0
    Nz = 20
    Nk = 100

    def schedule_camb_run(cosmo):

        new, ID, ran_TF, successful_TF, out_path, log_path = cosmoDB.add(cosmo)
        file_root = os.path.basename(out_path)
        root_path = out_path[:-len(file_root)]

        wrapper = AxionCAMBWrapper(root_path, file_root, log_path)
        if new:
            p_pre_compute.add_job(wrapper, cosmo)

        return wrapper

    def schedule_ov_compute(cosmo):
        ID, ran_TF, successful_TF, out_path, log_path = cosmoDB.get_by_cosmo(cosmo)
        if successful_TF:
            file_root = os.path.basename(out_path)
            root_path = out_path[:-len(file_root)]
            wrapper = AxionCAMBWrapper(root_path, file_root, log_path)

            id = p_eval.add_job(ov_eval_function, ID, ell_vals, cosmo, wrapper, intHelper, kMin=kMin, kMax=kMax, zmax=zmax, Nz=Nz, Nk=Nk)
        else:
            id = p_eval.add_job(lambda ells: np.full((len(ells)), np.nan), ell_vals)

        return id

    def ov_eval_function(id, ell_vals, cosmo, camb, intHelper, kMin=1e-4, kMax=1e2, zmax=12.0, Nz=20, Nk=100):
        k_vals = np.logspace(np.log10(kMin), np.log10(kMax), Nk)
        a_vals = np.linspace(1/(1+zmax), 1, Nz)
        z_vals = 1/a_vals - 1

        cosmo.set_H_interpolation(camb.get_hubble())
        ov = OV_spectra(cosmo, camb, intHelper, kMax=kMax, kMin=kMin)
        ov.compute_S(k_vals, z_vals)
        cells = ov.compute_OV(ell_vals, zmax=zmax)

        np.savetxt(out_path + "ov_out_data/ov_cells_ID={}.dat".format(id), np.vstack([ell_vals, cells]).T)

        return cells

if rank == 0:
    cosmologies = []

    for m in axion_masses:
        cosmologies.append([])
        for f_a in axion_abundances:
            cosmo = Cosmology.generate(axion_frac=f_a, m_axion=m, read_H_from_file=True)
            cosmologies[-1].append(cosmo)
            camb = schedule_camb_run(cosmo)

    p_pre_compute.run()

    for i in range(len(p_pre_compute.outputs)):
        cosmoDB.set_run_by_cosmo(*p_pre_compute.jobs[i][1], p_pre_compute.outputs[i])

    cosmoDB.save()

    output_ids = []
    for m_i, m in enumerate(axion_masses):
        output_ids.append([])
        for f_i, f_a in enumerate(axion_abundances):
            cosmo = cosmologies[m_i][f_i]

            output_ids[-1].append(schedule_ov_compute(cosmo))

    p_eval.run()

    outputs = np.empty((len(axion_masses), len(axion_abundances), len(ell_vals)))
    for m_i, m in enumerate(axion_masses):
        for f_i, f_a in enumerate(axion_abundances):
            outputs[m_i,f_i] = p_eval.outputs[output_ids[m_i][f_i]]

    np.save(out_path+"ov_cells", (ell_vals, outputs), allow_pickle=True)
