import numpy as np
from mpi4py import MPI
import axion_kSZ_source.parallelization_helpers.MPI_error_handler
import dill

MPI.pickle.__init__(dill.dumps,dill.loads)


class ParallelizationQueue(object):

    def __init__(self):
        self.__comm = MPI.COMM_WORLD
        self.__size = self.__comm.Get_size()
        self.__rank = self.__comm.Get_rank()

        if self.__rank==0:
            self.__jobs = []
            self.__outputs = None
        else:
            self.__run_child_node()

    def add_job(self, function, args=None, kwargs=None):
        assert(self.__rank == 0)

        self.__jobs.append((function, args, kwargs))
        return len(self.__jobs)-1

    def run(self):
        assert(self.__rank==0)

        n_jobs = len(self.__jobs)
        job_ids = np.arange(0, n_jobs, 1, dtype=np.int)
        np.random.shuffle(job_ids)

        job_ids_chunked = np.array_split(job_ids, self.__size)

        jobs_chunked = []
        for ids in job_ids_chunked:
            tmp=[]
            for id in ids:
                tmp.append(self.__jobs[id])
            jobs_chunked.append(tmp)

        jobs = self.__comm.scatter(jobs_chunked, root=0)

        outputs = self.__run_jobs(jobs)
        outputs = self.__comm.gather(outputs, root=0)

        self.__outputs = np.empty(n_jobs, dtype=np.object)
        for i,ids in enumerate(job_ids_chunked):
            for j,id in enumerate(ids):
                self.__outputs[id] = outputs[i][j]

        self.__jobs=[]

    @property
    def outputs(self):
        assert(self.__rank==0)
        return self.__outputs

    def __run_child_node(self):
        assert(self.__rank!=0)

        jobs = self.__comm.scatter(None, root=0)
        outputs = self.__run_jobs(jobs)
        outputs = self.__comm.gather(outputs, root=0)

        assert(outputs is None)

    @staticmethod
    def __run_jobs(jobs):
        outputs = []
        for job in jobs:
            if job[1] is not None and job[2] is not None:
                outputs.append(job[0](*job[1], **job[2]))
            elif job[1] is not None:
                outputs.append(job[0](*job[1]))
            elif job[2] is not None:
                outputs.append(job[0](**job[2]))
            else:
                outputs.append(job[0]())
        return outputs
