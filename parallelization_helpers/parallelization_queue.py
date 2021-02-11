import numpy as np
from mpi4py import MPI

class ParallelizationQueue(object):

    def __init__(self):
        self.__comm = MPI.COMM_WORLD
        self.__size = self.__comm.Get_size()
        self.__rank = self.__comm.Get_rank()

        if self.__rank==0:
            self.__jobs = []
            self.__outputs = None
        else:
            self.run_child_node()


    def add_job(self, function, args, kwargs):
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
        for ids in jobs_chunked:
            tmp=[]
            for id in ids:
                tmp.append(self.__jobs[id])
            jobs_chunked.append(tmp)

        jobs = self.__comm.scatter(jobs_chunked, root=0)

        outputs = self.run_jobs(jobs)
        outputs = self.__comm.gather(outputs, root=0)

        self.__outputs = np.array(n_jobs, dtype=np.object)
        for i,ids in enumerate(jobs_chunked):
            for j,id in enumerate(ids):
                self.__outputs[ids] = outputs[i][j]

    @property
    def outputs(self):
        assert(self.__rank==0)
        return self.__outputs

    def run_child_node(self):
        assert(self.__rank!=0)

        jobs = self.__comm.scatter(None, root=0)
        outputs = self.run_jobs(jobs)
        outputs = self.__comm.gather(outputs, root=0)

        assert(outputs is None)

    @staticmethod
    def run_jobs(jobs):
        outputs = []
        for job in jobs:
            outputs.append(job[0](*job[1], **job[2]))
        return outputs
