"""
code to take derivatives with respect to parameter values

Jul 2020
Gerrit Farren
"""
import time
import numpy as np
from ..auxiliary.differentiation_helpers import DifferentiationHelper
from ..parallelization_helpers.parallelization_queue import ParallelizationQueue
from ..auxiliary.cosmo_db import CosmoDB
import copy

class ParamDerivatives(object):
    """
    Class to compute derivatives with respect to a given parameter

    """
    def __init__(self, cosmoDB, fiducial_params, param_name, param_vals, eval_function, eval_function_args=(), eval_function_kwargs={}, pre_computation_function=None, has_to_precompute=True, pre_function_args=(), pre_function_kwargs={}, eval_queue=None, pre_compute_queue=None, stencil=None):
        """

        Parameters
        ----------
        cosmoDB : CosmoDB
            database to hold all CAMB runs
        pre_compute_queue : ParallelizationQueue
           ParallizationQueue to execute all pre-computation actions
        eval_queue : ParallelizationQueue
            ParallizationQueue to execute all evaluation actions
        fiducial_params : object
            class containing fiducial parameters
        param_name : str
            the name of the parameter being varried
        param_vals : array_like
            parameter values at which to evaluate function
        eval_function : (object, *args, **kwargs) -> array_like
            the function of which to take the derivative with respect to parameter `param_name`
        eval_function_args : tuple
            arguments passed to `eval_function`
        eval_function_kwargs : dict
            keyword arguments passed to `eval_function`
        pre_function_args : tuple
            arguments passed to `pre_computation_function`
        pre_function_kwargs : dict
            keyword arguments passed to `pre_computation_function`
        pre_computation_function : (object, string, string , *args, **kwargs) -> None
            function to execute for every parameter set before taking derivatives
        has_to_precompute: bool
            whether to execute pre_computation_function newly for every new parameter set
        """

        self.__fiducial = fiducial_params
        self.__param_name = param_name
        self.__param_vals = param_vals
        self.__eval_function = eval_function
        self.__eval_queue = eval_queue
        self.__pre_computation_function = pre_computation_function
        self.__has_to_precompute = has_to_precompute
        self.__pre_compute_queue = pre_compute_queue
        if has_to_precompute:
            assert(self.__pre_computation_function is not None)

        self.__cosmoDB = cosmoDB

        self.__eval_args = eval_function_args
        self.__eval_kwargs = eval_function_kwargs

        self.__pre_args = pre_function_args
        self.__pre_kwargs = pre_function_kwargs

        self.__fiducial_value = getattr(self.__fiducial, self.__param_name)
        self.__step_size = self.__param_vals[1] - self.__param_vals[0]
        if stencil is None:
            stencil = np.array((self.__param_vals - self.__fiducial_value) / self.__step_size, dtype=np.int)

        self.__coefficients = DifferentiationHelper.get_finite_difference_coefficients(stencil, 1)

        self.__parameter_sets = None
        self.__evals = None
        self.__queue_ids_precompute = None
        self.__queue_ids_evals = None
        self.__derivs = None

    def prep1(self):
        self.__parameter_sets = self.__get_parameter_sets()

    def prep2(self):
        if self.__pre_compute_queue is not None:
            for db_id,queue_id in self.__queue_ids_precompute:
                self.__cosmoDB.set_run(db_id, self.__pre_compute_queue.outputs[queue_id])

        if self.__eval_queue is None:
            self.__evals = self.__evaluate()
        else:
            self.__queue_ids_evals = self.__evaluate()

    def derivs(self):
        if self.__evals is None and self.__queue_ids_evals is not None:
            self.__evals = self.__eval_queue.outputs[self.__queue_ids_evals]
        elif self.__evals is None and self.__queue_ids_evals is None:
            raise Exception("Something went wrong with the evaluation!")

        self.__derivs = self.__get_derivatives()
        return self.__derivs


    def __get_parameter_sets(self):
        """
        Function to generate alternate parameter sets
        Returns
        -------
        list[object]
            List with generated parameter sets
        """
        params = []
        queue_ids = []
        for i in range(len(self.__param_vals)):
            if self.__coefficients[i]!=0:
                val = self.__param_vals[i]
                if val == self.__fiducial_value:
                    params.append(self.__fiducial)
                    id, ran_TF, success_TF, out_path, log_path = self.__cosmoDB.add(self.__fiducial)
                    if self.__has_to_precompute and not ran_TF:
                        if self.__pre_compute_queue is None:
                            success = self.__pre_computation_function(self.__fiducial, out_path, log_path, *self.__pre_args, **self.__pre_kwargs)
                            self.__cosmoDB.set_run(id, success)
                        else:
                            queue_ids.append((id,self.__pre_compute_queue.add_job(self.__pre_computation_function, (self.__fiducial, out_path, log_path, *self.__pre_args), self.__pre_kwargs)))

                else:
                    new_cosmo = copy.copy(self.__fiducial)
                    setattr(new_cosmo, self.__param_name, val)
                    id, ran_TF, success_TF, out_path, log_path = self.__cosmoDB.add(new_cosmo)
                    if self.__has_to_precompute and not ran_TF:
                        if self.__pre_compute_queue is None:
                            success = self.__pre_computation_function(new_cosmo, out_path, log_path, *self.__pre_args, **self.__pre_kwargs)
                            self.__cosmoDB.set_run(id, success)
                        else:
                            queue_ids.append((id,self.__pre_compute_queue.add_job(self.__pre_computation_function, (new_cosmo, out_path, log_path, *self.__pre_args), self.__pre_kwargs)))

                    params.append(new_cosmo)
            else:
                params.append(None)

        self.__queue_ids_precompute = queue_ids
        return params

    def __evaluate(self):
        """
        Function that evaluates the input function at the different parameter values.
        Returns
        -------
        list[array_like]
            List with outputs from the input function
        """
        values = []
        queue_ids = []
        for i in range(len(self.__parameter_sets)):
            if self.__coefficients[i]!=0:
                cosmo = self.__parameter_sets[i]
                id, ran_TF, success_TF, out_path, log_path = self.__cosmoDB.get_by_cosmo(cosmo)
                if self.__eval_queue is None:
                    vals = self.__eval_function(cosmo, out_path, log_path, *self.__eval_args, **self.__eval_kwargs)
                    values.append(vals)
                else:
                    queue_ids.append(self.__eval_queue.add_job(self.__eval_function, (cosmo, out_path, log_path, *self.__eval_args,), self.__eval_kwargs))

        if self.__eval_queue is None:
            return values
        else:
            return queue_ids

    def __get_derivatives(self):
        """
        Function that obtains derivatives from evaluation results
        Returns
        -------
        array_like
            ndarray of derivatives
        """

        return np.dot(np.array(self.__evals).T,self.__coefficients[self.__coefficients.nonzero()])/self.__step_size

    @property
    def outputs(self):
        return self.__evals

    @property
    def cosmologies(self):
        return self.__parameter_sets

