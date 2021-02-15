"""
code to take derivatives with respect to parameter values

Jul 2020
Gerrit Farren
"""
import time
import numpy as np
from ..auxiliary.differentiation_helpers import DifferentiationHelper
import copy

class ParamDerivatives(object):
    """
    Class to compute derivatives with respect to a given parameter

    """
    def __init__(self, fiducial_params, param_name, param_vals, eval_function, *function_args, pre_computation_function=None, stencil=None, has_to_precompute=True, verbose=False, **function_kwargs):
        """

        Parameters
        ----------
        fiducial_params : object
            class containing fiducial parameters
        param_name : str
            the name of the parameter being varried
        param_vals : array_like
            parameter values at which to evaluate function
        eval_function : (object, *args, **kwargs) -> array_like
            the function of which to take the derivative with respect to parameter `param_name`
        function_args : dict
            arguments passed to `eval_function`
        function_kwargs : dict
            keyword arguments passed to `eval_function`
        pre_computation_function: (object)
            function to execute for every parameter set before taking derivatives
        has_to_precompute: bool
            whether to execute pre_computation_function newly for every new parameter set
        """

        self.__fiducial = fiducial_params
        self.__param_name = param_name
        self.__param_vals = param_vals
        self.__eval_function = eval_function
        self.__pre_computation_function = pre_computation_function
        self.__has_to_precompute = has_to_precompute
        self.__verbose = verbose
        self.__args = function_args
        self.__kwargs = function_kwargs

        self.__fiducial_value = getattr(self.__fiducial, self.__param_name)
        self.__step_size = self.__param_vals[1] - self.__param_vals[0]
        if stencil is None:
            stencil = np.array((self.__param_vals - self.__fiducial_value) / self.__step_size, dtype=np.int)

        self.__coefficients = DifferentiationHelper.get_finite_difference_coefficients(stencil, 1)

        self.__parameter_sets = None
        self.__evals = None
        self.__derivs = None

    def __call__(self):
        """
        Computes and provides the derivatives of the input function with respect to the given parameter

        Returns
        -------
        array_like
            Derivatives
        """
        if self.__derivs is None:
            start_copy = time.time()
            self.__parameter_sets = self.__get_parameter_sets()
            if self.__verbose:
                print("It took {:.3f}s to create cosmo copies".format(time.time()-start_copy))
            start_evaluate = time.time()
            self.__evals = self.__evaluate()
            if self.__verbose:
                print("It took {:.3f}s to evaluate the input function".format(time.time() - start_evaluate))
            start_derivs=time.time()
            self.__derivs = self.__get_derivatives()
            if self.__verbose:
                print("It took {:.3f}s to generate derivatives".format(time.time() - start_derivs))

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
        for i in range(len(self.__param_vals)):
            if self.__coefficients[i]!=0:
                val = self.__param_vals[i]
                if val == self.__fiducial_value:
                    params.append(self.__fiducial)
                    if self.__has_to_precompute:
                        self.__pre_computation_function(self.__fiducial)
                else:
                    new_cosmo = copy.copy(self.__fiducial)
                    setattr(new_cosmo, self.__param_name, val)
                    if self.__has_to_precompute:
                        self.__pre_computation_function(self.__fiducial)
                    params.append(new_cosmo)
            else:
                params.append(None)

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
        for i in range(len(self.__parameter_sets)):
            if self.__coefficients[i]!=0:
                cosmo = self.__parameter_sets[i]
                vals = self.__eval_function(cosmo, *self.__args, **self.__kwargs)
                values.append(vals)

        return values

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

