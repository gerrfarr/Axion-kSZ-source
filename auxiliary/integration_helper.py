import numpy as np
from .helper_functions import is_array

class IntegrationHelper(object):
    def __init__(self, N=100, method='gauss'):
        if method=='gauss':
            self.x_vals, self.weights = np.polynomial.legendre.leggauss(N)

    def integrate(self, func, xmin, xmax):
        if is_array(xmin) or is_array(xmax):
            eval_x = np.outer((xmax-xmin)/2,self.x_vals) + np.outer((xmax+xmin)/2, np.ones(np.shape(self.x_vals)))

            return np.squeeze(np.sum(func(eval_x) * self.weights, axis=-1) * (xmax - xmin) / 2)
        else:
            eval_x = (xmax-xmin)/2*self.x_vals + (xmax+xmin)/2
            return np.sum(func(eval_x)*self.weights)*(xmax-xmin)/2

    def get_points_weights(self, xmin=-1, xmax=1):
        return (xmax-xmin)/2*self.x_vals + (xmax+xmin)/2, self.weights*(xmax-xmin)/2