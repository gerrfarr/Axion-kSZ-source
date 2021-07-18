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

    """
    	function to integrate a function with respect to two variables x and y using Gauss-Legendre Quadrature
    	"""

    def integrate2D(self, func, xmin, xmax, ymin=None, ymax=None):
        # if there are no separate y limits given assume the same limits as for x
        if ymin is None:
            ymin = xmin
        if ymax is None:
            ymax = xmax

        x = (xmax - xmin) / 2 * self.x_vals + (xmax + xmin) / 2
        y = (ymax - ymin) / 2 * self.x_vals + (ymax + ymin) / 2

        xgrid, ygrid = np.meshgrid(x, y)
        weightgrid_x, weightgrid_y = np.meshgrid(self.weights, self.weights)

        return (xmax - xmin) * (ymax - ymin) / 4 * sum(sum(weightgrid_x * weightgrid_y * func(xgrid, ygrid)))