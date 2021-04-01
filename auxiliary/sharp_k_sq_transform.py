from mcfit import mcfit,kernels
import numpy as np

class SharpKVar(mcfit):
    """Variance in a Gaussian window.
    Parameters
    ----------
    k : see `x` in :class:`mcfit.mcfit`
    See :class:`mcfit.mcfit`
    """
    def __init__(self, k, deriv=0, q=1.5, **kwargs):
        MK = Mellin_SharpK(deriv)
        mcfit.__init__(self, k, MK, q, **kwargs)
        self.prefac *= self.x**3 / (2 * np.pi**2)

def Mellin_SharpK(deriv=0):
    def MK(z):
        return 1.0/z
    return kernels._deriv(MK, deriv)