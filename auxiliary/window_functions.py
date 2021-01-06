import numpy as np

class WindowFunctions(object):
    @staticmethod
    def top_hat(x):
        return 3*(np.sin(x)-x*np.cos(x))/x**3

    @staticmethod
    def sharp_k(x):
        return np.piecewise(x, [x<=1.0,x>1.0], [1.0, 0.0])

    @staticmethod
    def gaussian(x):
        return np.exp(-x**2 / 2)

    @staticmethod
    def no_window(x):
        return 1

    @staticmethod
    def radius_of_mass_top_hat(M):
        return (3 * M / (4 * np.pi))**(1 / 3)

    @staticmethod
    def radius_of_mass_gaussian(M):
        return (2 * np.pi)**(-1 / 2) * M**(1 / 3)

    @staticmethod
    def radius_of_mass_sharp_k(M):
        return (9 * np.pi / 2)**(-1 / 3) * WindowFunctions.radius_of_mass_top_hat(M)

    @staticmethod
    def mass_of_radius_top_hat(R):
        return 4 / 3 * np.pi * R**3

    @staticmethod
    def mass_of_radius_gaussian(R):
        return np.sqrt(2 * np.pi) * (4 * np.pi / 3)**(-1 / 3) * WindowFunctions.mass_of_radius_top_hat(R)

    @staticmethod
    def mass_of_radius_sharp_k(R):
        return (9 * np.pi / 2) * WindowFunctions.mass_of_radius_top_hat(R)