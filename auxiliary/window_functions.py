import numpy as np

class WindowFunctions(object):
    @staticmethod
    def top_hat(x):
        return 3*(np.sin(x)-x*np.cos(x))/x**3

    @staticmethod
    def dtop_hat(x):
        return (9*x*np.cos(x)+3*(x**2-3)*np.sin(x)) / x**4

    @staticmethod
    def sharp_k(x):
        return np.piecewise(x, [x<=1.0,x>1.0], [1.0, 0.0])

    @staticmethod
    def dsharp_k(x):
        return -np.piecewise(x, [x==1.0,x>1.0, x<1.0], [np.inf, 0.0, 0.0])

    @staticmethod
    def gaussian(x):
        return np.exp(-x**2 / 2)

    @staticmethod
    def dgaussian(x):
        return -x*np.exp(-x**2 / 2)

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

    @staticmethod
    def set_window_functions(object, window_function, cosmo):
        if window_function == 'top_hat':
            object.window = WindowFunctions.top_hat
            object.dwindow = WindowFunctions.dtop_hat
            object.radius_of_mass = lambda m: WindowFunctions.radius_of_mass_top_hat(m) / cosmo.rho_mean**(1 / 3)
            object.mass_of_radius = lambda r: WindowFunctions.mass_of_radius_top_hat(r) * cosmo.rho_mean

        elif window_function == 'gaussian':
            object.window = WindowFunctions.gaussian
            object.dwindow = WindowFunctions.dgaussian
            object.radius_of_mass = lambda m: WindowFunctions.radius_of_mass_gaussian(m) / cosmo.rho_mean**(1 / 3)
            object.mass_of_radius = lambda r: WindowFunctions.mass_of_radius_gaussian(r) * cosmo.rho_mean

        elif window_function == 'sharp_k':
            object.window = WindowFunctions.sharp_k
            object.dwindow = WindowFunctions.dsharp_k
            object.radius_of_mass = lambda m: WindowFunctions.radius_of_mass_sharp_k(m) / cosmo.rho_mean**(1 / 3)
            object.mass_of_radius = lambda r: WindowFunctions.mass_of_radius_sharp_k(r) * cosmo.rho_mean

        elif window_function == 'none':
            object.window = WindowFunctions.no_window
            object.dwindow = lambda x: 0
            object.radius_of_mass = lambda m: WindowFunctions.radius_of_mass_top_hat(m) / cosmo.rho_mean**(1 / 3)
            object.mass_of_radius = lambda r: WindowFunctions.mass_of_radius_top_hat(r) * cosmo.rho_mean

        else:
            raise Exception("Unknown window function: " + str(window_function))