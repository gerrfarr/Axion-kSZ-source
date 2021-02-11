import numpy as np
import sympy as sp


class DifferentiationHelper:

    @staticmethod
    def get_finite_difference_coefficients(stencil, order_deriv):
        assert(len(stencil)>order_deriv)
        assert(np.all(np.diff(stencil)==1.0))

        powers=np.arange(0, len(stencil), 1)
        stencil_grid, powers_grid = np.meshgrid(stencil, powers)

        stencil_matrix = stencil_grid ** powers_grid

        sympy_vector=sp.Matrix([sp.factorial(order_deriv) if order_deriv==i else 0 for i in range(len(stencil))])
        sympy_matrix=sp.Matrix(stencil_matrix)
        try:
            inverse = sympy_matrix**-1
        except Exception as ex:
            raise ValueError("The given stencil resulted in an matrix that is not invertible.")

        return np.squeeze(np.array(inverse * sympy_vector, dtype=float))

class FiniteDifferencingRules:
    two_point_start_first_order = DifferentiationHelper.get_finite_difference_coefficients([0, 1], 1)
    two_point_end_first_order = DifferentiationHelper.get_finite_difference_coefficients([-1, 0], 1)

    three_point_start_first_order = DifferentiationHelper.get_finite_difference_coefficients([0, 1, 2], 1)
    three_point_end_first_order = DifferentiationHelper.get_finite_difference_coefficients([-2, -1, 0], 1)
    three_point_central_first_order = DifferentiationHelper.get_finite_difference_coefficients([-1,0,1], 1)

    five_point_start_first_order = DifferentiationHelper.get_finite_difference_coefficients([0, 1, 2, 3, 4], 1)
    five_point_end_first_order = DifferentiationHelper.get_finite_difference_coefficients([-4, -3, -2, -1, 0], 1)
    five_point_center_first_order = DifferentiationHelper.get_finite_difference_coefficients([-2, -1, 0, 1, 2], 1)
