from math import pi, log10
from finite_differences import *
from error_calculation import absolute_error, relative_error
from grid_generation import *

def f(x):
    return cos(4 * x**2)

def f_prime_exact(x):
    return -128 * x**3 * sin(4 * x**2)

def find_nearest_value(array, value):
    if not array:
        raise ValueError("Array is empty in find_nearest_value function")
    return min(array, key=lambda x: abs(x - value))


def main(diff_func, grid_func, err_func, nodes=11):
    """
    Main function to perform finite difference calculations and error analysis.

    Args:
        diff_func (function): Finite difference function to use.
        grid_func (function): Grid generation function to use.
        err_func (function): Error calculation function to use.
        nodes (int, optional): Number of nodes in the grid. Default is 11.

    Returns:
        tuple: Two lists containing log-scaled dx values and log-scaled errors.

    Example:
        To use central difference with a non-uniform grid and relative error:
            dx_space, diff = main(central_diff, gen_non_uniform_x, rel_err, nodes=15)
    """
    x_i = pi/2.
    dx_space = []
    diff = []

    for i in range(-52, 8):
        dx = 2**i
        dx_space.append(log10(dx))

        x_values = grid_func(x_i, dx, nodes)
        idx = min(x_values, key=lambda x: abs(x - x_i))

        # Calculate finite difference
        E = diff_func(f, idx, x_values, dx) - f_prime_exact(x_i)

        # Calculate error
        error = err_func(E, f_prime_exact(x_i))

        diff.append(log10(error))

    return dx_space, diff

if __name__ == "__main__":
    # Choose functions
    diff_function = central_diff
    grid_function = gen_x
    error_function = abs_err

    # Number of nodes (can be changed)
    num_nodes = 11

    dx_space, diff = main(diff_function, grid_function, error_function, nodes=num_nodes)

# Help Section
"""
Finite Difference and Error Analysis:

This script performs finite difference calculations and error analysis for a given function f(x) and its exact derivative.
You can change the value of x_i to analyze the function at different points. You can also change the function f(x) provided you
also know it's derivative 

Usage:
1. Choose the differentiation function, grid generation function, and error calculation function.
2. Optionally, set the number of nodes in the grid (default is 11).

Differentiation Functions:
- forward_diff(func, nodes, dx)
- backward_diff(func, nodes, dx)
- central_diff(func, nodes, dx)
- second_central_diff(func, nodes, dx)
- sixth_order_scheme(func, nodes, dx)
- forward_second_derivative(func, nodes, dx)
- backward_second_derivative(func, nodes, dx)
- mixed_derivative(func, nodes, dx_x, dx_y)

Grid Generation Functions:
- generate_non_uniform_x_values(x_i, dx_values, nodes)
- generate_x_values(x_i, dx, nodes)

Error Calculation Functions:
- absolute_error(approximate, exact)
- relative_error(approximate, exact)

Main Function:
- main(diff_func, grid_func, err_func, nodes=11)
  - diff_func: Chosen differentiation function
  - grid_func: Chosen grid generation function
  - err_func: Chosen error calculation function
  - nodes: Number of nodes in the grid (optional, default is 11)

Example Usage:
- To use central difference with a non-uniform grid and relative error:
  dx_space, diff = main(central_diff, generate_non_uniform_x_values, relative_error, nodes=15)
"""