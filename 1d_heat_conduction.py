"""
This script implements a numerical solution for a one-dimensional heat conduction problem.
It uses a tridiagonal matrix algorithm to solve the implicit finite difference method for the heat equation.

The heat equation being solved is:

    ∂u/∂t = ∂²u/∂x²

where 'u' represents the temperature distribution, 't' is time, and 'x' is space.

The code includes functions for initializing grids, vectors, and coefficients, as well as solving the linear system.
It also provides a method for plotting a time or space-dependent coefficient 'a'.

Usage:
- Initialize grids and vectors using 'initialize_grids' and 'initialize_vectors'.
- Calculate coefficients with 'calculate_coefficients'.
- Solve the linear system with 'tridiagonal_matrix_algorithm_solver'.
- Additional functions are available for specific calculations and plotting.

Author: [Your Name]
Date: [Date]
"""

import numpy as np
import matplotlib.pyplot as plt

def find_nearest_index(array, value):
    """
    Finds the index of the element in 'array' closest to 'value'.
    
    Parameters:
        array (np.ndarray): The input array.
        value (float): The target value.
        
    Returns:
        int: Index of the closest element.
    """
    idx = (np.abs(array - value)).argmin()
    return idx

def initialize_grids(space_step, time_step, total_time):
    """
    Initializes grids for space and time.
    
    Parameters:
        space_step (float): The spacing between grid points in space.
        time_step (float): The time step.
        total_time (float): The total time.
        
    Returns:
        tuple: x_grid, t_grid, iterations
    """
    if space_step <= 0 or time_step <= 0 or total_time <= 0:
        raise ValueError("All inputs (space_step, time_step, total_time) must be positive values.")
    
    vec_len = int(round(2 / space_step - 1))
    iterations = int(round(total_time / time_step))
    t_grid = np.linspace(0, total_time, iterations + 1)
    x_grid = np.linspace(-1 + space_step, 1 - space_step, vec_len)
    return x_grid, t_grid, iterations

def initialize_vectors(vec_len):
    """
    Initializes vectors for calculations.
    
    Parameters:
        vec_len (int): Length of the vectors.
        
    Returns:
        tuple: a_vec, b_vec, c_vec, d_vec, cdp_vec, ddp_vec, solution
    """
    if vec_len <= 0:
        raise ValueError("vec_len must be a positive integer.")
    
    a_vec = np.zeros(vec_len)
    b_vec = np.zeros(vec_len)
    c_vec = np.zeros(vec_len)
    d_vec = np.zeros(vec_len)
    cdp_vec = np.zeros(vec_len)
    ddp_vec = np.zeros(vec_len)
    solution = np.zeros((iterations + 1, vec_len))
    return a_vec, b_vec, c_vec, d_vec, cdp_vec, ddp_vec, solution

def zero_solution(array, grid):
    """
    Initializes the solution at t=0.
    
    Parameters:
        array (np.ndarray): The solution array.
        grid (np.ndarray): The grid.
        
    Returns:
        np.ndarray: The updated solution array.
    """
    array[0][:] = grid[:]**4 - 2 * grid[:]**2 + 2
    return array[0]

def calculate_coefficients(space_step, time_step, diffusion_coefficient, a_vec, b_vec, c_vec):
    """
    Calculates coefficients used in the tridiagonal matrix algorithm.
    
    Parameters:
        space_step (float): The spacing between grid points in space.
        time_step (float): The time step.
        diffusion_coefficient (float): The diffusion coefficient.
        a_vec (np.ndarray): Vector 'a'.
        b_vec (np.ndarray): Vector 'b'.
        c_vec (np.ndarray): Vector 'c'.
        
    Returns:
        tuple: a_vec, b_vec, c_vec, beta_val
    """
    if space_step <= 0 or time_step <= 0 or diffusion_coefficient <= 0:
        raise ValueError("space_step, time_step, and diffusion_coefficient must be positive values.")
    
    beta_val = diffusion_coefficient * time_step / (space_step**2)
    a_vec[1:] = beta_val
    b_vec[:] = -(1 + 2 * beta_val)
    c_vec[:-1] = beta_val
    return a_vec, b_vec, c_vec, beta_val

def calculate_rhs(iteration, array, d_vec, beta_val):
    """
    Calculates the right-hand side vector for the tridiagonal matrix algorithm.
    
    Parameters:
        iteration (int): The current iteration.
        array (np.ndarray): The solution array.
        d_vec (np.ndarray): Vector 'd'.
        beta_val (float): The beta value.
        
    Returns:
        np.ndarray: The updated 'd' vector.
    """
    if iteration < 0 or iteration >= len(array) or len(array[0]) != len(d_vec):
        raise ValueError("Invalid inputs for calculate_rhs.")
    
    d_vec[:] = np.copy(-array[iteration][:])
    d_vec[0] -= beta_val
    d_vec[-1] -= beta_val
    return d_vec

def calculate_unknowns(array, space_step):
    """
    Calculates the order, constant, and exact solution.
    
    Parameters:
        array (np.ndarray): The solution array.
        space_step (float): The spacing between grid points in space.
        
    Returns:
        tuple: order, constant, exact
    """
    if len(array) < 3 or len(array[0]) != len(array[1]):
        raise ValueError("Invalid input array for calculate_unknowns.")
    
    order = np.log2((array[0] - array[1]) / (array[1] - array[2]))
    constant = (array[0] - array[1]) / ((space_step**order) * (1 - (1 / 2**order)))
    exact = array[0] - constant * space_step**order
    return order, constant, exact

def plot_dependable_a(space_step, time_step, dpdn, total_time):
    """
    Generates data for plotting with a dependable 'a'.
    
    Parameters:
        space_step (float): The spacing between grid points in space.
        time_step (float): The time step.
        dpdn (str): Either 't' or 'x' to indicate dependence on time or space.
        total_time (float): The total time.
        
    Returns:
        tuple: x_grid, t_grid, a_depend, n
    """
    if space_step <= 0 or time_step <= 0 or total_time <= 0:
        raise ValueError("All inputs (space_step, time_step, total_time) must be positive values.")
    
    arr_len = int(round(2 / space_step - 1))
    n = int(round(total_time / time_step))
    t_grid = np.linspace(0, total_time, n + 1)
    x_grid = np.linspace(-1 + space_step, 1 - space_step, arr_len)
    
    if dpdn == 't':
        a_depend = 0.001 * np.exp(4 * t_grid / 10)
    elif dpdn == 'x':
        a_depend = np.exp(10 * x_grid)

    return x_grid, t_grid, a_depend, n

def tridiagonal_matrix_algorithm_solver(a, b, c, d, e, f):
    """
    Solves a tridiagonal linear system.
    
    Parameters:
        a (np.ndarray): Vector 'a'.
        b (np.ndarray): Vector 'b'.
        c (np.ndarray): Vector 'c'.
        d (np.ndarray): Vector 'd'.
        e (np.ndarray): Vector 'e'.
        f (np.ndarray): Vector 'f'.
        
    Returns:
        tuple: x, e, f
    """
    n = len(a)
    
    x = np.zeros(n)

    e[0] = float(c[0] / b[0])
    f[0] = float(d[0] / b[0])

    for i in range(1, n):
        e[i] = float(c[i] / (b[i] - e[i - 1] * a[i]))
        f[i] = float((d[i] - a[i] * f[i - 1]) / (b[i] - a[i] * e[i - 1]))

    x[-1] = float(f[-1])

    for i in range(n - 2, -1, -1):
        x[i] = float(f[i] - e[i] * x[i + 1])

    x = ["%.9e" % member for member in x]
    return x, e, f

# Example of how to run the code
if __name__ == '__main__':
    # Test 1
    x_space, t_space, total_iterations = initialize_grids(0.1, 0.01, 1)
    a_vals, b_vals, c_vals, d_vals, cdp_vals, ddp_vals, solutions = initialize_vectors(len(x_space))
    zero_solution(solutions, x_space)
    a_vals, b_vals, c_vals, beta = calculate_coefficients(0.1, 0.01, 1, a_vals, b_vals, c_vals)
    for i in range(1, total_iterations + 1):
        d_vals = calculate_rhs(i - 1, solutions, d_vals, beta)
        solutions[i], cdp_vals, ddp_vals = tridiagonal_matrix_algorithm_solver(a_vals, b_vals, c_vals, d_vals, cdp_vals, ddp_vals)
    index1 = find_nearest_index(x_space, 0)
    value1 = solutions[-1][index1]
    print(f"Test 1 Result (dx=0.1): {value1}")

    # Test 2
    x_space, t_space, total_iterations = initialize_grids(0.00004, 0.01, 1)
    a_vals, b_vals, c_vals, d_vals, cdp_vals, ddp_vals, solutions = initialize_vectors(len(x_space))
    zero_solution(solutions, x_space)
    a_vals, b_vals, c_vals, beta = calculate_coefficients(0.00004, 0.01, 1, a_vals, b_vals, c_vals)
    for i in range(1, total_iterations + 1):
        d_vals = calculate_rhs(i - 1, solutions, d_vals, beta)
        solutions[i], cdp_vals, ddp_vals = tridiagonal_matrix_algorithm_solver(a_vals, b_vals, c_vals, d_vals, cdp_vals, ddp_vals)
    index2 = find_nearest_index(x_space, 0)
    value2 = solutions[-1][index2]
    print(f"Test 2 Result (dx=4e-5): {value2}")

    # Extra Test: Plot a_depend with dependence on space
    x_grid, t_grid, a_depend, n = plot_dependable_a(0.1, 0.01, 'x', 1)

    # Plot the coefficient a_depend
    plt.figure(figsize=(10, 6))
    plt.plot(x_grid, a_depend, label="a_depend (dependence on x)")
    plt.xlabel('x')
    plt.ylabel('a_depend')
    plt.title('Coefficient a_depend vs. x')
    plt.legend()
    plt.grid(True)
    plt.show()



