"""
This module provides implementations of the Jacobi and Gauss-Seidel methods for solving systems of linear equations.

Functions:
- solve_jacobi_method(A, b, max_iterations, initial_guess): Solves a system of linear equations using the Jacobi method.
- solve_gauss_siedel_method(A, b, max_iterations, initial_guess): Solves a system of linear equations using the Gauss-Seidel method.

Usage:
- Define the coefficient matrix 'A', right-hand side vector 'b', initial guess, and maximum number of iterations.
- Call the respective function to get the final solution and an array of solutions for each iteration.
- Results are displayed using pandas DataFrames.

"""


import numpy as np
import pandas as pd

def solve_jacobi_method(A, b, max_iterations, initial_guess):
    """
    Solves a system of linear equations using the Jacobi method.
    
    Parameters:
        A (np.ndarray): Coefficient matrix.
        b (np.ndarray): Right-hand side vector.
        max_iterations (int): Maximum number of iterations.
        initial_guess (np.ndarray): Initial guess for the solution.
        
    Returns:
        np.ndarray: Final solution, array of all solutions for each iteration.
    """
    num_variables = len(initial_guess)
    num_equations = len(A)
    solutions = np.zeros((max_iterations, num_variables))
    x = initial_guess.copy()
    
    for k in range(max_iterations):
        D = np.diag(A)
        x_new = np.zeros(num_variables)
        
        for i in range(num_variables):
            sum_points = np.sum(A[i, j] * x[j] for j in range(num_equations) if i != j)
            x_new[i] = (1 / D[i]) * (b[i] - sum_points)
            
        solutions[k, :] = x_new
        x = x_new.copy()
    
    return x, solutions

def solve_gauss_siedel_method(A, b, max_iterations, initial_guess):
    """
    Solves a system of linear equations using the Gauss-Seidel method.
    
    Parameters:
        A (np.ndarray): Coefficient matrix.
        b (np.ndarray): Right-hand side vector.
        max_iterations (int): Maximum number of iterations.
        initial_guess (np.ndarray): Initial guess for the solution.
        
    Returns:
        np.ndarray: Final solution, array of all solutions for each iteration.
    """
    num_variables = len(initial_guess)
    num_equations = len(A)
    solutions = np.zeros((max_iterations, num_variables))
    x = initial_guess.copy()
    
    for k in range(max_iterations):
        D = np.diag(A)
        x_new = np.zeros(num_variables)
        
        for i in range(num_variables):
            sum_points_less = np.sum(A[i, j] * x_new[j] for j in range(num_equations) if j < i)
            sum_points_more = np.sum(A[i, j] * x[j] for j in range(num_equations) if j > i)
            x_new[i] = (1 / D[i]) * (b[i] - sum_points_less - sum_points_more)
            
        solutions[k, :] = x_new
        x = x_new.copy()
    
    return x, solutions

# Example Usage
if __name__ == '__main__':
    A = np.array([[12.0, 7.0, 3.0], [1.0, 5.0, 1.0], [2.0, 7.0, -11.0]])
    b = np.array([22.0, 7.0, -2.0])
    initial_guess = np.array([1., 2., 1.])
    max_iterations = 10

    final_gauss_siedel_solution, gauss_siedel_solutions = solve_gauss_siedel_method(A, b, max_iterations, initial_guess)
    final_jacobi_solution, jacobi_solutions = solve_jacobi_method(A, b, max_iterations, initial_guess)

    index = [i for i in range(1, max_iterations + 1)]
    column_headers = ['x1', 'x2', 'x3']
    df1 = pd.DataFrame(gauss_siedel_solutions, columns=column_headers, index=index)
    df1.index.name = 'Iteration #'
    df2 = pd.DataFrame(jacobi_solutions, columns=column_headers, index=index)
    df2.index.name = 'Iteration #'

    print('Solution of Gauss-Seidel method:')
    print(df1)
    print('\nSolution of Jacobi method:')
    print(df2)
