import numpy as np
from math import pi, cos, sin, log10, log

def initialize_parameters(dx, dt, c):
    """
    Initialize parameters for the solver.

    Args:
        dx (float): Spatial discretization step.
        dt (float): Temporal discretization step.
        c (float): Wave velocity.

    Returns:
        tuple: Constants C1, spatial grid x, time grid t, number of spatial nodes Nx,
               number of temporal nodes Nt, and index for x where x[index] = 0.25.
    """
    C1 = (-dt*c)/(2*dx)
    num_x_nodes = int(round(1/dx))
    x = np.linspace(0, (num_x_nodes+1)*dx, num_x_nodes+2)
    num_x_nodes = len(x)
    num_t_nodes = int(round(4/dt))
    t = np.linspace(0, num_t_nodes*dt, num_t_nodes+1)
    num_t_nodes = len(t)
    x_index = find_nearest(x, 0.25)
    return C1, x, t, num_x_nodes, num_t_nodes, x_index

def find_nearest(array, value):
    """
    Find the index in the array closest to the given value.

    Args:
        array (numpy.ndarray): Input array.
        value (float): Value to find.

    Returns:
        int: Index of the closest value in the array.
    """
    idx = (np.abs(array-value)).argmin()
    return idx

def finite_difference_2nd_order(arr, arr1, i, j, a):
    """
    Compute the finite difference scheme for 2nd order derivative.

    Args:
        arr (numpy.ndarray): Input array.
        arr1 (numpy.ndarray): Another input array.
        i (int): Index i.
        j (int): Index j.
        a (int): Parameter a.

    Returns:
        float: Result of finite difference scheme.
    """
    return ((arr[i-1,j+1] + a*arr1[i,j+1]) - (arr[i-1,j-1] + a*arr1[i,j-1]))

def apply_periodic_boundary_conditions(arr, n):
    """
    Apply periodic boundary conditions.

    Args:
        arr (numpy.ndarray): Input array.
        n (int): Parameter n.

    Returns:
        tuple: Updated boundary values.
    """
    arr[n,0] = arr[n,-2]
    arr[n,-1] = arr[n,1]
    return arr[n,0], arr[n,-1]

def predictor(arr1, arr2, i, j, a):
    arr2[i,j] = a * finite_difference_2nd_order(arr1, arr2, i, j, 0)
    return arr2[i,j]

def corrector(arr1, arr2, arr3, i, j, a):
    arr3[i,j] = a * finite_difference_2nd_order(arr1, arr2, i, j, 1)
    return arr3[i,j]

def solve_wave_equation(dx, dt, c, a):
    """
    PDE solver with inputs dx, dt, wave velocity, and parabolic term coefficients.

    Args:
        dx (float): Spatial discretization step.
        dt (float): Temporal discretization step.
        c (float): Wave velocity.
        a (int): Parameter a.

    Returns:
        tuple: Arrays u[t,x], k1[t,x], k2[t,x], time and spatial domains t,x,
               number of timesteps and grid nodes Nt, Nx, and index of spatial
               domain where x[index] = pi.
    """
    num_x_nodes = int(round(1/dx))
    num_t_nodes = int(round(4/dt))
    u = np.zeros((num_t_nodes, num_x_nodes))
    k1 = np.zeros((num_t_nodes, num_x_nodes))
    k2 = np.zeros((num_t_nodes, num_x_nodes))
    
    x = np.linspace(0, (num_x_nodes+1)*dx, num_x_nodes+2)
    for j in range(num_x_nodes):
        u[0,j] = sin(2*pi*x[j])

    for n in range(1, num_t_nodes):
        for j in range(1, num_x_nodes-1):
            k1[n,j] = predictor(u, k1, n, j, a)
        apply_periodic_boundary_conditions(k1, n)
        for j in range(1, num_x_nodes-1):
            k2[n,j] = corrector(u, k1, k2, n, j, a)
        apply_periodic_boundary_conditions(k2, n)
        for j in range(num_x_nodes):
            u[n,j] = u[n-1,j] + 0.5*(k1[n,j] + k2[n,j])
    return u, k1, k2

def calculate_absolute_error(u, indx, dt):
    abs_error = []
    for n in range(len(u)):
        diff = abs(u[n,indx] - cos(2*pi*n*dt))
        abs_error.append(diff)
    return abs_error

def find_convergence_order(t1, t2, abs_error_dx1, abs_error_dx2):
    try:
        index_dx1 = [find_nearest(np.array(t1), i*0.25) for i in range(7,16,2)]
        index_dx2 = [find_nearest(np.array(t2), i*0.25) for i in range(7,16,2)]

        conv_n = [log((abs_error_dx1[i]/abs_error_dx2[j]), 2) for i, j in zip(index_dx1, index_dx2)]

        table = [[t1[i], abs_error_dx1[i], abs_error_dx2[j]] for i, j in zip(index_dx1, index_dx2)]

        table = np.column_stack((table, conv_n))
        np.savetxt('table.txt', table)

    except ZeroDivisionError as e:
        print(f"Error: Division by zero - {e}")

    except Exception as e:
        print(f"An error occurred - {e}")

# Main part of the code
dx = 0.01
c = 1
dt = 0.5*dx/c

start_time = time.time()
Cns1, x1, t1, Nx1, Nt1, indx1 = initialize_parameters(dx, dt, c)
u1, k11, k21 = solve_wave_equation(dx, dt, c, Cns1)
abs_error_dx1 = calculate_absolute_error(u1, indx1, dt)

dx = 0.005
c = 1
dt = 0.5*dx/c

Cns2, x2, t2, Nx2, Nt2, indx2 = initialize_parameters(dx, dt, c)
u2, k12, k22 = solve_wave_equation(dx, dt, c, Cns2)
abs_error_dx2 = calculate_absolute_error(u2, indx2, dt)

print('######----------- Finding convergence order ----------####')
find_convergence_order(t1, t2, abs_error_dx1, abs_error_dx2)

elapsed_time = time.time() - start_time
print('Exiting...',' The system took '+str(elapsed_time)+' seconds to run')

# Help Section
"""
This code solves the wave equation using a finite difference scheme with predictor-corrector steps.

The key functions are:
- initialize_parameters: Initializes parameters for the solver.
- find_nearest: Finds the index in the array closest to the given value.
- finite_difference_2nd_order: Computes the finite difference scheme for 2nd order derivative.
- apply_periodic_boundary_conditions: Applies periodic boundary conditions.
- predictor: Computes the predictor step of the predictor-corrector scheme.
- corrector: Computes the corrector step of the predictor-corrector scheme.
- solve_wave_equation: Solves the wave equation.
- calculate_absolute_error: Calculates the absolute error of the solution.
- find_convergence_order: Finds the convergence order of the solution.

The code then applies these functions to solve the wave equation with different spatial discretizations, 
calculates the absolute errors, and finds the convergence order.
"""
