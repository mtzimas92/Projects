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
    Nx = int(round(1/dx))
    x = np.linspace(0, (Nx+1)*dx, Nx+2)
    Nx = len(x)
    Nt = int(round(4/dt))
    t = np.linspace(0, Nt*dt, Nt+1)
    Nt = len(t)
    indx = find_nearest(x, 0.25)
    return C1, x, t, Nx, Nt, indx

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

def periodic_boundary_conditions(arr, n):
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

def pr(arr1, arr2, i, j, a):
    arr2[i,j] = a * finite_difference_2nd_order(arr1, arr2, i, j, 0)
    return arr2[i,j]

def cor(arr1, arr2, arr3, i, j, a):
    arr3[i,j] = a * finite_difference_2nd_order(arr1, arr2, i, j, 1)
    return arr3[i,j]

def solver(dx, dt, c, a):
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
    Nx = int(round(1/dx))
    Nt = int(round(4/dt))
    u = np.zeros((Nt, Nx))
    k1 = np.zeros((Nt, Nx))
    k2 = np.zeros((Nt, Nx))
    
    x = np.linspace(0, (Nx+1)*dx, Nx+2)
    for j in range(Nx):
        u[0,j] = sin(2*pi*x[j])

    for n in range(1, Nt):
        for j in range(1, Nx-1):
            k1[n,j] = pr(u, k1, n, j, a)
        periodic_boundary_conditions(k1, n)
        for j in range(1, Nx-1):
            k2[n,j] = cor(u, k1, k2, n, j, a)
        periodic_boundary_conditions(k2, n)
        for j in range(Nx):
            u[n,j] = u[n-1,j] + 0.5*(k1[n,j] + k2[n,j])
    return u, k1, k2

def calculate_absolute_error(u, indx, dt):
    abs_error = []
    for n in range(len(u)):
        diff = abs(u[n,indx] - cos(2*pi*n*dt))
        abs_error.append(diff)
    return abs_error

def find_convergence_order(t1, t2, abs_error_dx1, abs_error_dx2):
    index_dx1 = [find_nearest(np.array(t1), i*0.25) for i in range(7,16,2)]
    index_dx2 = [find_nearest(np.array(t2), i*0.25) for i in range(7,16,2)]
    
    conv_n = [log((abs_error_dx1[i]/abs_error_dx2[j]), 2) for i, j in zip(index_dx1, index_dx2)]
    
    table = [[t1[i], abs_error_dx1[i], abs_error_dx2[j]] for i, j in zip(index_dx1, index_dx2)]
    
    table = np.column_stack((table, conv_n))
    np.savetxt('table2.txt', table)

dx = 0.01
c = 1
dt = 0.5*dx/c

Cns1, x1, t1, Nx1, Nt1, indx1 = initialize_parameters(dx, dt, c)
u1, k11, k21 = solver(dx, dt, c, Cns1)
abs_error_dx1 = calculate_absolute_error(u1, indx1, dt)

dx = 0.005
c = 1
dt = 0.5*dx/c

Cns2, x2, t2, Nx2, Nt2, indx2 = initialize_parameters(dx, dt, c)
u2, k12, k22 = solver(dx, dt, c, Cns2)
abs_error_dx2 = calculate_absolute_error(u2, indx2, dt)

print('######----------- Finding convergence order ----------####')
find_convergence_order(t1, t2, abs_error_dx1, abs_error_dx2)

elapsed_time = time.time() - start_time
print('Exiting...',' The system took '+str(elapsed_time)+' seconds to run')
