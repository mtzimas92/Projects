import numpy as np
import time

def find_nearest(array, value):
    """
    Finds the index in the array that is closest to the input value.
    
    Args:
        array (numpy.ndarray): Input array.
        value (float): Target value.
        
    Returns:
        int: Index of the closest value in the array.
    """
    if array is None or len(array) == 0:
        raise ValueError("Input array is empty or None in find_nearest function")
    return np.abs(array - value).argmin()


def apply_periodic_boundary_conditions(array, n):
    """
    Applies periodic boundary conditions to the array.
    
    Args:
        array (numpy.ndarray): Input array.
        n (int): Time step.
        
    Returns:
        float, float: Values at the boundaries.
    """
    array[n,0] = array[n,-2]
    array[n,-1] = array[n,1]
    return array[n,0], array[n,-1]

def calculate_second_order_fd(array, i, j):
    """
    Calculates a second-order finite difference.
    
    Args:
        array (numpy.ndarray): Input array.
        i (int): Time step.
        j (int): Spatial step.
        
    Returns:
        float: Second-order finite difference.
    """
    return array[i, j+1] - 2 * array[i, j] + array[i, j-1]

def initialize_parameters(dx, dt, a_coefficient, b_coefficient, c_coefficient):
    """
    Initializes parameters and spatial/temporal domains.
    
    Args:
        dx (float): Spatial step size.
        dt (float): Temporal step size.
        a_coefficient (float): Coefficient a.
        b_coefficient (float): Coefficient b.
        c_coefficient (float): Coefficient c.
        
    Returns:
        tuple: Coefficients C1_coeff, C2_coeff, C3_coeff, C4_coeff, spatial domain x, temporal domain t, 
               number of spatial steps Nx, number of temporal steps Nt, and index indx.
    """
    if any(val <= 0 for val in (dx, dt, a_coefficient, b_coefficient, c_coefficient)):
        raise ValueError("All input parameters must be positive numbers.")
    
    C1_coeff = -(a_coefficient / (2 * dt))
    C2_coeff = ((c_coefficient**2) * dt) / (dx**2)
    C3_coeff = b_coefficient / 2
    C4_coeff = a_coefficient / 2
    
    Nx = int(round(4 * np.pi / dx))
    x = np.linspace(-dx, Nx * dx, Nx + 1)
    Nx = len(x)
    
    Nt = int(round(16 * np.pi / dt))
    t = np.linspace(0, Nt * dt, Nt + 1)
    Nt = len(t)
    
    indx = find_nearest(np.array(x), np.pi)
    
    return C1_coeff, C2_coeff, C3_coeff, C4_coeff, x, t, Nx, Nt, indx



def solve_pde(dx, dt, a, b, c):
    """
    Solves the PDE using finite difference methods.
    
    Args:
        dx (float): Spatial step size.
        dt (float): Temporal step size.
        a (float): Coefficient a.
        b (float): Coefficient b.
        c (float): Coefficient c.
        
    Returns:
        tuple: Arrays u and v representing the solution.
    """
    global u, v
    u = np.zeros((Nt, Nx))
    v = np.zeros((Nt, Nx))

    for j in range(Nx):
        u[0,j] = np.cos(x[j])
        u[1,j] = u[0,j]

    for j in range(1, Nx-1):
        v[0,j] = C1 * (calculate_second_order_fd(u, 0, j))
    apply_periodic_boundary_conditions(v, 0)

    for j in range(1, Nx-1):
        v[1,j] = v[0,j] + C2 * (calculate_second_order_fd(u, 0, j)) + C3 * (calculate_second_order_fd(v, 0, j))
    apply_periodic_boundary_conditions(v, 1)

    # for n in range(1, Nt-1):
        for j in range(1, Nx-1):
            u[n+1,j] = u[n,j] + dt * (v[n,j]) + C4 * calculate_second_order_fd(u, n, j)
            v[n+1,j] = v[n,j] + C2 * calculate_second_order_fd(u, n, j) + C3 * calculate_second_order_fd(v, n, j)
        apply_periodic_boundary_conditions(u, n+1)
        apply_periodic_boundary_conditions(v, n+1)

    return u, v

# Example usage (with time calculation)
start_time = time.time()
a = b = 0.25
c = 1
dx = 0.01*4*pi
dt = 0.25*dx/c
conv_n1, conv_n2 =[],[]
abs_error_dx1 =[]
abs_error_dx2 = []
index_dx1 = []
index_dx2 =[]
table1,table2 = [],[]

print('######----------- Starting the solution ----------####')
print('Initializing values for solver,dx = 0.01*pi, b = 0.25')
C1,C2,C3,C4,x1,t1,Nx,Nt,indx = initialize_parameters(dx,dt,a,b,c)
print('Running the solver for dx = 0.01*pi, b = 0.25')
u1,v1 = solve_pde(dx,dt,a,b,c) 

print('Calculating absolute error for first solution')
for n in range(Nt):
    diff = abs(u1[n,indx] -(-cos(n*dt)))
    abs_error_dx1.append(diff)


elapsed_time = time.time() - start_time
print(f'Exiting... The system took {elapsed_time} seconds to run')


# Help Section
"""
This code implements a numerical solver for a partial differential equation (PDE) using finite difference methods.
The PDE is of the form: a*u_xx + b*u_t - c^2*u_tt = 0, solved in a periodic domain.

Functions:
- find_nearest(array, value): Finds the index in the array that is closest to the input value.
- apply_periodic_boundary_conditions(array, n): Applies periodic boundary conditions to the array at time step n.
- calculate_second_order_fd(array, i, j): Calculates a second-order finite difference.
- initialize_parameters(dx, dt, a, b, c): Initializes parameters and spatial/temporal domains.
- solve_pde(dx, dt, a, b, c): Solves the PDE using finite difference methods.
"""
