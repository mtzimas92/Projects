def forward_diff(func, nodes, dx):
    if dx == 0:
        raise ValueError("dx cannot be zero in numerical derivative function")
    return (func(nodes + 1) - func(nodes)) / dx

def backward_diff(func, nodes, dx):
    if dx == 0:
        raise ValueError("dx cannot be zero in numerical derivative function")
    return (func(nodes) - func(nodes - 1)) / dx

def central_diff(func, nodes, dx):
    if dx == 0:
        raise ValueError("dx cannot be zero in numerical derivative function")
    return (func(nodes + 1) - func(nodes - 1)) / (2 * dx)

def second_central_diff(func, nodes, dx):
    if dx == 0:
        raise ValueError("dx cannot be zero in numerical derivative function")
    return (func(nodes + 1) - 2 * func(nodes) + func(nodes - 1)) / (dx**2)

def sixth_order(func, nodes, dx):
    if dx == 0:
        raise ValueError("dx cannot be zero in numerical derivative function")
    return (-11/6 * func(nodes) + 3 * func(nodes + 1) - 3/2 * func(nodes + 2) + 1/3 * func(nodes + 3)) / dx

def forward_second_deriv(func, nodes, dx):
    if dx == 0:
        raise ValueError("dx cannot be zero in numerical derivative function")
    return (func(nodes) - 2 * func(nodes + 1) + func(nodes + 2)) / (dx**2)

def backward_second_deriv(func, nodes, dx):
    if dx == 0:
        raise ValueError("dx cannot be zero in numerical derivative function")
    return (func(nodes - 2) - 2 * func(nodes - 1) + func(nodes)) / (dx**2)

def mixed_deriv(func, nodes, dx_x, dx_y):
    if dx == 0:
        raise ValueError("dx cannot be zero in numerical derivative function")
    return (func(nodes[0] + 1, nodes[1]) - 2 * func(nodes) + func(nodes[0] - 1, nodes[1])) / (dx_x**2)
