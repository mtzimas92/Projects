def gen_non_uniform_x(x_i, dx_values, nodes):
    if nodes <= 0 or dx == 0:
        raise ValueError("Invalid nodes or dx value in gen_non_uniform_x function")
    return [x_i + sum(dx_values[:j]) for j in range(nodes)]

def gen_x(x_i, dx, nodes):
    if nodes <= 0 or dx == 0:
        raise ValueError("Invalid nodes or dx value in gen_x function")
    x_min = x_i - ((nodes - 1) / 2) * dx
    return [x_min + j * dx for j in range(nodes)]