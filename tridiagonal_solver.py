import numpy as np

def tridiagonal_solver(a, b, c, d):
    """
    Solves a tridiagonal system of linear equations using the Thomas algorithm.

    Args:
        a (list): Lower diagonal elements.
        b (list): Main diagonal elements.
        c (list): Upper diagonal elements.
        d (list): Right-hand side values.

    Returns:
        list: Solution vector.
    """
    n = len(b)
    x = np.zeros(n)
    cp = np.zeros(n)
    dp = np.zeros(n)

    # Forward elimination
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        cp[i] = c[i] / (b[i] - cp[i-1] * a[i])
        dp[i] = (d[i] - a[i] * dp[i-1]) / (b[i] - a[i] * cp[i-1])

    # Back substitution
    x[-1] = dp[-1]
    for i in range(n-2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i+1]

    return x

# Example usage
a = [0, 1, 2, 7, 1, 5, 7, -1, -8, 4, 4, 4, -1, -2]
b = [-1, 2, 1, 2, 9, 5, -2, 9, 7, 2, 2, 2, -1, 2]
c = [3, 1, 3, -1, -2, 5, 1, 6, 2, 2, 1, 3, 1, 0]

# Test cases
d1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4]
x1 = tridiagonal_solver(a, b, c, d1)

d2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4000]
x2 = tridiagonal_solver(a, b, c, d2)

print(x1)
print(x2)
