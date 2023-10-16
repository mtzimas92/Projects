def factorial(num):
    """Calculate the factorial of a nonnegative integer 'num'."""
    if num <= 1:
        return 1.0
    else:
        return float(num * factorial(num - 1))

def double_factorial(num):
    """Calculate the double factorial of a nonnegative integer 'num'."""
    if num <= 1:
        return 1.0
    else:
        return float(num * double_factorial(num - 2.0))

def power(x, y):
    """Raise 'x' to the power 'y', where 'y' must be a nonnegative integer."""
    result = 1.0
    for i in range(int(y)):   # 'y' must be a nonnegative integer, so converting it to an integer
        result *= x  
    return result

def taylor(xin, order):
    """Calculate the Taylor series approximation for a given 'xin' and 'order'."""
    n = order
    xl = float(xin)  # Convert 'xin' to a float to ensure compatibility with operations
    summation = 0.0
    for i in range(n):
        sign = power((-1), i)
        a1 = float(factorial(2 * i))
        a2 = power(xl, i)
        b1 = float(4 ** i)
        b2 = factorial(i)
        b3 = float(b2 * b2)
        b4 = float(1 - 2 * i)
        numerator = float(sign * a1 * a2)
        denominator = float(b1 * b3 * b4)
        summation += numerator / denominator
    return summation
