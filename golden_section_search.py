import csv

def f(x):
    return x**2 - 4*x + 4

def g(x):
    return x**3 - 6*x**2 + 11*x - 6

def golden_section_search_recursive(func, a, b, epsilon=1e-6, max_iterations=1000):
    """
    Perform a golden section search to find the minimum of a function within an interval.

    Args:
        func (callable): The function to minimize.
        a (float): The left endpoint of the interval.
        b (float): The right endpoint of the interval.
        epsilon (float, optional): The desired level of accuracy. Defaults to 1e-6.
        max_iterations (int, optional): The maximum number of iterations. Defaults to 1000.

    Returns:
        float: The x-coordinate of the estimated minimum.

    Raises:
        ValueError: If the interval [a, b] is not valid.

    """
    phi = (1 + 5**0.5) / 2

    x1 = a + (b - a) / phi
    x2 = b - (b - a) / phi

    if abs(b - a) < epsilon:
        return (a + b) / 2

    f1, f2 = func(x1), func(x2)

    if f1 < f2:
        return golden_section_search_recursive(func, a, x2, epsilon, max_iterations)
    else:
        return golden_section_search_recursive(func, x1, b, epsilon, max_iterations)

def save_results_to_csv(results, filename):
    with open(filename, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(results)

# Define the interval [a, b]
a, b = 0.15, 0.5

# Run Golden Section Search for f(x)
try:
    result_f = golden_section_search_recursive(f, a, b)
    print(f"The minimum occurs at x = {result_f} with f(x) = {f(result_f)}")
except Exception as e:
    print(f"Error: {e}")

# Run Golden Section Search for g(x) (if defined)
try:
    result_g = golden_section_search_recursive(g, a, b)
    print(f"The minimum occurs at x = {result_g} with g(x) = {g(result_g)}")
except Exception as e:
    print(f"Error: {e}")
    
# Help Section:
"""
To run this script, follow these steps:

1. Define the function you want to minimize, such as 'f(x)' or 'g(x)'.
2. Set the interval [a, b] where the minimum is expected to occur.
3. Optionally, adjust the desired level of accuracy (epsilon) and maximum number of iterations (max_iterations).

You can change the following parameters:
- func: The function you want to minimize.
- a: The left endpoint of the interval.
- b: The right endpoint of the interval.
- epsilon: The desired level of accuracy (default is 1e-6).
- max_iterations: The maximum number of iterations (default is 1000).

Example:
    a, b = 0.15, 0.5
    result = golden_section_search_recursive(f, a, b, epsilon=1e-8, max_iterations=2000)

Note: Ensure that the function 'f(x)' or 'g(x)' is defined before running the script.
"""