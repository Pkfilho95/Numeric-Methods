import numpy as np

def bisection(
        f: np.vectorize,
        a: float,
        b: float,
        tolerance: float = 1e-6,
        max_iter: int = 100
    ) -> tuple[float, int]:

    """
    Approximates the root of a given function using the Bisection method.

    Parameters
    --------------
    f: 
        The function for which the root is sought. (function).
    a: 
        The left endpoint of the interval (float).
    b: 
        The right endpoint of the interval (float).
    tolerance (default 1e-6):
        The maximum allowable error in the root approximation (float).
    max_iter (default 100):
        Maximum number of iterations (int).
    
    Returns
    --------------
    Tuple containing the approximate root and the number of iterations performed.
    """

    try:
        a = float(a)
        b = float(b)
    except:
        raise TypeError("The lower and upper limit must be float types.")
    
    if f(a) * f(b) >= 0:
        raise ValueError("The function must change sign in the interval [a, b].")
    
    # Bisection method
    for k in range(1, max_iter+1):

        # Point within the interval [a,b]
        c = (a + b) / 2

        # Check if f(c) is close enough to zero or width of the interval [a, b] is less than a tolerance
        if abs(f(c)) < tolerance or abs(b - a) < tolerance:
            return c, k
        
        # Update the interval
        if f(c) * f(a) < 0:
            b = c
        else:
            a = c
    
    raise Exception("Bisection method did not converge within the specified number of iterations.")

f = lambda x: x**2 - 4
a = 0
b = 3

print(bisection(f, a, b))
