import numpy as np

def secant(
        f: np.vectorize,
        x0: float,
        x1: float,
        tolerance: float = 1e-6,
        max_iter: int = 100
    ) -> tuple[float, int]:

    """
    Approximates the root of a given function using the Secant method.

    Parameters
    --------------
    f: 
        The function for which the root is sought. (function).
    x0: 
        The first initial estimate of the root (float).
    x1: 
        The second initial estimate of the root (float).
    tolerance (default 1e-6):
        The maximum allowable error in the root approximation (float).
    max_iter (default 100):
        Maximum number of iterations (int).
    
    Returns
    --------------
    Tuple containing the approximate root and the number of iterations performed.
    """

    try:
        x0 = float(x0)
        x1 = float(x1)
    except:
        raise ValueError("The initial estimates must be a float types.")

    # Secant method
    for k in range(1, max_iter+1):
        
        # Calculate the slope of the line (derivative)
        m = (f(x1) - f(x0)) / (x1 - x0)

        # Calculate new point x by Newton-Raphson method
        x_k = x1 - (f(x1) / m)

        # Check if x_k - x1 is less than the tolerance
        if abs(x_k - x1) < tolerance:
            return x_k, k

        # Update x
        x0, x1 = x1, x_k
    
    raise Exception("Newton-Raphson method did not converge within the specified number of iterations.")

f = lambda x: x**2 - 4
x0 = 1
x1 = 3

print(secant(f, x0, x1))
