import numpy as np

def newton_raphson(
        f: np.vectorize,
        df: np.vectorize,
        x0: float,
        tolerance: float = 1e-6,
        max_iter: int = 100
    ) -> tuple[float, int]:

    """
    Approximates the root of a given function using the Newton-Raphson method.

    Parameters
    --------------
    f: 
        The function for which the root is sought. (function).
    df: 
        The derivative of the function f (function).
    x0: 
        The initial estimate of the root (float).
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
    except:
        raise ValueError("The initial estimate must be a float type.")
    
    # Set x = x0
    x = x0

    # Check if df(x0)/dx = 0
    if df(x0) == 0:
        raise Exception("The derivative of the function at point x0 can not be zero. Choose a best estimate.")

    # Newton-Raphson method
    for k in range(1, max_iter+1):
        x_k = x - (f(x) / df(x))

        # Check if x_k - x is less than the tolerance
        if abs(x_k - x) < tolerance:
            return x_k, k

        # Update x
        x = x_k
    
    raise Exception("Newton-Raphson method did not converge within the specified number of iterations.")

f = lambda x: x**2 - 4
df = lambda x: 2*x
x0 = 1

print(newton_raphson(f, df, x0))
