import numpy as np

def Trapezoidal(
        f: np.vectorize,
        a: float,
        b: float,
        n: int
    ) -> float:
    
    """
    Calculates the integral of a function using the trapezoidal rule.

    Parameters
    --------------
    f:
        The function to be integrated (function).
    a: 
        The lower limit of the integration interval (float).
    b: 
        The upper limit of the integration interval (float).
    n:
        The number of subintervals to divide the interval [a, b] (int).
    
    Returns
    --------------
    Approximation of the integral.
    """

    try:
        a = float(a)
        b = float(b)
    except:
        raise TypeError("The lower and upper limit must be float types.")
    
    try:
        n = int(n)
    except:
        raise TypeError("The number of subintervals must be int type.")
    
    # Width of each subinterval
    h = (b - a) / n

    # Interval array
    x = np.arange(a, b+h, h)

    # f(xi) array
    f_array = [(h / 2) * f(x[i]) for i in range(0, n+1)]
    f_array = np.array(f_array)

    # Integration
    f_array[1:-1] = 2 * f_array[1:-1]
    integral = np.sum(f_array)

    return integral

f = lambda x: x**2
a = 0
b = 2
n = 1000

print(Trapezoidal(f, a, b, n))
print((b**3)/3)
