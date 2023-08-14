import numpy as np
import matplotlib.pyplot as plt

def runge_kutta(
        f: np.vectorize,
        x0: float,
        y0: float,
        h: float,
        stop: float,
    ) -> tuple[np.ndarray, np.ndarray]:

    """
    Solves a first-order ordinary differential equation (ODE) using the Runge-Kutta's 4th order method.

    Parameters
    --------------
    f:
        The function representing the derivative of the solution (dy/dx) (function).
    x0: 
        The initial value of the independent variable (float).
    y0: 
        The initial value of the dependent variable at x0 (y(x0)) (float).
    h: 
        The step size for advancing along the domain (float).
    stop: 
        The stopping point for the integration (float).
    
    Returns
    --------------
    Tuple containing the x and y values of the approximate solution.
    """

    try:
        x0 = float(x0)
        y0 = float(y0)
    except:
        raise TypeError("The initial values must be float types.")
    
    try:
        h = float(h)
    except:
        raise TypeError("The step size must be float type.")
    
    try:
        stop = float(stop)
    except:
        raise TypeError("The stopping point must be float type.")

    # Values of x and y
    x = [x0]
    y = [y0]

    # Euler's method
    while x[-1] < stop:
        k1 = h * f(x[-1], y[-1])
        k2 = h * f(x[-1] + h/2, y[-1] + k1/2)
        k3 = h * f(x[-1] + h/2, y[-1] + k2/2)
        k4 = h * f(x[-1] + h, y[-1] + k3)

        y_i = y[-1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x_i = x[-1] + h

        y.append(y_i)
        x.append(x_i)
    
    return x, y

f = lambda x,y: -2 * x
x0 = 0
y0 = 1
h = 0.1
stop = 2

x,y = runge_kutta(f, x0, y0, h, stop)

# Exact solution
exact_solution = lambda x: 1 - x**2
x_exact = np.linspace(x0, stop, 1000)
y_exact = exact_solution(x_exact)

# Plotting
plt.plot(x, y, label="Runge-Kutta's 4th Order Method")
plt.plot(x_exact, y_exact, label="Exact Solution")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Runge-Kutta's 4th Order Method: y\'(x) = -2x")
plt.legend()
plt.grid()
plt.show()
