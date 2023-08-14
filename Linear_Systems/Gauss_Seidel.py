import numpy as np

def gauss_seidel(
        A: np.ndarray | list[list[int | float]],
        B: np.ndarray | list[int | float] | list[list[int | float]],
        x: np.ndarray | list[int | float] = None,
        tolerance: float = 1e-6,
        max_iter: int = 1000
    ) -> np.ndarray:
    
    """
    Gauss-Seidel method to solve a linear system Ax = B.

    Parameters
    --------------
    A: 
        Coefficient matrix of the system (ndarray or list[list[int | float]]).
    B: 
        Vector of independent terms (np.ndarray or list[int | float] or list[list[int | float]]).
    x (default None): 
        Vector with the initial estimate of the solution. (np.ndarray or list[int | float]).
    tolerance (default 1e-6):
        Convergence tolerance to stop iteration (float).
    max_iter (default 1000):
        Maximum number of iterations (int).
    
    Returns
    --------------
    Tuple containing the solution x, number of iterations and the error.
    """

    try:
        A = np.array(A, dtype="float64")
        B = np.array(B, dtype="float64")
    except:
        raise TypeError("A and B only accepts ndarray or list[list[int | float]] types.")
    
    if np.linalg.det(A) == 0:
        raise ValueError("It is not possible to do the Jacobi method for the given system of equations.")
    
    # Get dimension
    n = A.shape[0]
    
    # Initial estimate x0
    if not x:
        x = np.zeros((n, 2))
    else:
        x = np.array(x, dtype="float64")

        # Reshape to make sure x is a column matrix.
        x = x.reshape((n, 1))

        # x[:,0] is at iteration k and x[:,1] is at iteration k+1.
        x = np.concatenate((x, np.zeros((n, 1))), axis=1)

    # Jacobi method
    iteration = 0
    error = 1

    while error > tolerance and iteration < max_iter:
        for i in range(0, n):
            x[i,1] = (B[i] - np.dot(A[i,:i], x[:i,1]) - np.dot(A[i,i+1:],x[i+1:,1])) / A[i,i]
        
        # Norm of the x
        error = np.linalg.norm(x[:,1] - x[:,0], np.inf)

        # Swap columns (k for k+1)
        x[:,0] = x[:,1].copy()

        iteration += 1 
    
    # Check the convergence
    if iteration == max_iter and error > tolerance:
        raise Exception("It did not converge.")
    
    return x[:,1], iteration, error

A = [[-3,1,1],[2,5,1],[2,3,7]]
B = [2,5,-17]
x = [1,1,-1]

tolerance = 5e-3
max_iter = 100

# expect [-1.,  2., -3.], 10, 3.8e-3)
print(gauss_seidel(A, B, x, tolerance, max_iter))
