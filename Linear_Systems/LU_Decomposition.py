import numpy as np

def LU(
        A: np.ndarray | list[list[int | float]]
        ) -> np.ndarray:
    
    """
    LU decomposition to solve a linear system Ax = B.

    Parameters
    --------------
    A: 
        Coefficient matrix of the system (ndarray or list[list[int | float]]).
    
    Returns
    --------------
    Tuple containing the L and U matrices.
    """

    try:
        A = np.array(A, dtype="float64")
    except:
        raise TypeError("A only accepts ndarray or list[list[int | float]] types.")

    # Get dimension
    n = A.shape[0]

    # Check if it is possible to do LU decomposition
    for i in range(0, n):
        if np.linalg.det(A[:i,:i]) == 0:
            raise ValueError("It is not possible to do the LU decomposition for the given system of equations.")
    
    # Create L
    L = np.eye(n, n, dtype="float64")
    
    # Gaussian elimination
    for j in range(0, n):
        for i in range(j+1, n):
            L[i,j] = A[i,j] / A[j,j]
            A[i] -= L[i,j] * A[j]
    
    # U = A
    return L,A

def solve(
        L: np.ndarray,
        U: np.ndarray,
        B: np.ndarray | list[int | float] | list[list[int | float]]
        ) -> np.ndarray:
    
    """
    Function solving the problem Ax = B after the LU decomposition.
    First step: solve Ly = B.
    Second step: solve Ux = y.

    Parameters
    --------------
    L: 
        Lower matrix (ndarray).
    U: 
        Upper matrix (ndarray).
    B: 
        Vector of independent terms, (np.ndarray or list[int | float] or list[list[int | float]]).

    Returns
    --------------
    Solution vector.
    """

    try:
        B = np.array(B, dtype="float64")
    except:
        raise TypeError("B only accepts ndarray or list[list[int | float]] types.")
    
    # Get dimension
    n = L.shape[0]

    # Set the y matrix
    y = np.zeros(n)

    # Solve Ly = B
    for i in range(0, n):
        y[i] = (B[i] - np.dot(L[i,:i+1], y[:i+1])) / L[i,i]
    
    # Set the x matrix (solution)
    x = np.zeros(n)

    # Solve Ux = y
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i,i+1:], x[i+1:])) / U[i,i]
    
    return x

A = [[3,2,4],[1,1,2],[4,3,-2]]
B = [1,2,3]

L,U = LU(A)

# expected output [-3. 5. 0.]
print(solve(L,U,B))   
