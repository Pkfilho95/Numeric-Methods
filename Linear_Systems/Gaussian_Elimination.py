import numpy as np

def gaussian(
        A: np.ndarray | list[list[int | float]],
        B: np.ndarray | list[int | float] | list[list[int | float]]
    ) -> tuple[np.ndarray, np.ndarray]:
    
    """
    Gaussian elimination to solve a linear system Ax = B.

    Parameters
    --------------
    A: 
        Coefficient matrix of the system (ndarray or list[list[int | float]]).
    B: 
        Vector of independent terms (np.ndarray or list[int | float] or list[list[int | float]]).
    
    Returns
    --------------
    Tuple containing the echelon form of A and B.
    """
    
    try:
        A = np.array(A, dtype="float64")
        B = np.array(B, dtype="float64")
    except:
        raise TypeError("A and B only accepts ndarray or list[list[int | float]] types.")
    
    if np.linalg.det(A) == 0:
        raise ValueError("There is no solution for the given system of equations.")
    
    # Get dimension
    n = A.shape[0]

    # Reshape to make sure B is a column matrix
    B = B.reshape((n, 1))

    # Concatenate A and B for easy manipulation [A|B]
    M = np.concatenate((A, B), axis=1)

    for j in range(0, n-1):

        # Swap rows by max pivot
        index_max_pivot = np.argmax(abs(M[j:,j])) + j
        if index_max_pivot != j:
            M[j], M[index_max_pivot] = M[index_max_pivot].copy(), M[j].copy()

        # Elimination step
        for i in range(j+1, n):
            pivot = M[i,j] / M[j,j]
            M[i] -= pivot * M[j]

    # Split M into A and B
    A = M[:, :-1]
    B = M[:, -1]

    return A, B

def solve(
        A: np.ndarray,
        B: np.ndarray
        ) -> np.ndarray:
    
    """
    Function solving the problem Ax = B after the Gaussian elimination.

    Parameters
    --------------
    A: 
        Coefficient matrix of the system, after Gaussian elimination (ndarray).
    B: 
        Vector of independent terms, after Gaussian elimination (ndarray).

    Returns
    --------------
    Solution vector.
    """

    # Get dimension
    n = A.shape[0]

    # Set the solution matrix
    x = np.zeros(n)

    # Back substitution
    for i in range(n-1, -1, -1):
        x[i] = (B[i] - np.dot(A[i,i+1:], x[i+1:])) / A[i,i]
    
    return x

A = [[3,2,4],[1,1,2],[4,3,-2]]
B = [1,2,3]

A,B = gaussian(A,B)

# expected output [-3. 5. 0.]
print(solve(A,B))   