"""
    Solves the system of linear equations Ax = b using the Gauss-Seidel iterative method.
    Parameters:
    A (list of list of floats): Coefficient matrix.
    b (list of floats): Constant terms.
    tol (float, optional): Tolerance for the stopping criterion. Defaults to 0.001.
    max_iterations (int, optional): Maximum number of iterations. Defaults to 100.

    Returns:
    list of floats: Solution vector x if the method converges within the maximum number of iterations.
    None: If the method does not converge.
    """
def gaussSeidel(A, b, tol=0.001, max_iterations=100):
    n = len(A)
    x = [0.0 for _ in range(n)]

    for k in range(max_iterations):
        x_old = x[:]

        for i in range(n):

            sum_L = sum(A[i][j] * x[j] for j in range(i))
            sum_U = sum(A[i][j] * x_old[j] for j in range(i + 1, n))
            x[i] = (b[i] - sum_L - sum_U) / A[i][i]

        if max(abs(x[i] - x_old[i]) for i in range(n)) < tol:
            return x

    print("Gauss-Seidel method did not converge.")
    return None



"""
    Solves the system of linear equations Ax = b using the Jacobi iterative method.
    Parameters:
    A (list of list of floats): Coefficient matrix.
    b (list of floats): Constant terms.
    tol (float, optional): Tolerance for the stopping criterion. Defaults to 0.001.
    max_iterations (int, optional): Maximum number of iterations. Defaults to 100.

    Returns:
    list of floats: Solution vector x if the method converges within the maximum number of iterations.
    None: If the method does not converge.
    """
def jacobi(A, b, tol=0.001, max_iterations=100):
    n = len(A)
    x = [0.0 for _ in range(n)]
    D = [[A[i][j] if i == j else 0 for j in range(n)] for i in range(n)]
    L = [[A[i][j] if i > j else 0 for j in range(n)] for i in range(n)]
    U = [[A[i][j] if i < j else 0 for j in range(n)] for i in range(n)]

    D_inv = [[1 / D[i][i] if i == j else 0 for j in range(n)] for i in range(n)]

    for k in range(max_iterations):
        x_new = [0.0 for _ in range(n)]
        for i in range(n):
            sum_LU = sum((L[i][j] + U[i][j]) * x[j] for j in range(n))
            x_new[i] = D_inv[i][i] * (b[i] - sum_LU)

        if max(abs(x_new[i] - x[i]) for i in range(n)) < tol:
            return x_new

        x = x_new[:]

    print("Jacobi method did not converge.")
    return None



"""
    Main function to test the Gauss-Seidel and Jacobi methods with a sample system of linear equations.
    """
def main():
    A = [
        [4, 2, 0],
        [2, 10, 4],
        [0, 4, 5]
    ]
    b = [2, 6, 5]

    print("Solution using the gauss seidel method:")
    solution_gauss_seidel = gaussSeidel(A, b)
    if solution_gauss_seidel is not None:
        print( solution_gauss_seidel)

    print("\nSolution using the jacobi method:")
    solution_jacobi = jacobi(A, b)
    if solution_jacobi is not None:
        print( solution_jacobi)



if __name__ == "__main__":
    main()
