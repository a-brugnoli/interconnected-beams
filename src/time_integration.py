import numpy as np
from src.linear_algebra import solve_bcs
from scipy.sparse.linalg import spsolve


def implicit_midpoint(x_0, M, A, B, forcing, dt, nt):
    x_solution = np.zeros((len(x_0), nt+1))
    x_solution[: ,0] = x_0

    x_old = x_0

    A_imp_midpoint = (M - dt/2 * A)

    for n in range(nt):

        mid_time = dt*(n + 1/2)
        b = (M + dt/2 * A) @ x_old + dt * B @ forcing(mid_time)
        x_new = solve_bcs(A_imp_midpoint, b)
        
        x_solution[:, n+1] = x_new
        x_old = x_new

    return x_solution
