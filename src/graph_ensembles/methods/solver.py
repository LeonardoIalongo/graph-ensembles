import numpy as np


def linesearch(x, dx, f, f_p):
    return 1


def solver(x0,
           fun,
           fun_jac=None,
           tol=1e-6,
           eps=1e-12,
           max_iter=100,
           method="newton",
           full_return=False,
           verbose=False):
    """Find roots of eq. f(x) = 0, using newton or fixed-point methods.

    Parameters
    ----------
    x0: float or np.ndarray
        starting points of the method
    fun: function
        function of which to find roots.
    fun_jac: function
        Jacobian of the function.
    tol : float
        tolerance for the exit condition on the norm
    eps : float
        tolerance for the exit condition on difference between two iterations
    max_iter : int or float
        maximum number of iteration
    method: 'newton' or 'fixed-point'
        selects which method to use for the solver
    full_return: boolean
        if true return all info on convergence
    verbose: boolean
        if true print debug info while iterating
    Returns
    -------
    tuple
        Returns a named tuple with the solution, the number
        of iterations and the norm sequence.
    """
    n_iter = 0
    x = x0
    alpha = 1
    f = fun(x)
    norm = np.abs(f)
    diff = 1

    if verbose:
        print("x0 = {}".format(x))
        print("|f(x0)| = {}".format(norm))

    if full_return:
        norm_seq = [norm]
        diff_seq = [diff]
        alpha_seq = [1]

    while (norm > tol and n_iter < max_iter and diff > eps):
        # Save previous iteration
        x_old = x

        # Compute update
        if method == "newton":
            f_p = fun_jac(x)
            if f_p > tol:
                dx = - f/f_p
            else:
                dx = - f/tol  # Note that H is always positive
        elif method == "fixed-point":
            dx = f - x

        # Line search
        # alpha = linesearch(x, dx, f, f_p)

        # Update values
        x = x + alpha * dx
        f = fun(x)

        # stopping condition computation
        norm = np.linalg.norm(f)
        diff = np.linalg.norm(x - x_old)

        if full_return:
            norm_seq.append(norm)
            diff_seq.append(diff)
            alpha_seq.append(alpha)

        # step update
        n_iter += 1
        if verbose:
            print("    Iteration {}".format(n_iter))
            print("    fun = {}".format(f))
            if method == 'newton':
                print("    fun_prime = {}".format(f_p))
            print("    dx = {}".format(dx))
            print("    x = {}".format(x))
            print("    alpha = {}".format(alpha))
            print("    |f(x)| = {}".format(norm))
            print("    diff = {}".format(diff))
            print(' ')

    if verbose:
        print(' ')

    if full_return:
        return (x, n_iter, np.array(norm_seq),
                np.array(norm_seq), np.array(alpha_seq))
    else:
        return x
