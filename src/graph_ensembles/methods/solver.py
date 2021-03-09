import numpy as np


def sufficient_decrease_condition(
    f_old, f_new, alpha, grad_f, p, c1=1e-4, c2=0.9
):
    """return boolean indicator if upper wolfe condition are respected."""
    if isinstance(p, np.ndarray):
        sup = f_old + c1 * alpha * grad_f @ p.T
    else:
        sup = f_old + c1 * alpha * grad_f * p
    return bool(f_new < sup)


def linsearch_fun(x, dx, alpha):
    """ Function searching the descent direction.
    """
    eps = 1e-4
    ind = dx != 0
    alpha0 = (eps - 1) * x[ind] / dx[ind]
    if isinstance(alpha0, np.ndarray):
        # assure that the zetas are positive
        for a in alpha0:
            if a >= 0:
                alfa = min(alpha, a)

    else:
        # assure that zeta is positive
        if alpha0 > 0:
            alpha = min(alfa, alpha0)

    return alpha


def solver(x0,
           fun,
           fun_jac=None,
           tol=1e-6,
           jac_tol=1e-6,
           eps=1e-12,
           max_iter=100,
           method="newton",
           full_return=False,
           verbose=False):
    """Find roots of eq. f = 0, using newton or fixed-point methods.

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
    flat_jac = 0
    f = fun(x)
    norm = np.linalg.norm(f)
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
        f_old = f
        # Compute update
        if method == "newton":
            H = fun_jac(x)
            if np.linalg.norm(H) > jac_tol:
                flat_jac = 0
                dx = - f/H
            else:
                flat_jac += 1
                dx = - f/jac_tol
        elif method == "fixed-point":
            dx = f - x

        # Line search
        # alpha = linsearch_fun(x, dx, 1)

        # Update values
        x = x + alpha * dx
        f = fun(x)

        if (flat_jac > 2) and np.any(np.sign(f) != np.sign(f_old)):
            alpha = alpha / 2
        elif (flat_jac > 2):
            alpha = min(1, alpha * 1.2)
        else:
            alpha = min(1, alpha * 2)

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
