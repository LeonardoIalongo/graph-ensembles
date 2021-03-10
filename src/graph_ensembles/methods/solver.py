import numpy as np


class Solution():
    """ Solution of solver function.

    It allows to more easily determine why the solver has stopped.
    """
    def __init__(self, x, n_iter, f_seq, x_seq, norm_seq, diff_seq, alpha_seq,
                 max_iter, tol, xtol):
        self.x = x
        self.n_iter = n_iter
        self.f_seq = f_seq
        self.x_seq = x_seq
        self.norm_seq = norm_seq
        self.diff_seq = diff_seq
        self.alpha_seq = alpha_seq
        self.max_iter = max_iter
        self.tol = tol
        self.xtol = xtol

        # Check stopping conditions
        if norm_seq[-1] <= tol:
            self.converged = True
        else:
            self.converged = False

        if n_iter >= max_iter:
            self.max_iter_reached = True
        else:
            self.max_iter_reached = False

        if diff_seq[-1] <= xtol:
            self.no_change_stop = True
        else:
            self.no_change_stop = False


def solver(x0,
           fun,
           fun_jac=None,
           tol=1e-6,
           xtol=1e-12,
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
        f_seq = [f]
        x_seq = [x]
        norm_seq = [norm]
        diff_seq = [diff]
        alpha_seq = [1]

    while (norm > tol and n_iter < max_iter and diff > xtol):
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

        # Update values
        x = x + alpha * dx
        f = fun(x)

        # stopping condition computation
        norm = np.abs(f)
        diff = np.abs(x - x_old)

        if full_return:
            f_seq.append(f)
            x_seq.append(x)
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
        return Solution(x, n_iter, np.array(f_seq), np.array(x_seq),
                        np.array(norm_seq), np.array(diff_seq),
                        np.array(alpha_seq), max_iter, tol, xtol)
    else:
        return x
