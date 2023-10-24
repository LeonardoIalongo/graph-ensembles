import numpy as np


class Solution():
    """ Solution of solver function.

    It allows to more easily determine why the solver has stopped.
    """
    def __init__(self, x, n_iter, max_iter, method, **kwargs):
        self.x = x
        self.n_iter = n_iter
        self.max_iter = max_iter
        self.method = method

        # Extract keyword arguments
        allowed_arguments = ['f_seq', 'x_seq', 'norm_seq', 'x_seq',
                             'diff_seq', 'tol', 'xtol']
        for name in kwargs:
            if name not in allowed_arguments:
                raise ValueError('Illegal argument passed: ' + name)
            else:
                setattr(self, name, kwargs[name])

        # Check stopping conditions
        if n_iter >= max_iter:
            self.max_iter_reached = True
        else:
            self.max_iter_reached = False

        if self.method == 'newton':
            if hasattr(self, 'tol') and hasattr(self, 'norm_seq'):
                if self.norm_seq[-1] <= self.tol:
                    self.converged = True
                else:
                    self.converged = False
            else:
                raise ValueError('tol or norm_seq not set.')

            if hasattr(self, 'xtol') and hasattr(self, 'diff_seq'):
                if self.diff_seq[-1] <= self.xtol:
                    self.no_change_stop = True
                else:
                    self.no_change_stop = False
            else:
                raise ValueError('xtol or diff_seq not set.')

        elif self.method == 'fixed-point':
            if hasattr(self, 'xtol') and hasattr(self, 'diff_seq'):
                if self.diff_seq[-1] <= self.xtol:
                    self.converged = True
                else:
                    self.converged = False
            else:
                raise ValueError('xtol or diff_seq not set.')


def monotonic_newton_solver(x0, fun, tol=1e-6, xtol=1e-6, max_iter=100, 
                            x_l=0, x_u=np.infty, full_return=False, 
                            verbose=False):
    """Find roots of eq. f(x) = 0, using the newton method.
    This implementation assumes that the function is monotonic in x:
    it identifies a bracket within which the root must be and uses a 
    bisection method if the newton algorithm is trying results outside
    the bracket.

    Parameters
    ----------
    x0: float or np.ndarray
        starting points of the method
    fun: function
        function of which to find roots.
    fun_jac: function
        Jacobian of the function.
    tol: float
        tolerance for the exit condition on the norm
    xtol: float
        tolerance for the exit condition on difference between two iterations
    max_iter: int or float
        maximum number of iteration
    full_return: boolean
        if true return all info on convergence
    verbose: boolean
        if true print debug info while iterating

    Returns
    -------
    Solution
        Returns a Solution object.
    """
    n_iter = 0
    x = x0
    f, f_p = fun(x)
    norm = np.abs(f)
    diff = np.array([1])

    if full_return:
        f_seq = [f]
        x_seq = [x]
        norm_seq = [norm]
        diff_seq = diff

    # Initialize bounds on x
    if norm < tol:
        if full_return:
            return Solution(x, n_iter, max_iter, 'newton',
                            f_seq=np.array(f_seq),
                            x_seq=np.array(x_seq),
                            norm_seq=np.array(norm_seq),
                            diff_seq=np.array(diff_seq),
                            tol=tol,
                            xtol=xtol)
        else:
            return x
    elif f > 0:
        x_u = x
        f_u = f
        x_l = x_l
        f_l = -np.infty
    else:
        x_u = x_u
        f_u = +np.infty
        x_l = x
        f_l = f

    if verbose > 1:
        print("x0 = {}".format(x))
        print("|f(x0)| = {}".format(norm))

    while (norm > tol and n_iter < max_iter and diff > xtol):
        # Save previous iteration
        x_old = x

        # Compute update
        if f_p > tol:
            dx = - f/f_p
        else:
            dx = - f/tol  # Note that H is always positive

        # Check that new x exists within the bound (x_u, x_l)
        # otherwise use bisection method
        if (x + dx >= x_u) or (x + dx <= x_l):
            x = x_u - (f_u * (x_u - x_l))/(f_u - f_l)

        # Update values
        x = x + dx
        f, f_p = fun(x)

        if f > 0:
            x_u = x
            f_u = f
        else:
            x_l = x
            f_l = f
        
        # Compute stopping condition
        norm = np.abs(f)
        if x_old != 0:
            diff = np.abs((x - x_old)/x_old)
        else:
            diff = 1

        if full_return:
            f_seq.append(f)
            x_seq.append(x)
            norm_seq.append(norm)
            diff_seq = np.append(diff_seq, diff)

        # step update
        n_iter += 1
        if verbose > 1:
            print("    Iteration {}".format(n_iter))
            print("    fun = {}".format(f))
            print("    fun_prime = {}".format(f_p))
            print("    dx = {}".format(dx))
            print("    x = {}".format(x))
            print("    |f(x)| = {}".format(norm))
            print("    diff = {}".format(diff))
            print(' ')

    if verbose > 1:
        print(' ')

    if verbose:
        print('Converged: ', norm <= tol)
        print('Final distance from root: ', norm)
        print('Last relative change in x: ', diff)
        print('Iterations: ', n_iter)

    if full_return:
        return Solution(x, n_iter, max_iter, 'newton',
                        f_seq=np.array(f_seq),
                        x_seq=np.array(x_seq),
                        norm_seq=np.array(norm_seq),
                        diff_seq=np.array(diff_seq),
                        tol=tol,
                        xtol=xtol)
    else:
        return x
