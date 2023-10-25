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
                             'diff_seq', 'atol', 'xtol']
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
            if hasattr(self, 'atol') and hasattr(self, 'norm_seq'):
                if self.norm_seq[-1] <= self.atol:
                    self.converged = True
                else:
                    self.converged = False
            else:
                raise ValueError('atol or norm_seq not set.')

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


def monotonic_newton_solver(x0, fun, atol=1e-6, xtol=1e-6, max_iter=100, 
                            x_l=-np.infty, x_u=np.infty, f_l=None, f_u=None, 
                            full_return=False, verbose=False):
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
    atol: float
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

    # Check that initial value given is in interval
    if (x > x_u) or (x < x_l):
        raise ValueError('Initial value for solver outside allowed bracket.')

    if full_return:
        f_seq = [f]
        x_seq = [x]
        norm_seq = [norm]
        diff_seq = diff

    # Initialize bounds on x
    if norm < atol:
        if full_return:
            return Solution(x, n_iter, max_iter, 'newton',
                            f_seq=np.array(f_seq),
                            x_seq=np.array(x_seq),
                            norm_seq=np.array(norm_seq),
                            diff_seq=np.array(diff_seq),
                            atol=atol,
                            xtol=xtol)
        else:
            return x

    if (f > 0) & (f_p > 0) & (x <= x_u):
        x_u = x
        f_u = f
    elif (f < 0) & (f_p > 0) & (x >= x_l):
        x_l = x
        f_l = f
    elif (f > 0) & (f_p < 0) & (x >= x_l):
        x_l = x
        f_l = f
    elif (f < 0) & (f_p < 0) & (x <= x_u):
        x_u = x
        f_u = f
    elif f_p == 0:
        pass
    else:
        msg = 'The function does not respect the expected monotonicity.'
        assert False, msg

    if verbose > 1:
        print("x0 = {}".format(x))
        print("|f(x0)| = {}".format(norm))

    while (norm > atol and n_iter < max_iter and diff > xtol):
        # Save previous iteration
        x_old = x

        # Compute update
        if np.abs(f_p) > atol:
            dx = - f/f_p
        elif f_p != 0:
            dx = - f/(np.sign(f_p)*atol) 
        else:
            dx = - f/atol 

        # Check that new x exists within the bound (x_u, x_l)
        # otherwise use secant method
        if (x + dx >= x_u) or (x + dx <= x_l):
            if f_u is None:
                x = np.array([x_u])
            elif f_l is None:
                x = np.array([x_l])
            else:
                x = x_u - (f_u * (x_u - x_l))/(f_u - f_l)
        else:
            x = x + dx
    
        f, f_p = fun(x)

        if (f > 0) & (f_p > 0) & (x <= x_u):
            x_u = x
            f_u = f
        elif (f < 0) & (f_p > 0) & (x >= x_l):
            x_l = x
            f_l = f
        elif (f > 0) & (f_p < 0) & (x >= x_l):
            x_l = x
            f_l = f
        elif (f < 0) & (f_p < 0) & (x <= x_u):
            x_u = x
            f_u = f
        elif f_p == 0:
            if x == x_u:
                f_u = f
            elif x == x_l:
                f_l = f
        else:
            msg = 'The function does not respect the expected monotonicity.'
            assert False, msg
        
        # Compute stopping condition
        norm = np.abs(f)
        if (x_old != 0) and not np.any(np.isinf(x_old)):
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
        print('Converged: ', norm <= atol)
        print('Final distance from root: ', norm)
        print('Last relative change in x: ', diff)
        print('Iterations: ', n_iter)

    if full_return:
        return Solution(x, n_iter, max_iter, 'newton',
                        f_seq=np.array(f_seq),
                        x_seq=np.array(x_seq),
                        norm_seq=np.array(norm_seq),
                        diff_seq=np.array(diff_seq),
                        atol=atol,
                        xtol=xtol)
    else:
        return x
