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
                             'diff_seq', 'alpha_seq', 'tol', 'xtol']
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


def newton_solver(x0, fun, tol=1e-6, xtol=1e-6, max_iter=100,
                  full_return=False, verbose=False):
    """Find roots of eq. f(x) = 0, using the newton method.

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
    alpha = 1
    f, f_p = fun(x)
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
        if f_p > tol:
            dx = - f/f_p
        else:
            dx = - f/tol  # Note that H is always positive

        # Update values
        x = x + alpha * dx
        f, f_p = fun(x)
        # stopping condition computation
        norm = np.abs(f)
        if x_old != 0:
            diff = np.abs((x - x_old)/x_old)
        else:
            diff = 1

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
        return Solution(x, n_iter, max_iter, 'newton',
                        f_seq=np.array(f_seq),
                        x_seq=np.array(x_seq),
                        norm_seq=np.array(norm_seq),
                        diff_seq=np.array(diff_seq),
                        alpha_seq=np.array(alpha_seq),
                        tol=tol,
                        xtol=xtol)
    else:
        return x


def fixed_point_solver(x0, fun, xtol=1e-6, max_iter=100, full_return=False,
                       verbose=False):
    """Find roots of eq. f(x) = 0, using the fixed-point method.

    Parameters
    ----------
    x0: float or np.ndarray
        starting points of the method
    fun: function
        function of which to find roots.
    xtol : float
        tolerance for the exit condition on difference between two iterations
    max_iter : int or float
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
    alpha = 1
    f = fun(x)
    diff = 1

    if verbose:
        print("x0 = {}".format(x))

    if full_return:
        x_seq = [x]
        diff_seq = [diff]
        alpha_seq = [1]

    while (n_iter < max_iter and diff > xtol):
        # Save previous iteration
        x_old = x

        # Compute update
        dx = f - x

        # Update values
        x = x + alpha * dx
        f = fun(x)
        # stopping condition computation
        if x_old != 0:
            diff = np.abs((x - x_old)/x_old)
        else:
            diff = 1

        if full_return:
            x_seq.append(x)
            diff_seq.append(diff)
            alpha_seq.append(alpha)

        # step update
        n_iter += 1
        if verbose:
            print("    Iteration {}".format(n_iter))
            print("    dx = {}".format(dx))
            print("    x = {}".format(x))
            print("    alpha = {}".format(alpha))
            print("    diff = {}".format(diff))
            print(' ')

    if verbose:
        print(' ')

    if full_return:
        return Solution(x, n_iter, max_iter, 'fixed-point',
                        x_seq=np.array(x_seq),
                        diff_seq=np.array(diff_seq),
                        alpha_seq=np.array(alpha_seq),
                        xtol=xtol)
    else:
        return x


def newton_alpha_solver(x0, fun, min_d, tol=1e-6, xtol=1e-6, max_iter=100,
                        full_return=False, verbose=False):
    """Find the optimal z and alpha that satisfy the constraints:
        - Expected number of links == empirical number of links
        - Mininum non-zero expected degree is at least one
        - z > 0 and alpha > 0

    Note alpha will be chosen to be the minimum deviation from one that
    satisfies the constraints.

    Parameters
    ----------
    x0: float or np.ndarray
        starting points of the method
    fun: function (z, a)
        function that returns both the value of the function and the Jacobian
    min_d: function (z, a)
        function for the minimum degree value
    tol: float
        tolerance for the exit condition on the norm
    xtol: float
        relative tol for the exit condition on difference between iterations
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
    a = 1
    alpha = 1
    f, f_x, f_a = fun(x, a)
    d, d_x, d_a = min_d(x, a)
    norm = np.abs(f)
    diff = 1

    # Introduce bounds on x (max x with negative f, min x with positive f)
    x_p = np.inf
    x_n = 0

    if verbose:
        print("x0 = {}".format(x))
        print("|f(x0)| = {}".format(norm))
        print("min_d = {}".format(d))

    if full_return:
        f_seq = [f]
        x_seq = [x]
        a_seq = [a]
        norm_seq = [norm]
        min_d_seq = [d]
        diff_seq = [diff]
        alpha_seq = [1]

    for n_iter in range(max_iter):
        # Save previous iteration
        x_old = x

        # Compute update
        if f_x > tol:
            dx = - f/f_x
        else:
            dx = - f/tol  # Note that H is always positive

        # Update x
        x = x + alpha * dx
        if x <= x_n:
            alpha = alpha / 2
            x = x_n
        elif x >= x_p:
            alpha = alpha / 2
            x = x_p
        else:
            alpha = 1

        # update a
        if f >= - tol and d < 1:
            a = a - d / d_x
        else:
            a = 1

        # Update objective functions
        f, f_z, f_a = fun(x, a)
        d, d_x, d_a = min_d(x, a)

        # Update bounds (how does it work with alpha changing?)
        # if f > 0 and x < x_p:
        #     x_p = x
        # elif f < 0 and x > x_n:
        #     x_n = x

        # stopping condition computation
        norm = np.abs(f)
        if x_old != 0:
            diff = np.abs((x - x_old)/x_old)
        else:
            diff = 1

        if ((norm <= tol or diff > xtol) and d >= 1):
            break

        if full_return:
            f_seq.append(f)
            min_d_seq.append(d)
            x_seq.append(x)
            a_seq.append(a)
            norm_seq.append(norm)
            diff_seq.append(diff)
            alpha_seq.append(alpha)

        if verbose:
            print("    Iteration {}".format(n_iter))
            print("    fun = {}".format(f))
            print("    min_d = {}".format(d))
            print("    fun_prime = {}".format(f_x))
            print("    fun_prime_alpha = {}".format(f_a))
            print("    dx = {}".format(dx))
            print("    x = {}".format(x))
            print("    alpha = {}".format(alpha))
            print("    |f(x)| = {}".format(norm))
            print("    diff = {}".format(diff))
            print(' ')

    if verbose:
        print(' ')

    if full_return:
        return Solution(x, n_iter, max_iter, 'newton',
                        f_seq=np.array(f_seq),
                        min_d_seq=np.array(min_d_seq),
                        x_seq=np.array(x_seq),
                        a_seq=np.array(a_seq),
                        norm_seq=np.array(norm_seq),
                        diff_seq=np.array(diff_seq),
                        alpha_seq=np.array(alpha_seq),
                        tol=tol,
                        xtol=xtol)
    else:
        return x
