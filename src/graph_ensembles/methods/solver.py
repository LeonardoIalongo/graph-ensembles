import numpy as np
from scipy.optimize import minimize
import scipy


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
        x_l = 0
        f_l = -np.infty
    else:
        x_u = +np.infty
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
    f, f_p = fun(x)
    norm = np.abs(f)
    diff = np.array([1])

    if verbose > 1:
        print("x0 = {}".format(x))
        print("|f(x0)| = {}".format(norm))

    if full_return:
        f_seq = [f]
        x_seq = [x]
        norm_seq = [norm]
        diff_seq = diff

    while (norm > tol and n_iter < max_iter and diff > xtol):
        # Save previous iteration
        x_old = x

        # Compute update
        if f_p > tol:
            dx = - f/f_p
        else:
            dx = - f/tol  # Note that H is always positive

        # Update values
        x = x + dx
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


def newton_solver_pool(x0, parallel_fun, pool, tol=1e-6, xtol=1e-6, 
                       max_iter=100, full_return=False, verbose=False):
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
    f, f_p = parallel_fun(x, pool)
    norm = np.abs(f)
    diff = np.array([1])

    if verbose > 1:
        print("x0 = {}".format(x))
        print("|f(x0)| = {}".format(norm))

    if full_return:
        f_seq = [f]
        x_seq = [x]
        norm_seq = [norm]
        diff_seq = diff

    while (norm > tol and n_iter < max_iter and diff > xtol):
        # Save previous iteration
        x_old = x

        # Compute update
        if f_p > tol:
            dx = - f/f_p
        else:
            dx = - f/tol  # Note that H is always positive

        # Update values
        x = x + dx
        f, f_p = parallel_fun(x, pool)
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


def stochasticNewton(func, target, x0=0, args=(), kwargs={}, maxiter=50, 
                     patience=20, atol=0.0, rtol=1e-9, nullhypo=0.1, 
                     fprime2=False, verbose=0):
    """ Newton Raphson zero point finding with absolute or statistical convergence.
    
    Converges if function values are close to target value for a given number
    of consequtive iterations or if function values are statistically stable, 
    with a zero slope in a linear fit over given number of iterations
        
    Args:
        func: function that returns tuple with function value and gradient 
            wrt x, and possibly second derivative wrt x0
        target: target value for function. Determine x0 such that func(x) = 
            target
        x0: starting value for parameter
        args: extra arguments for func
        kwargs: extra keyword arguments for func
        maxiter: maximum number of iterations to perform
        patience: number of iterations to determine convergence over, either 
            absolute of statistical
        atol: absolute tolerance for absolute convergence
        rtol: relative tolerance for absolute convergence
        nullhypo: confidence level to establish zero slope of fit at
            Probability that function values come from zero slope stochastic 
            process is at least this
            Bigger is better, but around 0.3 becomes a coin toss. 
        fprime2: Use second derivative of function or not
        verbose: verbosity level
    """
    # Store function values
    funcvalues = []
    # Store x values
    xvalues = []
    
    # Iterate without checking for convergence to fill the funcvalues array
    if verbose >= 1: 
        print(f"\nPerforming {patience} burn in iterations\n")

    for itr in range(patience):
        if verbose >= 1:
            print(f"Iteration {itr+1}")
            print(f"\t x0={x0}")
        xvalues.append(x0)
        f = func(x0, *args, **kwargs)
        if verbose >= 1: 
            print(f"\tFuction output={f}")
        funcvalues.append(f[0])
        newton_step = (f[0] - target) / f[1]
        if np.isinf(newton_step):
            raise RuntimeError(
                f"Infinite step size at function call values {f}")
        elif np.isnan(newton_step):
            raise RuntimeError(f"NaN step size at function call values {f}")
        if verbose >= 1: 
            print(f"\tStep={newton_step}")
        if fprime2:
            adj = newton_step * f[2] / f[1] / 2
            if np.abs(adj) < 1:
                newton_step /= 1.0 - adj
                if verbose >= 1: 
                    print(f"\tAdjusted step={newton_step}")
        x = x0 - newton_step
        x0 = x
        
    # Iterate with checking for convergence on the last 'patience' number of 
    # iterations
    if verbose >= 1: 
        print("\nAttempting convergence\n")
    for itr in range(patience, maxiter):
        if verbose >= 1:
            print(f"Iteration {itr+1}")
            print(f"\t x0={x0}")
        xvalues.append(x0)
        f = func(x0, *args, **kwargs)
        if verbose >= 1: 
            print(f"\tFuction output={f}")
        funcvalues.append(f[0])

        # If the last 'patience' function values are close to target, then we 
        # have absolute convergence
        # The x value returned is in this case the last used x value, but you 
        # can get your own from the xvalues array if you want
        if np.all(np.isclose(funcvalues[-patience:], target, 
                  atol=atol, rtol=rtol)):
            return {"convergence": True,
                    "status": "absolute convergence", 
                    "patience": patience,
                    "x": xvalues[-1],
                    "x values": xvalues, 
                    "target": target, 
                    "iterations": itr+1,
                    "function values": funcvalues}
        
        # If the slope of the last 'patience' function values is 
        # significantly close to zero, then we have statistical convergence
        # The x value returned is in this case the mean over the last 
        # 'patience' iterations
        fitresult = scipy.stats.linregress(
            np.arange(patience), funcvalues[-patience:], 
            alternative="two-sided")
        if verbose >= 2: 
            print(f"\t\tFitresult={fitresult}")
        if fitresult.pvalue > nullhypo:
            return {"convergence": True,
                    "status": "statistical convergence", 
                    "patience": patience,
                    "x": np.mean(xvalues[-patience:]),
                    "x mean": np.mean(xvalues[-patience:]),
                    "x std": np.std(xvalues[-patience:]),
                    "x values": xvalues, 
                    "target": target, 
                    "iterations": itr+1,
                    "function values": funcvalues,
                    "fit result": fitresult}
            
        newton_step = (f[0] - target) / f[1]
        if np.isinf(newton_step):
            raise RuntimeError(
                f"Infinite step size at function call values {f}")
        elif np.isnan(newton_step):
            raise RuntimeError(f"NaN step size at function call values {f}")
        if verbose >= 1: 
            print(f"\tStep={newton_step}")
        if fprime2:
            adj = newton_step * f[2] / f[1] / 2
            if np.abs(adj) < 1:
                newton_step /= 1.0 - adj
                if verbose >= 1: 
                    print(f"\tAdjusted step={newton_step}")
        x = x0 - newton_step
        x0 = x
    return {"convergence": False,
            "status": "max iterations", 
            "patience": patience,
            "x values": xvalues, 
            "target": target, 
            "iterations": maxiter,
            "function values": funcvalues}


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
    f = fun(x)
    diff = np.array([1])

    if verbose > 1:
        print("x0 = {}".format(x))

    if full_return:
        x_seq = [x]
        diff_seq = diff

    while (n_iter < max_iter and diff > xtol):
        # Save previous iteration
        x_old = x

        # Compute update
        dx = f - x

        # Update values
        x = x + dx
        f = fun(x)
        # stopping condition computation
        if x_old != 0:
            diff = np.abs((x - x_old)/x_old)
        else:
            diff = 1

        if full_return:
            x_seq.append(x)
            diff_seq = np.append(diff_seq, diff)

        # step update
        n_iter += 1
        if verbose > 1:
            print("    Iteration {}".format(n_iter))
            print("    dx = {}".format(dx))
            print("    x = {}".format(x))
            print("    diff = {}".format(diff))
            print(' ')

    if verbose > 1:
        print(' ')

    if verbose:
        print('Converged: ', diff <= xtol)
        print('Last relative change in x: ', diff)
        print('Iterations: ', n_iter)

    if full_return:
        return Solution(x, n_iter, max_iter, 'fixed-point',
                        x_seq=np.array(x_seq),
                        diff_seq=np.array(diff_seq),
                        xtol=xtol)
    else:
        return x


def alpha_solver(x0, fun, jac, min_d, jac_min_d, tol=1e-6,
                 max_iter=100, full_return=False, verbose=False):
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
    method: 'SLSQP' or 'trust-constr'
        optimization method
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

    constraints = [{'type': 'eq',
                    'fun': fun,
                    'jac': jac},
                   {'type': 'ineq',
                    'fun': min_d,
                    'jac': jac_min_d}]

    res = minimize(lambda x: (x[1] - 1)**2,
                   x0,
                   method='SLSQP',
                   jac=lambda x: np.array([0, 2*(x[1]-1)],
                                          dtype=np.float64),
                   bounds=[(0, None), (0, None)],
                   constraints=constraints,
                   tol=tol,
                   options={'maxiter': max_iter, 'disp': verbose})

    if full_return:
        sol = Solution(res.x, res.nit, max_iter, 'SLSQP', tol=tol)
        sol.converged = res.success
        sol.status = res.status
        sol.message = res.message
        sol.alpha_seq = res.fun
        return sol
    else:
        return res.x
