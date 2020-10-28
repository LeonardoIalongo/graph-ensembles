import os
import numpy as np
from numba import jit,prange


@jit(nopython=True)
def iterative_stripe_one_z(z, out_str, in_str, L):
    """
    function computing the next iteration with
    the fixed point method for CSM-I model 
    """
    aux1 = 0.0
    for i in np.arange(out_str.shape[0]):
        ind_out = int(out_str[i,0])
        sect_out = int(out_str[i,1])
        s_out = out_str[i,2]
        for j in np.arange(in_str.shape[0]):
            ind_in = int(in_str[j,0])
            sect_in = int(in_str[j,1])
            s_in = in_str[j,2]
            if (ind_out != ind_in)&(sect_out==sect_in):
                aux2 = s_out*s_in
                aux1 += aux2/(1+z*aux2)
    aux_z = L/aux1
    return aux_z


@jit(nopython=True)
def loglikelihood_stripe_one_z(z, out_str, in_str, L):
    """
    loglikelihood function of CSM-I model (deprecated)
    """
    aux1 = 0.0
    for i in np.arange(out_str.shape[0]):
        ind_out = int(out_str[i,0])
        sect_out = int(out_str[i,1])
        s_out = out_str[i,2]
        for j in np.arange(in_str.shape[0]):
            ind_in = int(in_str[j,0])
            sect_in = int(in_str[j,1])
            s_in = in_str[j,2]
            if (ind_out != ind_in)&(sect_out==sect_in):
                aux2 = z*s_out*s_in
                aux1 += np.log(1+aux2)/np.log(aux2)
    aux_z = L - aux1
    return aux_z


@jit(nopython=True)
def loglikelihood_prime_stripe_one_z(z, out_str, in_str, L):
    """
    first derivative of the loglikelihood function 
    of the CSM-I model
    """
    aux1 = 0.0
    for i in np.arange(out_str.shape[0]):
        ind_out = int(out_str[i,0])
        sect_out = int(out_str[i,1])
        s_out = out_str[i,2]
        for j in np.arange(in_str.shape[0]):
            ind_in = int(in_str[j,0])
            sect_in = int(in_str[j,1])
            s_in = in_str[j,2]
            if (ind_out != ind_in)&(sect_out==sect_in):
                aux2 = z*s_out*s_in
                aux1 += aux2/(1+aux2)
    aux_z = L - aux1
    return aux_z


@jit(nopython=True)
def loglikelihood_hessian_stripe_one_z(z, out_str, in_str):
    """
    second derivative of the loglikelihood function 
    of the CSM-I model
    """
    aux1 = 0.0
    for i in np.arange(out_str.shape[0]):
        ind_out = int(out_str[i,0])
        sect_out = int(out_str[i,1])
        s_out = out_str[i,2]
        for j in np.arange(in_str.shape[0]):
            ind_in = int(in_str[j,0])
            sect_in = int(in_str[j,1])
            s_in = in_str[j,2]
            if (ind_out != ind_in)&(sect_out==sect_in):
                aux2 = s_out*s_in
                aux1 -= aux2 / (1 + z*aux2)**2
    return aux1


@jit(nopython=True)
def iterative_block_one_z(z, out_str, in_str, L):
    """
    function computing the next iteration with
    the fixed point method for CBM-I model 
    """
    aux1=0.0
    for i in np.arange(out_str.shape[0]):
        ind_out = int(out_str[i, 0])
        sect_out = int(out_str[i, 1])
        sect_node_i = int(out_str[i, 2])
        s_out = out_str[i, 3]
        for j in np.arange(in_str.shape[0]):
            ind_in = int(in_str[j, 0])
            sect_in = int(in_str[j, 1])
            sect_node_j = int(in_str[j, 2])
            s_in = in_str[j, 3]
            if (ind_out != ind_in)&(sect_out == sect_node_j)&(sect_in == sect_node_i):
                aux2 = s_out*s_in
                aux1 += aux2/(1+z*aux2)
    aux_z = L/aux1
    return aux_z


@jit(nopython=True)
def loglikelihood_block_one_z(z, out_str, in_str, L):
    """
    loglikelihood function of CBM-I model (deprecated)
    """
    aux1=0.0
    for i in np.arange(out_str.shape[0]):
        ind_out = int(out_str[i, 0])
        sect_out = int(out_str[i, 1])
        sect_node_i = int(out_str[i, 2])
        s_out = out_str[i, 3]
        for j in np.arange(in_str.shape[0]):
            ind_in = int(in_str[j, 0])
            sect_in = int(in_str[j, 1])
            sect_node_j = int(in_str[j, 2])
            s_in = in_str[j, 3]
            if (ind_out != ind_in)&(sect_out == sect_node_j)&(sect_in == sect_node_i):
                aux2 = z*s_out*s_in
                aux1 += np.log(1 + z*s_out*s_in)/np.log(z*s_out*s_in)
    aux_z = L - aux1
    return aux_z


@jit(nopython=True)
def loglikelihood_prime_block_one_z(z, out_str, in_str, L):
    """
    first derivative of the loglikelihood function 
    of the CBM-I model
    """
    aux1=0.0
    for i in np.arange(out_str.shape[0]):
        ind_out = int(out_str[i, 0])
        sect_out = int(out_str[i, 1])
        sect_node_i = int(out_str[i, 2])
        s_out = out_str[i, 3]
        for j in np.arange(in_str.shape[0]):
            ind_in = int(in_str[j, 0])
            sect_in = int(in_str[j, 1])
            sect_node_j = int(in_str[j, 2])
            s_in = in_str[j, 3]
            if (ind_out != ind_in)&(sect_out == sect_node_j)&(sect_in == sect_node_i):
                aux2 = z*s_out*s_in
                aux1 += aux2/(1+aux2)
    aux_z = L - aux1
    return aux_z


@jit(nopython=True)
def loglikelihood_hessian_block_one_z(z, out_str, in_str):
    """
    second derivative of the loglikelihood function 
    of the CBM-I model
    """
    aux1 = 0.0
    for i in np.arange(out_str.shape[0]):
        ind_out = int(out_str[i, 0])
        sect_out = int(out_str[i, 1])
        sect_node_i = int(out_str[i, 2])
        s_out = out_str[i, 3]
        for j in np.arange(in_str.shape[0]):
            ind_in = int(in_str[j, 0])
            sect_in = int(in_str[j, 1])
            sect_node_j = int(in_str[j, 2])
            s_in = in_str[j, 3]
            if (ind_out != ind_in)&(sect_out == sect_node_j)&(sect_in == sect_node_i):
                aux2 = s_out*s_in
                aux1 -= aux2 / (1 + z*aux2)**2
    return aux1


@jit(nopython=True)
def iterative_stripe_mult_z(z, out_strength, in_strength, L):
    """
    function computing the next iteration with
    the fixed point method for CSM-II model 
    """
    aux1 = np.zeros(len(z))
    for i in np.arange(out_strength.shape[0]):
        ind_out = int(out_strength[i,0])
        sect_out = int(out_strength[i,1])
        s_out = out_strength[i,2]
        for j in np.arange(in_strength.shape[0]):
            ind_in = int(in_strength[j,0])
            sect_in = int(in_strength[j,1])
            s_in = in_strength[j,2]
            if (ind_out != ind_in) & (sect_out==sect_in):
                aux2 = s_out*s_in
                aux1[sect_out] += aux2/(1+z[sect_out]*aux2)
    aux_z = L/aux1
    return aux_z


@jit(nopython=True)
def iterative_block_mult_z(z, out_strength, in_strength, L):
    """
    function computing the next iteration with
    the fixed point method for CBM-II model 
    """
    n_sector = int(np.sqrt(max(z.shape)))
    aux1 = np.zeros(shape=(n_sector,n_sector),dtype=np.float)
    z = np.reshape(z,newshape=(n_sector,n_sector))
    for i in np.arange(out_strength.shape[0]):
        ind_out = int(out_strength[i,0])
        sect_out = int(out_strength[i,1])
        s_out = out_strength[i,2]
        for j in np.arange(in_strength.shape[0]):
            ind_in = int(in_strength[j,0])
            sect_in = int(in_strength[j,1])
            s_in = in_strength[j,2]
            if (ind_out != ind_in):
                aux2 = s_out * s_in
                aux1[sect_out,sect_in] += aux2/(1+z[sect_out,sect_in]*aux2)
    aux_z = L/aux1
    return aux_z


def sufficient_decrease_condition(
    f_old, f_new, alpha, grad_f, p, c1=1e-4, c2=0.9
):
    """return boolean indicator if upper wolfe condition are respected."""
    
    if isinstance(p, np.ndarray):
        sup = f_old + c1 * alpha * grad_f @ p.T
    else:
        sup = f_old + c1 * alpha * grad_f * p
    return bool(f_new < sup)

def linsearch_fun(X): #, args):
    """
    Function searching the descent direction
    """
    x = X[0]
    dx = X[1]
    beta = X[2]
    alfa = X[3]
    f = X[4]
    step_fun = X[5]

    eps2 = 1e-2
    alfa0 = (eps2 - 1) * x / dx
    if isinstance(alfa0, np.ndarray):
        # assure that the zetas are positive
        for a in alfa0:
            if a >= 0:
                alfa = min(alfa, a)
            #i = 0
            #s_old = step_fun(x)
            #while (
            #    sufficient_decrease_condition(
            #        s_old, step_fun(x + alfa * dx), alfa, f, dx
            #    )
            #    == False
            #    and i < 50
            #):
            #    alfa *= beta
            #    i += 1
    else:
        # assure that zeta is positive
        if alfa0 > 0:
            alfa = min(alfa, alfa0)
    
    return alfa


def solver(
           x0,
           fun,
           step_fun,
           fun_jac=None,
           tol=1e-6,
           eps=1e-14,
           max_steps=100,
           method="newton",
           full_return=False,
           linsearch=True,
           ):
    """Find roots of eq. f = 0, using newton or fixed-point."""

    # algorithm
    beta = 0.5  # to compute alpha
    n_steps = 0
    x = x0  # initial point

    f = fun(x)
    norm = np.linalg.norm(f)
    diff = 1

    if full_return:
        norm_seq = [norm]
        alfa_seq = [1]

    while (
        norm > tol and n_steps < max_steps and diff > eps
    ):
        x_old = x  # save previous iteration

        if method == "newton":
            H = fun_jac(x)
            dx = - f/H
        elif method == "fixed-point":    
            dx = f - x

        # Linsearch
        if True:
            alfa1 = 1
            X = (x, dx, beta, alfa1, f, step_fun)
            alfa = linsearch_fun(X)
            if full_return:
                alfa_seq.append(alfa)
        else:
            alfa = 1

        x = x + alfa * dx

        f = fun(x)

        # stopping condition computation
        norm = np.linalg.norm(f)
        diff = np.linalg.norm(x - x_old)

        if full_return:
            norm_seq.append(norm)

        # step update
        n_steps += 1

    if full_return:
        return (x, n_steps, np.array(norm_seq), np.array(alfa_seq))
    else:
        return x
    
    
def iterative_fit(z0, model, method, nz_out_str,
                  nz_in_str, L, tol = 10e-6,
                  eps = 10e-14, max_steps=100
                 ):
    """
    Function solving stripe and block fitness models
    using iterative methods.

    Parameters
    ----------
    z0: float or np.ndarray
        starting points of the method
    model: str
        fitness model to solve iteratively
    nz_out_str: np.ndarray
        the nonzero out strength sequence
    nz_in_str: np.ndarray
        the nonzero in strength sequence
    strengths_block: np.ndarray
        total strengths between every couple of groups
    L : float or np.ndarray
        Links constrains to preserve
    tol : float
        tollerance for the exit condition on the norm
    eps : float
        tollerance for the exit condition on difference 
        between two iterations
    max_steps : int or float
        maximum number of iteration
    Returns
    -------
    tuple
        Returns a tuple with the solution, the number
        of iterations and the norm sequence.
    """
    mod_method = '-'
    mod_method = mod_method.join([model, method])

    d_fun = {
        'CSM-I-fixed-point' : lambda x: iterative_stripe_one_z(x, nz_out_str,
                                                         nz_in_str, L),
        'CSM-II-fixed-point' : lambda x: iterative_stripe_mult_z(x, nz_out_str,
                                                           nz_in_str, L),
        'CBM-I-fixed-point' : lambda x: iterative_block_one_z(x, nz_out_str,
                                                        nz_in_str, L),
        'CBM-II-fixed-point': lambda x: iterative_block_mult_z(x, nz_out_str,
                                                         nz_in_str, L),
        'CSM-I-newton': lambda x: loglikelihood_prime_stripe_one_z(x, nz_out_str,
                                                                   nz_in_str, L),
        'CSM-II-newton': None,
        'CBM-I-newton': lambda x: loglikelihood_prime_block_one_z(x, nz_out_str,
                                                                  nz_in_str, L),
        'CBM-II-newton': None,
    }

    d_fun_step = {
        'CSM-I-fixed-point' : lambda x: loglikelihood_stripe_one_z(x, nz_out_str,
                                                             nz_in_str, L),
        'CSM-II-fixed-point' : None,
        'CBM-I-fixed-point' : lambda x: loglikelihood_block_one_z(x, nz_out_str,
                                                            nz_in_str, L),
        'CBM-II-fixed-point': None,
        'CSM-I-newton' : lambda x: loglikelihood_stripe_one_z(x, nz_out_str,
                                                             nz_in_str, L),
        'CSM-II-newton' : None,
        'CBM-I-newton' : lambda x: loglikelihood_block_one_z(x, nz_out_str,
                                                            nz_in_str, L),
        'CBM-II-newton': None,
    }

    d_fun_jac = {
        'CSM-I-fixed-point' : None,
        'CSM-II-fixed-point' : None,
        'CBM-I-fixed-point' : None,
        'CBM-II-fixed-point': None,
        'CSM-I-newton' : lambda x: loglikelihood_hessian_stripe_one_z(x, nz_out_str,
                                                                      nz_in_str),
        'CSM-II-newton' : None,
        'CBM-I-newton' : lambda x: loglikelihood_hessian_block_one_z(x, nz_out_str,
                                                                     nz_in_str),
        'CBM-II-newton': None,
    }

    fun = d_fun[mod_method]
    fun_step = d_fun_step[mod_method]
    fun_jac = d_fun_jac[mod_method]


    z = solver(x0 = z0,
               fun = fun,
               step_fun = fun_step,
               fun_jac = fun_jac,
               tol = tol,
               eps = eps,
               max_steps = max_steps,
               method = method,
               full_return = True,
               linsearch = True,
              )
    return z