from src.graph_ensembles import methods as mt
import numpy as np
from numba import jit
import time
import scipy.stats
import multiprocessing as mp
from multiprocessing import Process
import queue

n_nodes = 50000
X_in  = np.random.exponential(scale=100, size=n_nodes)
X_out = np.random.exponential(scale=200, size=n_nodes) + X_in
density = 0.001
target = density * n_nodes**2

# X_cross = np.outer(X_in, X_out)

# def func(delta):
#     fval = np.sum(1 - np.exp(-delta * X_cross)).item()
#     grad = np.sum(X_cross * np.exp(-delta * X_cross)).item()
    
#     return fval, grad

# st = time.time()
# mt.stochasticNewton(func, target)
# elapsed_time = time.time() - st
# print('Stochastic Newton time:', elapsed_time)

start = time.time()
@jit(nopython=True)
def fit_f_jac_selfloops_multiprocess(p_f, jac_f, param, fit_out, fit_in):
    """ Compute the objective function of the newton solver and its
    derivative for a single label of the stripe model.
    """
    jac = 0
    f = 0
    for i in np.arange(len(fit_out)):
        s_out = fit_out[i]
        for j in np.arange(len(fit_in)):
            s_in = fit_in[j]
            f += p_f(param, s_out, s_in)
            jac += jac_f(param, s_out, s_in)

    return f, jac


# def func21(delta):
#     fval, grad = fit_f_jac_selfloops_multiprocess(mt.p_invariant, mt.jac_invariant, delta, X_out[:3333], X_in, target)
#     return fval, grad

# def func22(delta):
#     fval, grad = fit_f_jac_selfloops_multiprocess(mt.p_invariant, mt.jac_invariant, delta, X_out[3334:6667], X_in, target)
#     return fval, grad

# def func23(delta):
#     fval, grad = fit_f_jac_selfloops_multiprocess(mt.p_invariant, mt.jac_invariant, delta, X_out[6667:], X_in, target)
#     return fval, grad

# if __name__ == '__main__':
#     p1 = Process(target=mt.newton_solver, args=(0, func21))
#     p2 = Process(target=mt.newton_solver, args=(0, func22))
#     p3 = Process(target=mt.newton_solver, args=(0, func23))
#     p1.start()
#     p2.start()
#     p3.start()
#     p1.join()
#     p2.join()
#     p3.join()

# # mt.newton_solver(x0=0,fun=func21)
# elapsed_time2 = time.time() - st2
# print('Regular Newton time:', elapsed_time2)

n_blocks = np.floor(n_nodes/10000)
n_blocks = n_blocks.astype(int)
p_function = mt.p_invariant
jac_function = mt.jac_invariant
newton_func = mt.newton_solver_pool

def jac_fit(delta, pool, n_blocks):
    jobs = []

    for i in range(n_blocks):
        #Define blocks of 10000
        begin_xout = i*10000
        end_xout = (i+1)*10000
        jobs.append(
            pool.apply_async(fit_f_jac_selfloops_multiprocess, 
                             (p_function, jac_function, delta, X_out[begin_xout:end_xout], X_in))
        )

    #Add the last block with the remaining calculations to the pool
    last_xout = n_blocks*10000
    jobs.append(
        pool.apply_async(fit_f_jac_selfloops_multiprocess, 
                         (p_function, jac_function, delta, X_out[last_xout:], X_in))
    )
    
    # Collect results from the workers through the pool result queue
    tot_fval = -target
    tot_fgrad = 0
    for job in jobs:
        tmp = job.get()
        tot_fval += tmp[0]
        tot_fgrad += tmp[1]
    
    return tot_fval, tot_fgrad



if __name__ =='__main__':
    start=time.time()
    #Generate manager queue and pool
    pool = mp.Pool(mp.cpu_count())
    print(newton_func(0, lambda x, y: jac_fit(x,y, n_blocks), pool))

    pool.close()
    pool.join()
end = time.time()
elapsed_time = end-start 
print("elapsed time:", elapsed_time)