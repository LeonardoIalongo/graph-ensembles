from src.graph_ensembles import methods as mt
import numpy as np
from numba import jit
import time
import scipy.stats
import multiprocessing as mp
from multiprocessing import Process
import queue

n_nodes = 100000
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

st2 = time.time()
@jit(nopython=True)
def fit_f_jac_selfloops_multiprocess(p_f, jac_f, param, fit_out, fit_in, n_edges):
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

    return f - n_edges, jac


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

def listener(q, n, t0):
    '''Listens for messages on the q, to update status of computation.'''
    i = 0
    j = 0
    while 1:
        try:
            m = q.get(block=True, timeout=1)
            if m == 'kill':
                break
            elif m == 'skip':
                i += 2
                j += 2
            elif m == 'done':
                i += 1
        except queue.Empty:
            pass
        print('Saved files {0}/{1}. Elapsed time: {3}. (skipped {2})'.format(
              i, n, j,
              time.strftime('%H:%M:%S', time.gmtime(time.time() - t0))),
              end='\r', flush=True)

    print('Saved files {0}/{1}. Elapsed time: {3}. (skipped {2})'.format(
          i, n, j, time.strftime('%H:%M:%S', time.gmtime(time.time() - t0))))

n_blocks = np.floor(n_nodes/10000)
n_blocks = n_blocks.astype(int)
p_function = mt.p_invariant
jac_function = mt.jac_invariant
newton_func = mt.newton_solver

def jac_fit(delta, begin_xout, end_xout):
            fval, fgrad = fit_f_jac_selfloops_multiprocess(p_function, 
                                jac_function, delta, X_out[begin_xout:end_xout], X_in, target)
            return fval, fgrad


if __name__ =='__main__':
    start=time.time()
    #Generate manager queue and pool
    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(mp.cpu_count())
    jobs = []

    watcher = pool.apply_async(listener, (q, 2*n_blocks, start))

    for i in range(n_blocks):
        #Define blocks of 10000
        begin_xout = i*10000
        end_xout = (i+1)*10000
        jac_fit_delta = lambda x: jac_fit(delta=x, begin_xout=begin_xout, end_xout=end_xout)
        # def jac_func(delta):
        #     fval, fgrad = fit_f_jac_selfloops_multiprocess(p_function, 
        #                         jac_function, delta, X_out[begin_xout:end_xout], X_in, target)
        #     return fval, fgrad
        jobs.append(
        pool.apply_async(newton_func, (0, jac_fit_delta))
        )
    #Add the last block with the remaining calculations to the pool
    last_xout = n_blocks*10000
    jac_fit_delta_final = lambda x: jac_fit(delta=x, begin_xout=last_xout, end_xout=n_nodes)
    # def jac_func_final(delta):
    #         fval, fgrad = fit_f_jac_selfloops_multiprocess(p_function, 
    #                             jac_function, delta, X_out[last_xout:], X_in, target)
    #         return fval, fgrad
    jobs.append(
    pool.apply_async(newton_func, (0,jac_fit_delta_final))
    )
    
    # Collect results from the workers through the pool result queue
    for job in jobs:
        job.get()
    
    # Kill the listener
    q.put('kill')
    pool.close()
    pool.join()