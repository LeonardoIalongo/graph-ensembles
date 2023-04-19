from src.graph_ensembles import methods as mt
import numpy as np
from numba import jit
import time
import multiprocessing as mp

n_nodes = 100000
block_size = 10000
X_in  = np.random.exponential(scale=100, size=n_nodes)
X_out = np.random.exponential(scale=200, size=n_nodes) + X_in
density = 0.001
target = density * n_nodes**2

start = time.time()

n_blocks = np.floor(n_nodes/block_size)
n_blocks = n_blocks.astype(int)
p_function = mt.p_invariant
jac_function = mt.jac_invariant
newton_func = mt.newton_solver_pool

def jac_fit(delta, pool, n_blocks):
    jobs = []

    for i in range(n_blocks):
        #Define blocks of block_size
        begin_xout = i*block_size
        end_xout = (i+1)*block_size
        jobs.append(
            pool.apply_async(mt.fit_f_jac_selfloops, 
                             (p_function, jac_function, delta, X_out[begin_xout:end_xout], X_in))
        )

    #Add the last block with the remaining calculations to the pool
    last_xout = n_blocks*block_size
    jobs.append(
        pool.apply_async(mt.fit_f_jac_selfloops, 
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
    #Generate pool
    pool = mp.Pool(mp.cpu_count())
    print(pool)
    print(newton_func(0, lambda x, y: jac_fit(x,y, n_blocks), pool))

    pool.close()
    pool.join()
end = time.time()
elapsed_time = end-start 
print("elapsed time:", elapsed_time)