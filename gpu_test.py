import numpy as np
from numba import njit
import pyopencl as cl
import time


# Simulation parameters
n = int(2e4)
rtol = 1e-6
loop_ref = False
ref_set = False

# Generate vectors
X = np.random.rand(n).astype(np.float32)
Y = np.random.rand(n).astype(np.float32)

# ############## LOOP ###############
if loop_ref:
    print('Starting loop code:')
    start = time.time()
    ref_res = 0
    for x in X:
        for y in Y:
            ref_res += np.float32(x*y / (1 + x*y))

    print('elapsed time:', time.time() - start, 's')

    ref_set = True


# ############## NUMPY ###############
print('Starting numpy code:')

try:
    start = time.time()
    Z = np.outer(X, Y)
    np_res = np.sum(Z / (1 + Z))
    print('elapsed time:', time.time() - start, 's')

    if loop_ref:
        assert abs(np_res - ref_res) / ref_res < rtol
    else:
        ref_res = np_res
        ref_set = True

except MemoryError:
    print('numpy could not allocate memory.')


# ############## NUMBA ###############
@njit
def fun(x, y):
    return x*y / (1 + x*y)


@njit
def loop(X, Y):
    res = 0
    for x in X:
        for y in Y:
            res += fun(x, y)
    return res


print('Starting numba code:')
start = time.time()
nb_res = loop(X, Y)
print('elapsed time:', time.time() - start, 's')

if ref_set:
    assert abs(np_res - ref_res) / ref_res < rtol
else:
    ref_res = nb_res
    ref_set = True


# ############## OPENCL CPU ###############

platforms = cl.get_platforms()
ctx = cl.Context(
        dev_type=cl.device_type.CPU,
        properties=[(cl.context_properties.PLATFORM, platforms[0])])
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

print('Starting OCL (CPU) code:')
start = time.time()

# Allocate memory
x_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=X)
y_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Y)
Z = np.empty(int(n**2), dtype=np.float32)
z_g = cl.Buffer(ctx, mf.WRITE_ONLY, size=Z.nbytes)

prg = cl.Program(ctx, """
__kernel void outer(
    uint n, __global const float *x_g, 
    __global const float *y_g, __global float *z_g)
{
    int gid = get_global_id(0);
    int row = gid/n;
    int col = gid%n;

    z_g[gid] = (x_g[row] * y_g[col]) / (1 + x_g[row] * y_g[col]);

}
""").build()

prg.outer(queue, Z.shape, None, np.uint32(n), x_g, y_g, z_g)

cl.enqueue_copy(queue, Z, z_g)
ocl_res = np.sum(Z)

print('elapsed time:', time.time() - start, 's')

assert abs(ocl_res - ref_res) / ref_res < rtol

# ############## OPENCL GPU ###############

platforms = cl.get_platforms()
ctx = cl.Context(
        dev_type=cl.device_type.GPU,
        properties=[(cl.context_properties.PLATFORM, platforms[0])])
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

print('Starting OCL (GPU) code:')
start = time.time()

# Allocate memory
x_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=X)
y_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Y)
Z = np.empty(int(n**2), dtype=np.float32)
z_g = cl.Buffer(ctx, mf.WRITE_ONLY, size=Z.nbytes)

prg = cl.Program(ctx, """
__kernel void outer(
    uint n, __global const float *x_g, 
    __global const float *y_g, __global float *z_g)
{
    int gid = get_global_id(0);
    int row = gid/n;
    int col = gid%n;

    z_g[gid] = (x_g[row] * y_g[col]) / (1 + x_g[row] * y_g[col]);

}
""").build()

prg.outer(queue, Z.shape, None, np.uint32(n), x_g, y_g, z_g)

cl.enqueue_copy(queue, Z, z_g)
ocl_res = np.sum(Z)

print('elapsed time:', time.time() - start, 's')

assert abs(ocl_res - ref_res) / ref_res < rtol

# __kernel void sumGPU ( __global const double *input, 
#                          __global double *partialSums,
#                          __local double *localSums)
#  {
#   uint local_id = get_local_id(0);
#   uint group_size = get_local_size(0);

#   // Copy from global to local memory
#   localSums[local_id] = input[get_global_id(0)];

#   // Loop for computing localSums : divide WorkGroup into 2 parts
#   for (uint stride = group_size/2; stride>0; stride /=2)
#      {
#       // Waiting for each 2x2 addition into given workgroup
#       barrier(CLK_LOCAL_MEM_FENCE);

#       // Add elements 2 by 2 between local_id and local_id + stride
#       if (local_id < stride)
#         localSums[local_id] += localSums[local_id + stride];
#      }

#   // Write result into partialSums[nWorkGroups]
#   if (local_id == 0)
#     partialSums[get_group_id(0)] = localSums[0];
#  }                                      
