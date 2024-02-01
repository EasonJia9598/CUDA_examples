#include <stdio.h>
#include <stdlib.h>
#include "../include/cxtimers.h"

// CUDA header files
#include "cuda_runtime.h"
#include "thrust/device_vector.h"

/* 
device tells the compiler to compile a version of the function that runs on the GPU and can be called by 
kernels and other functions running on the GPU.
__host __ tells the to create a version of the function that runs on the CPU.
inline keyword is part of the standrad C++ language and tells the compiler to insert the function body.
Generate function code embeded in the caller's code. 
In cuda programming, inline is used to avoid function call overhead. And is the default for __device__ functions.
*/

// For the host and device function, we don't need to adjust the code if we only want to run it on the CPU
__host__ __device__ inline float sinsum(float x, int terms)
{
    float x2 = x * x;
    float term = x;
    float sum = term;
    for (int n = 1; n < terms; n++){
        term *= -x2 / ((2 * n) * (2 * n + 1));
        sum += term;
    }
    return sum;
}

/*
    Cuda kernel function
    for the CUDA Kernel function, we use __global__ instead of __device__, this reflects their dual nature.
    Callable by the host but running on the GPU. 
    It cannot access any memory from the host. All kernels must be declared void. And their aruments are restricted to scalar items or pointers
    to previously allocated regions of device memory. All kernel arugments are passed by value.

    Try to avoid pass large C++ objects to kernels. Since they are passing by values. And all values are copied during this process. 
    This coulde be very time expensive.

    Also, any changes made by the kernel will not be relfected back in the host copy after the kernel call. 

    Additionally, any C++ classes or structs passed to a kernel must have __device__ versions of all their member functions.
*/
__global__ void gpu_sin(float *sums, int steps, int terms, float step_size){
    // unique thread ID
    // int step = blockIdx.x * blockDim.x + threadIdx.x;

    // blockIdx.x: Will be set to the rank of the block to which the current thread bleongs and will be in the range [0, blocks - 1]
    // blockDim.x: Number of threads in one block
    // threadIdx.x : Will be set to the rank of the current thread within its thread block and will be in the range [0, threads - 1]
    // step = blockIdx.x * blockDim.x + threadIdx.x is in the range [0, threads * blocks - 1]

    // If we want to use all threads in one block, 
    // then comment this part of code.
    /*
    if (step < steps){
        float x = step_size * step;
        sums[step] = sinsum(x, terms);
    }
    */

    // A good practice to use all threads in one block
    
    // gridDim.x: Number of blocks in the grid
    // As the grid size stride
    for (int step = blockIdx.x * blockDim.x + threadIdx.x; step < steps; step += blockDim.x * gridDim.x){
        float x = step_size * step;
        sums[step] = sinsum(x, terms);
    }

}

int main(int argc, char *argv[]){
    // get command line arguments
    int steps = (argc > 1) ? atoi(argv[1]) : 10000000;
    int terms = (argc > 2) ? atoi(argv[2]) : 1000;
    int threads = 256;
    int blocks = (steps + threads - 1) / threads;

    double pi = 3.14159265358979323846;
    double step_size = pi / (steps - 1);    
    
    // allocate GPU buffer and get pointer
    thrust::device_vector<float> dsums(steps);
    float *dptr = thrust::raw_pointer_cast(&dsums[0]);
    cx::timer time;

    /************************************************************************************************/

    // Define the number of blocks and threads per block 
    // And call the kernel function
    /*
        Blocks are usually defined by 4 * Nsm, sm as the number of streaming multiprocessors on the GPU.
        Threads per block are usually defined by 128 or 256.
        The number of blocks and threads per block are not independent.

        For the debuging or testing usage, using <<<1, 1>>> to run the kernel function 
        on one thread block with one thread, is usually a good practice.
    */

    
    gpu_sin<<<blocks, threads>>>(dptr, steps, terms, (float)step_size);
    
    /*
        Kernel Call Syntax
        THe general form of a cal to a CUDA kernel uses up to four special arugmetns in the <<< >>>> brackets and the kernel itself
        can have a number of function arguments. THe four arguments in the <<< >>> brackets are:
        
        1. Defines the dimensions of the frid of thread blocks used by the kernel. Either an interger (or unsigned integer) for linear 
        block addressing or a dim3 type defining a 2D or 3D grid of thread blocks. 

        2. Defines the number of threads in a single thread block. Either an iteger (or unsigned integer) for linear thread-linear addresing 
        within a block or a dim3 type to define 2D or 3D array structure for hte threads within a thread block.

        3. An optional argument of type size_t (or int) defining the number of bytes of dynamically allocated shared memory used by each thread block of the kernel.
        No shared memory is reserved if this argument is omitted or set to zero. Note that as an alternative the kernel itself can declare static shared memory.
        The size of a static shared memory must be known at compile time but the szie of dynamically allocated shared memory can be determined at run time.

        4. An optional argument of type cudaStream_t specifying the CUDA stream in which to run the kernel. This option is only needed in advanced applications running 
        multiple simultaneous kernels.
    */



    /************************************************************************************************/

    /*
        thrust::reduce is a parallel reduction algorithm. It takes a range of values and combines them into a single value.
        The first two arguments are the begin and end of the range. The third argument is the initial value of the reduction.
        The fourth argument is a binary function that combines two values. The default is addition.

        Here, we use the host callable reduce function in the thrust library to sum all the elements of the array dsums in GPU memory.
        This call invovles two steps, firstly we perform the required additions on the GPU and secondly we copy the result from GPU memory to CPU memory.
        This is often referred to as as D2H (device to host) transfer.


    */
    double gpu_sum = thrust::reduce(dsums.begin(), dsums.end());



    double gpu_time = time.lap_ms(); // get eplapsed time 

    // Trapezoidal Rule Correction
    gpu_sum -= 0.5 * (sinsum(0.0f, terms) + sinsum(pi, terms));
    gpu_sum *= step_size;
    printf("GPU SUM %.10f, steps %d terms %d time %.3f ms\n", gpu_sum, steps, terms, gpu_time);


}










