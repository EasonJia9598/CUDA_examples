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
    
*/
__global__ void gpu_sin(float *sums, int steps, int terms, float step_size){
    // unique thread ID
    int step = blockIdx.x * blockDim.x + threadIdx.x;
    if (step < steps){
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

    gpu_sin<<<blocks, threads>>>(dptr, steps, terms, (float)step_size);

    double gpu_sum = thrust::reduce(dsums.begin(), dsums.end());
    double gpu_time = time.lap_ms(); // get eplapsed time 

    // Trapezoidal Rule Correction
    gpu_sum -= 0.5 * (sinsum(0.0f, terms) + sinsum(pi, terms));
    gpu_sum *= step_size;
    printf("GPU SUM %.10f, steps %d terms %d time %.3f ms\n", gpu_sum, steps, terms, gpu_time);


}










