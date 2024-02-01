
// cx.h is part of our example set and contains all the standard includes and some helpful definitions. 
// It is fully described in APpendix G.
#include "../include/cx.h"
#include "../include/cxtimers.h"

#include <random> 


__global__ void reduce1(float *x, int N){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    float tsum = 0.0f;

    /*
        We use thread-linear addressing to sum all the N values contained in x into lower block*threads elements. Each thread
        accumulates its own partial sum in its copy of the register variable tsum and then stores the finnal resultin x[tid] where
        tid is the thread's unqiue rank in the grid. In this example, we have used a for loop instead of a while clause to keep 
        the code compact.
    */
    for (int k = tid; k < N; k += gridDim.x*blockDim.x) tsum += x[k];

    /*
        Where we change the value of an element of x, requires thought. Not all threads actually run at the same time so using 
        the same array for a kernel's input and output is always potentially dangerous. Can we be sure no thread other than tid needs 
        the original value in x[tid]? If the answer is no, then the kernel would have a race condition and the results would be undefined.
        In the present case, we can be sure because every thread uses a separate disjoint subset of the elements of x. If in doubt you 
        should use different arrays for kernel input and output.
    */
    x[tid] = tsum;
    
}


int main(int argc, char *argv[]){
    int N = (argc > 1) ? atoi(argv[1]) : 1 << 24; // 2^24
    thrust::host_vector<float> x(N);
    thrust::device_vector<float> dev_x(N);
    std::default_random_engine gen(12345678);
    std::uniform_real_distribution<float> fran(0.0, 1.0);
    for(int k = 0; k < N; k++) x[k] = fran(gen);
    dev_x = x;
    
    cx::timer tim;
    double host_sum = 0.0; // host reduce
    for(int k = 0; k < N;k++) host_sum += x[k];
    double t1 = tim.lap_ms();

    // simple GPU reduce for N = power of 2
    // GPU summation or reduction
    tim.reset();

    // ###################################  NEW CHANGE ###################################
    // This can be user defined values
    int threads = 256;
    int blocks = 256;
    reduce1<<<blocks, threads>>>(dev_x.data().get(), N);
    reduce1<<<1, threads>>>(dev_x.data().get(), blocks * threads);
    reduce1<<<1, 1>>>(dev_x.data().get(), threads);
    


    cudaDeviceSynchronize();
    double t2 = tim.lap_ms();

    // Here we copy the final sum in the dev_x[0] back to the host, again using thrust, and print results.
    double gpu_sum = dev_x[0]; // D2H copy
    printf("sum of %d random numbers: host %.1f %.3f ms, GPU %.1f %.3f ms\n", N, host_sum, t1, gpu_sum, t2);

}


//Result:
// (base) eeepc@eeepc-Legion-T7-34IMZ5:~/Documents/CUDA$ ./build/reduce1_array 
// sum of 16777216 random numbers: host 8389645.1 257.668 ms, GPU 8389644.0 0.290 ms