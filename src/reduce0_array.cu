
// cx.h is part of our example set and contains all the standard includes and some helpful definitions. 
// It is fully described in APpendix G.
#include "../include/cx.h"
#include "../include/cxtimers.h"

#include <random> 


__global__ void reduce0(float *x, int m){
    /*
        Each thread find its tid as the rank, in the grid and, making the tacit assumption that tid is in the range 0 to m -1,
        adds the appropriate element from the top half of the arrray to the bottom half. 
    */
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    // It does the additions will be excuted in parallel by all the cores on the GPU, potentially delivering one or more operations 
    // for each core on each clock-cycle.
    x[tid] += x[tid + m];

    /*
        It triggers three global memory operations, namely loading both the values stored in x[tid] and x[tid +m] into GPU 
        registers and then storing the sum of these values back into x[tid]. If we could accumulate partial sums in local registers, that 
        would reduce the number of global memory accesses needed for each addition down to one, which offers a speed-up by a potential 
        factor of three.

    */
}


int main(int argc, char *argv[]){

    // Here we set the number of array size
    int N = (argc > 1) ? atoi(argv[1]) : 1 << 24; // 2^24

    // Here we allocate thrust host and device vectors x and dev_x to hold the data
    thrust::host_vector<float> x(N);
    thrust::device_vector<float> dev_x(N);

    // initialize x with random numbers and copy to dx
    // Those 3 lines initialise a C++ random number generator and use it to fill x. The use of generators from <random> 
    // is much preferred over the deprecated rand() function in C.
    std::default_random_engine gen(12345678);
    std::uniform_real_distribution<float> fran(0.0, 1.0);
    for(int k = 0; k < N; k++) x[k] = fran(gen);

    // The contents of x are copied from the host to dev_x on the GPU. The details of the transfer are handled by thrust.
    dev_x = x;
    
    cx::timer tim;
    double host_sum = 0.0; // host reduce
    // Summation or reduction on the host by CPU
    for(int k = 0; k < N;k++) host_sum += x[k];
    double t1 = tim.lap_ms();

    // simple GPU reduce for N = power of 2
    // GPU summation or reduction
    tim.reset();

    /*
        Implement the GPU-based parallel iterations of Algorithm 1. For each pass through the for loop the reduce0
        kernel called in line "reduce0<<<blocks, threads>>>(dev_x.data().get(), m);" causes the top half of the array
        dev_x to be "folded" down to an array of size m by adding the top m elements to the bottom m elements. The last
        pass through the loop has m = 1 and leaves the final sum in dev_x[0]. This value is copied back to the host in 
        "gpu_sum = dev_x[0]". 

    */
    for(int m = N/2; m>0;  m /= 2){
        int threads = std::min(256, m);
        int blocks = std::max(m/256,1);
        // ####################################################################################################################
        // The kernel lanuch parameters blocks and threads are set so that the total number of threads in the grid is exactly m.
        // This code will fail if N is not a pwer of 2 due to rounding down errors at one or more steps in the process.
        // #################################################################################################################### 
        reduce0<<<blocks, threads>>>(dev_x.data().get(), m);
    }

    // ####################################################################################################################
    /*
        The kernel call is asynchronous, so we need to wait for it to complete before we can use the result. This is done by
        calling cudaDeviceSynchronize() which waits for all the kernels to complete. This is a blocking call and will not return
        until all the kernels have completed.
        In CUDA programs a kerel lauch such as that used in the "reduce0<<<blocks, threads>>>(dev_x.data().get(), m);" will not 
        block the host which will proceed to the next line of the host program without wating for the kernel call to finish. In
        this case that means all the kernel calls(23 in all for N = 2^24) will be rapidly queued to run successively on the GPU.
        In principle the host can do other CPU work while these kernels are running on the GPU. In this case, we just want to measure
        the duration of the reduction operation so before making the time measurement we must use a cudaDeviceSynchronize call in line 
        30 which causes the host to wait for all pending GPU operations to complete before coninuting. This kind of synchronisation issue
        often occurs in parallel code. 
    */
    cudaDeviceSynchronize();
    double t2 = tim.lap_ms();

    // Here we copy the final sum in the dev_x[0] back to the host, again using thrust, and print results.
    double gpu_sum = dev_x[0]; // D2H copy
    printf("sum of %d random numbers: host %.1f %.3f ms, GPU %.1f %.3f ms\n", N, host_sum, t1, gpu_sum, t2);

}


//Result:
// (cuda) eeepc@eeepc-Legion-T7-34IMZ5:~/Documents/CUDA$ ./build/reduce0_array 
// sum of 16777216 random numbers: host 8389645.1 252.873 ms, GPU 8389646.0 0.450 ms