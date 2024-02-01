#include "../include/cx.h"
#include "../include/cxtimers.h"
#include <random> 


/*
    Kernel that uses shared memory to reduce the number of global memory accesses.
    This kernel uses y as an output array and x as the input array with N elements.
    The previous reduce1 kernel used x for both input and output.
*/
__global__ void reduce2(float *y, float *x, int N){

    /*
        Here we declare the float array tsum to be a shared memory array of size determined by the host 
        at kernel launch time. Shared memory is on-chip and very fast. Each SM has its own block of shared memory
        which has to be shared by all the active thread blocks on that SM. All threads in any given thread block 
        share tsum and can read or write to any of its elements.
        Inter-block communication is not possible using tsum beacuse each thread block has a separate allocation for its tsum. 
        For this kernel, an array size of blockDim.x is assumend for y and it is up to the host code to ensure that the correct
        amount has been reserved. 
        #####################################################################################
        ######## Incorrectly specified kernel launches could cause hard-to-find bugs ########
        #####################################################################################


    */
    extern __shared__ float tsum[]; // Dynamic Shared Memory

    /*
        To prepare for thread-linear addressing, we set id to the rank of the current thread in
        its thread block, tid to the rank of the current thread in the whole grid and stride to 
        the number of threads in the whole grid.
    */
    int id = threadIdx.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Each threads "owns" one element of tsum, tsum[id] for that part of the calculation.
    // Here we initialize tsum[id] to zero. 
    tsum[id] = 0.0f;

    /*
        Here each thread sums the subset of elements of xcorresponding to x[id + n*stride] for 
        all valid intergers n >= 0. Although there is a large stride between successive elements,
        this is a parallel calculation and adjacent threads will simultaneoursly be reading adjacent
        elements of x so this arrangement is maximall efficient for reading GPU main memory. Note 
        that for large arrays, most of the kernel's execution time is used on this statement and 
        very little calculation is done per memory access.
    */
    for (int k = tid; k < N; k += stride){
        tsum[id] += x[k];
    } // end for
    
    /*
        The next step of the alogrithm requires threads to read elements of tsum that have been 
        updated by different threads in the thread block. Technically this's fine. this is what shared 
        memory is for. Hoever, not all threads in the thread block have updated their elements of tsum.
        Hence, we need to synchronize the threads in the thread block to ensure that all the updates have
        been completed before any thread reads any element of tsum. This is done using the __syncthreads()
        function call. This is a barrier synchronization function that ensures that all threads in the thread
        block have reached the same point in the code before any of them are allowed to proceed.

        ###################################################################################################
        ########## Important: __syncthreads() is only valid for threads in the same thread block ##########
        ###################################################################################################

        This is in contrast to the host function cudaDeviceSynchronize() which ensures that all threads in all
        thread blocks have reached the same point in the code before any of them are allowed to proceed.
        cudaDeviceSynchronize() ensures that all pending CUDA kernels and memory transfers have completed before
        allowing the host to continue.


        ###################################################################################################
        If you want to ensure that all theads in all thread blocks have reached a particular point in a kernel 
        then in most cases your only option is to split the kernel into two separate kernels and use
        cudaDeviceSynchronize() between their lanuches.

    */
    __syncthreads();

    // Implementation of the powe of 2 reduction scheme.
    for (int k = blockDim.x/2; k > 0 ; k /= 2){
        if (id < k) {
            tsum[id] += tsum[id + k];
        } // end if
        __syncthreads();

    } // end for 

    // The final block sum accumulated in tsum[0] is stored in the output array y using blockIdx.x as an index
    if (id == 0) y[blockIdx.x] = tsum[0];

}// end kernel function reduce2

int main(int argc, char *argv[]){
    int N = (argc > 1) ? atoi(argv[1]) : 1 << 24; // 2^24
    int blocks = (argc > 2) ? atoi(argv[2]) : 256; // 256
    int threads = (argc > 3) ? atoi(argv[3]) : 256; // 256
    
    thrust::host_vector<float> x(N);
    thrust::device_vector<float> dev_x(N);
    thrust::device_vector<float> dev_y(blocks);

    // initialize x with random numbers and copy to dx
    std::default_random_engine gen(12345678);
    std::uniform_real_distribution<float> fran(0.0, 1.0);
    for(int k = 0; k < N; k++) x[k] = fran(gen);

    dev_x = x; // H2D copy
    cx::timer tim;
    double host_sum = 0.0; // host reduce
    for(int k = 0; k < N;k++) host_sum += x[k];
    double t1 = tim.lap_ms();

    // simple GPU reduce for any value of N
    tim.reset();

    /*
        
    */
    reduce2<<<blocks, threads, threads*sizeof(float)>>>(dev_y.data().get(), dev_x.data().get(), N);
    reduce2<<<1, blocks, blocks*sizeof(float)>>> (dev_x.data().get(), dev_y.data().get(), blocks);

    cudaDeviceSynchronize();
    double t2 = tim.lap_ms();
    double gpu_sum = dev_x[0]; // D2H copy
    printf("sum of %d numbers: host sum %.1f time %.3f ms, GPU sum %.1f time %.3f ms\n", N, host_sum, t1, gpu_sum, t2);
}

//Result
// (base) eeepc@eeepc-Legion-T7-34IMZ5:~/Documents/CUDA$ ./build/reduce2_with_shared_memory 
// sum of 16777216 numbers: host sum 8389645.1 time 255.686 ms, GPU sum 8389645.0 time 0.233 ms