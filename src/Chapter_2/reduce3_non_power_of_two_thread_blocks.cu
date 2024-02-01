#include "../include/cx.h"
#include "../include/cxtimers.h"
#include <random> 


/*
    Kernel that uses shared memory to reduce the number of global memory accesses.
    This kernel uses y as an output array and x as the input array with N elements.
    The previous reduce1 kernel used x for both input and output.
*/
__global__ void reduce2(float *y, float *x, int N){

    extern __shared__ float tsum [];
    int id = threadIdx.x;
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    int stride = gridDim.x *blockDim.x;

    tsum[id] = 0.0f;

    for (int k = tid; k < N ; k+= stride){
        tsum[id] += x[k];
    }

    __syncthreads();

    // Here we add a new variable block2 which is set the value of blockDim.x rounded up to
    // the lowest power of 2 greater than or equal to blockDim.x. We use the cs utility funciton pow2ceil
    // for this. That function is implemented using the NVIDIA itrinsic function __clz (int n) which returns
    // the number of the most significant non-zero bit in n. This is a device-only function.
    int block2 = cx::pow2ceil(blockDim.x);
    
    // power of 2 reduction loop
    // THis is the same as previous reduce2 with an added out-of-range check on id+k.

    for (int k = block2/2; k > 0; k >>=1 ){
        if(id < k && id+k < blockDim.x){
            tsum[id] += tsum[id+k];
            __syncthreads();
        }
    }

    //store one value per block
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
// (base) eeepc@eeepc-Legion-T7-34IMZ5:~/Documents/CUDA$ ./build/reduce3_non_power_of_two_thread_blocks 
// sum of 16777216 numbers: host sum 8389645.1 time 261.246 ms, GPU sum 8389645.0 time 0.235 ms