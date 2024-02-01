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


    /*
        The result of lines of if statements replace the for loop in the previous example.

    */
    if(id < 256 && id+256 < blockDim.x){
        tsum[id] += tsum[id+256];
        __syncthreads();
    }

    
    if(id < 128) {
        tsum[id] += tsum[id+128];
        __syncthreads();
    }

    if (id < 64){
        tsum[id] += tsum[id+64];
        __syncthreads();
    }

    if (id < 32){
        tsum[id] += tsum[id+32];
        __syncthreads();
    }

    // only warp 0 array elements used from here

    /*
        For the device of CC < 7 all threads in the same warp act in strict 
        lockstep so here it is possible to reply on implicit warp synchronisation
        and omit the __syncwarp calls entirely. You will find this is done in early (now deprecated) tutorials.
        Even if you only have access to older devices, we strongly recommend that you always use syncwarp where
        it would be ncessary on newer devices, we strongly recommend that you always use 
        syncwarp where it would be necessary on new devices to maintain code portability.
    */



    if (id < 16){
        tsum[id] += tsum[id+16];
        __syncwarp();
    }

    if(id < 8){
        tsum[id] += tsum[id+8];
        __syncwarp();
    }

    if(id < 4){
        tsum[id] += tsum[id+4];
        __syncwarp();
    }

    if(id < 2){
        tsum[id] += tsum[id+2];
        __syncwarp();
    }

    if(id == 0){
        tsum[0] += tsum[1];
        y[blockIdx.x] = tsum[0];
    }




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
// (base) eeepc@eeepc-Legion-T7-34IMZ5:~/Documents/CUDA$ ./build/reduce4_explicit_loop_unrolling 
// sum of 16777216 numbers: host sum 8389645.1 time 258.819 ms, GPU sum 8389645.0 time 0.230 ms