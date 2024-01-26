#include <stdio.h>
#include <thrust/sequence.h>
#include "../include/cx.h"

__global__ void kernel(int* array, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;


    // blockDim.x * gridDim.x is the total number of threads in the grid
    // If we want to calculate a number which is larger than the total number of threads we are using,
    // Then we need to use the while loop to increment the tid by the total number of threads in the grid.
    // That is we use m threads to calculate n numbers, then we need to run n/m passes.

    for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
        // Thread-linear addressing: each thread processes consecutive elements
        printf("gridDim.x = %d blockIdx.x = %d, blockDim.x = %d, threadIdx.x = %d, tid = %d \n", gridDim.x, blockIdx.x, blockDim.x, threadIdx.x, tid);
        printf("tid = %d i is %d, incremental size is %d \n", tid, i, blockDim.x * gridDim.x);
        array[i] = array[i] * 2;
    }
}

int main() {
    const int N = 16;
    thrust::host_vector<int> x(N);
    thrust::device_vector<int> dev_x(N);
    thrust::sequence(x.begin(), x.end(), 1);

    dev_x = x;

    // Launch the CUDA kernel with 2 blocks, each with 4 threads

    /*
        dev_x: This is an instance of thrust::device_vector. 
        It's a container provided by the Thrust library that manages an array of elements on the device.

        dev_x.data(): The data() function is used to get a pointer to the underlying data on the device. 
        This pointer is of type thrust::device_ptr, which is a wrapper around a raw pointer and provides some additional functionality.

        dev_x.data().get(): The get() function is used to obtain the raw pointer to the underlying data. 
        This is a regular C++ raw pointer (int* in the case of a vector of integers), and you can use it as such.
    */

    kernel<<<2, 4>>>(dev_x.data().get(), N);

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    x = dev_x;
    // Print the modified array
    for (int i = 0; i < N; ++i) {
        printf("%d ", x[i]);
    }
    printf("\n");
    return 0;
}


//Result 

/*
gridDim.x = 2 blockIdx.x = 1, blockDim.x = 4, threadIdx.x = 0, tid = 4 
gridDim.x = 2 blockIdx.x = 1, blockDim.x = 4, threadIdx.x = 1, tid = 5 
gridDim.x = 2 blockIdx.x = 1, blockDim.x = 4, threadIdx.x = 2, tid = 6 
gridDim.x = 2 blockIdx.x = 1, blockDim.x = 4, threadIdx.x = 3, tid = 7 
gridDim.x = 2 blockIdx.x = 0, blockDim.x = 4, threadIdx.x = 0, tid = 0 
gridDim.x = 2 blockIdx.x = 0, blockDim.x = 4, threadIdx.x = 1, tid = 1 
gridDim.x = 2 blockIdx.x = 0, blockDim.x = 4, threadIdx.x = 2, tid = 2 
gridDim.x = 2 blockIdx.x = 0, blockDim.x = 4, threadIdx.x = 3, tid = 3 
tid = 4 i is 4, incremental size is 8 
tid = 5 i is 5, incremental size is 8 
tid = 6 i is 6, incremental size is 8 
tid = 7 i is 7, incremental size is 8 
tid = 0 i is 0, incremental size is 8 
tid = 1 i is 1, incremental size is 8 
tid = 2 i is 2, incremental size is 8 
tid = 3 i is 3, incremental size is 8 
gridDim.x = 2 blockIdx.x = 1, blockDim.x = 4, threadIdx.x = 0, tid = 4 
gridDim.x = 2 blockIdx.x = 1, blockDim.x = 4, threadIdx.x = 1, tid = 5 
gridDim.x = 2 blockIdx.x = 1, blockDim.x = 4, threadIdx.x = 2, tid = 6 
gridDim.x = 2 blockIdx.x = 1, blockDim.x = 4, threadIdx.x = 3, tid = 7 
gridDim.x = 2 blockIdx.x = 0, blockDim.x = 4, threadIdx.x = 0, tid = 0 
gridDim.x = 2 blockIdx.x = 0, blockDim.x = 4, threadIdx.x = 1, tid = 1 
gridDim.x = 2 blockIdx.x = 0, blockDim.x = 4, threadIdx.x = 2, tid = 2 
gridDim.x = 2 blockIdx.x = 0, blockDim.x = 4, threadIdx.x = 3, tid = 3 
tid = 4 i is 12, incremental size is 8 
tid = 5 i is 13, incremental size is 8 
tid = 6 i is 14, incremental size is 8 
tid = 7 i is 15, incremental size is 8 
tid = 0 i is 8, incremental size is 8 
tid = 1 i is 9, incremental size is 8 
tid = 2 i is 10, incremental size is 8 
tid = 3 i is 11, incremental size is 8 
2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 
*/