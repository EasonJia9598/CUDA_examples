#include <stdio.h>

// CUDA kernel to print "Hello, World!" from each thread
__global__ void helloWorld() {
    printf("Hello, World! from thread %d\n", threadIdx.x);
}

int main() {
    // Launch the CUDA kernel with 1 block and 256 threads per block
    helloWorld<<<1, 256>>>();

    // Ensure that all the CUDA kernel launches are completed
    cudaDeviceSynchronize();

    return 0;
}
