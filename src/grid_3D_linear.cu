#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h> 

__device__ int a[256][512][512]; // file scope
__device__ float b[256][512][512]; // file scope


__global__ void grid3D_linear(int nx, int ny, int nz, int id)
{

    // The variable tid is set to the current threads' rank in the grid of threads blocks using the standard formula for 1D thread and grid-blocks. 
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int array_size = nx * ny * nz;
    int total_threads = gridDim.x * blockDim.x;
    int tid_start = tid;
    int pass = 0;

    while (tid < array_size){ // linear td => (x, y, z)
        int x = tid%nx;  // tid modulo nx
        int y = (tid/nx)%ny; // tid divide nx, then modulo ny
        int z = tid/(nx * ny); // tid / (x - y slice size)

        /*
            This can be replaced with bit shifting operations
            int x = tid & 0x1ff; // x = bits 0-8
            int y = (tid >> 9) & 0x1ff; // y = bits 9-17
            int z = tid >> 18; // z = bits 18 and above

            Relation Between Linear and 3D Indices
            index = (z * ny + y) * nx + x; // index is the linear index

            x = index % nx;
            y = (index / nx) % ny;
            z = index / (nx * ny);
            
            Here we are using tid instead of thread_rank_in_grid
        */



        // do some work here
        a[z][y][x] = tid;
        b[z][y][x] = sqrtf((float)a[z][y][x]);
        
        if (tid == id){
            printf("array size %3d %3d %3d = %d\n", nx, ny, nz, array_size);
            printf("thread block %3d\n", blockDim.x);
            printf("thread Grid %3d\n", gridDim.x);
            printf("total threads %d\n", total_threads);
            printf("a[%d][%d][%d] = %d\n", z, y, x, a[z][y][x]);
            printf("b[%d][%d][%d] = %f\n", z, y, x, b[z][y][x]);
            printf("rank_in_block = %d, rank_in_grid = %d, pass %d tid offset %d \n", threadIdx.x, tid_start, pass, tid - tid_start);
        }
        // Here we increment the thread rank by the total number of threads in the grid.
        tid += gridDim.x * blockDim.x;

        // Here we increment a counter pass and continue to the next pass of the while loop. The variable pass
        // is onlye used as part of the information printed. The actual linear address being used by a given tid
        // within the while loop is rank_in_grid + pass * total_threads
        pass++; 
    }// end while
} // end funciton grid3D_linear


int main(int argc, char *argv[]){
    int id = (argc > 1) ? atoi(argv[1]) : 12345;
    int blocks = (argc > 2) ? atoi(argv[2]) : 288;
    int threads = (argc > 3) ? atoi(argv[3]) : 256;

    grid3D_linear<<<blocks, threads>>>(512, 512, 256, id);
    
    // ensure this function will be executed before it returns
    cudaDeviceSynchronize();
    return 0;
}

//Result 

/*
D:\ > grid3d_linear.exe 1234567 288 256
array size
512 x 512 x 256 = 67108864
thread block 256
thread grid 288
total number of threads in grid 73728
a[4][363][135] = 1234567 and b[4][363][135] = 1111.110718
rank_in_block = 135 rank_in_grid = 54919 rank of
block_rank_in_grid = 214 pass 16
Results from example 2.4 using the grid3D_linear kernel to process 3D arrays with thread-linear-
addressing. The displayed array element has different 3D indices as compared to example 2.2 even
though its linear index is the same as used in that example.

*/