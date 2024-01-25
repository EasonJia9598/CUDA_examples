
#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/*
    Declare two large 3D arrays which have file scope and so can be used by any of the functions declared later in the same file.
    This is standard C++ but with an extra CUDA feature. __device__ will allocate these arrys in the GPU memory not in the host CPU memory.
    

*/
// In CUDA, they defined the array in z, y, x order from left to right.
__device__ int a[256][512][512]; // file scopre 
__device__ float b[256][512][512]; // file scope

// Kernel Function
__global__ void grid3D(int nx, int ny, int nz, int id){

    /*
        In those 3 lines of code, we calculate the threads' x,y, and z coordinates within its thread block. 
        The launch parameters defined in line near (dim3 thread3d(32,8,2)) set the block dimensions to 32,8, and 2. and the grid
        dimensions to 16, 64, and 128 for x, y, z respectively. 

        That is 
        dim3 thread3d(32,8,2) ;   // 32 * 8 * 2 = 512 threads per block
        dim3 block3d(16,64,128) ; // 16 * 64 * 128 = 131072 blocks per grid
        
        blockIdx.x range from (0, 32), gridDim.x from (0, 16)
        blockIdx.y range from (0, 8), gridDim.y from (0, 64)
        blockIdx.z range from (0, 2), gridDim.z from (0, 128)

        Hence, the desired x has range (32 * 16) = 512
        y has range (8 * 64) = 512
        z has range (2 * 128) = 256
        Note the x range corresponds to one complete wrap of threads; this is a design choice not by chance. 
        Having decide to use an x range of 32 we are restricted to smlaaer ranges for y and z as the product of all three is the thread block size
        which is limited by hardware to a maximum of 1024 threads.
    */
    int x = blockIdx.x * blockDim.x + threadIdx.x; // find the x index
    int y = blockIdx.y * blockDim.y + threadIdx.y; // find the y index
    int z = blockIdx.z * blockDim.z + threadIdx.z; // find the z index

    // this is the range check for not going out of bounds of the arrays. 
    /***********************************************************************/
    // It's a good practice to always include the range checks in kernel code.
    /***********************************************************************/
    if (x > nx || y > ny || z > nz){
        return;
    }

    // the size is their product of dimensions
    int array_size = nx * ny * nz;
    int block_size = blockDim.x * blockDim.y * blockDim.z;
    int grid_size = gridDim.x * gridDim.y * gridDim.z;
    // The total number of threads is the product of the thread block size times the grid size
    int total_threads = block_size * grid_size;


    // The rank of the thread within its 3D thread block is calculated using the standard 3D adressing rank formula
    // rand = (z * dim_y + y)*dim_x + x
    int threads_rank_in_block = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    int block_rank_in_grid = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    int thread_rank_in_grid = block_rank_in_grid * block_size + threads_rank_in_block; 


    a[z][y][x] = thread_rank_in_grid;
    b[z][y][x] = sqrtf((float)a[z][y][x]);

    if(thread_rank_in_grid == id){
        printf("array size %3d %3d %3d = %d\n", nx, ny, nz, array_size);
        printf("thread block %3d %3d %3d = %d\n", blockDim.x, blockDim.y, blockDim.z, block_size);
        printf("thread Grid %3d %3d %3d = %d\n", gridDim.x, gridDim.y, gridDim.z, grid_size);
        printf("Total number of threads in grid = %d\n", total_threads);
        printf("a[%d][%d][%d] = %i and b[%d][%d][%d] %f\n", z, y, x, a[z][y][x], z,y,x, b[z][y][x]);
        printf("For thread with 3D-rank %d the 1D-rank %d , block rank in grid %d\n", thread_rank_in_grid, threads_rank_in_block, block_rank_in_grid);
    }
}


int main(int argc, char *argv[]){
    int id = (argc > 1) ? atoi(argv[1]) : 12345;
    dim3 thread3d(32,8,2) ;   // 32 * 8 * 2 = 512 threads per block
    dim3 block3d(16,64,128) ; // 16 * 64 * 128 = 131072 blocks per grid
    grid3D<<<block3d, thread3d>>>(512, 512, 256, id);

    // Must have this line to print out the message
    // Ensure that all the CUDA kernel launches are completed
    
    cudaDeviceSynchronize();

    return 0;
}


//Result 

/*
Case 1 Last thread in ﬁrst thread block:

D:\ > grid3D.exe 511
array size : 512 x 512 x 256 = 67108864
thread block 32 x 8 x 2 = 512
thread grid 16 x 64 x 128 = 131072
total number of threads in grid 67108864
a[1][7][31] = 511 and b[1][7][31] = 22.605309
rank_in_block = 511 rank_in_grid = 511 rank of block_rank_in_grid = 0

Explanation:
Case id = 511: This is the last thread in the first block which spans the range [0-31, 0-7, 0-1]
and the last point in this range is (31, 7, 1) which is shown correctly as the index [1][7][31] in the array a.




Case 2 Thread 135 in block 2411

D:\ grid3d.exe 1234567
array size 512 x 512 x 256 = 67108864
thread block 32 x 8 x 2 = 512
thread grid 16 x 64 x 128 = 131072
total number of threads in grid 67108864
a[4][180][359] = 1234567 and b[4][180][359] = 1111.110718
rank_in_block = 135 rank_in_grid = 1234567 rank of
block_rank_in_grid = 2411
Results from running grid3D
*/