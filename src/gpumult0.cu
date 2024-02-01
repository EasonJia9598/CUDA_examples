#include "thrust/host_vector.h"
#include "thrust/device_vector.h"
#include "../include/cxtimers.h"
#include <random>


/*
    The kernel is designed to use one thread to calculate one element of the matrix product.
    THe kernel expects to be called with a 2D grid of thread blocks with sufficient threads in
    the x and y dimensions to span all the elements of C. As before x is the column index and 
    y is the row index.

*/
__global__ void gpumult0(float *C, const float *A, const float *B, int Ay, int Ax, int Bx){
    // compute C=A*B for matrices (assumes Ax = By)

    /*
        Her we set tx and ty from the built-in variables to determine which element of C this thread wil calculate.
        These lines effectively replace the loops over i and j used in the host version. We can think of the kernel 
        as effectively calculating all the elements of C in parallel.
    */
    
    int tx = blockIdx.x*blockDim.x + threadIdx.x; // col j
    int ty = blockIdx.y*blockDim.y + threadIdx.y; // row i
    if (ty >= Ay || tx >= Bx) return;

    C[ty*Bx + tx] = 0.0;
    for(int k = 0; k < Ax; k++){
        C[ty*Bx + tx] += A[ty*Ax + k] * B[k*Bx + tx];
    }
   
} // end hostmult0


int main(int argc, char *argv[]){
    int Arow = (argc > 1) ? atoi(argv[1]) : 1024; // 1024
    int Acol = (argc > 2) ? atoi(argv[2]) : Arow; // 1024

    int Brow = Acol;
    int Bcol = (argc > 3) ? atoi(argv[3]) : Brow; // 1024
    int Crow = Arow;
    int Ccol = Bcol;

    uint tilex = (argc > 4) ? atoi(argv[4]) : 32; //tile x
    uint tiley = (argc > 5) ? atoi(argv[5]) : 32; //tile y


    thrust::host_vector<float> A(Arow*Acol);
    thrust::host_vector<float> B(Brow*Bcol);
    thrust::host_vector<float> C(Crow*Ccol);

    thrust::device_vector<float> dev_C(Crow*Ccol);
    thrust::device_vector<float> dev_A(Arow*Acol);
    thrust::device_vector<float> dev_B(Brow*Bcol);

    // initialize A and B with random numbers
    std::default_random_engine gen(12345678);
    std::uniform_real_distribution<float> fran(0.0, 1.0);
    for(int k = 0; k < Arow*Acol; k++) A[k] = fran(gen);
    for(int k = 0; k < Brow*Bcol; k++) B[k] = fran(gen);


    // H2D copy
    dev_A = A;
    dev_B = B;


    dim3 threads = {tilex, tiley, 1};
    dim3 blocks = {(Bcol + threads.x - 1)/threads.x, (Arow + threads.y - 1)/threads.y, 1};

    cx::timer tim;

    gpumult0<<<blocks, threads>>>(dev_C.data().get(), dev_A.data().get(), dev_B.data().get(), Arow, Acol, Bcol);
    cudaDeviceSynchronize();

    double t2 = tim.lap_ms();

    C = dev_C; // D2H copy

    double t1 = tim.lap_ms();
    double flops = 2.0 * (double)Arow * (double)Acol * (double)Bcol;
    double gflops = flops / (t1*1.0e6);
    double gbytes = gflops * 6.0;
    printf("A[%d][%d] * B[%d][%d] = C[%d][%d]: time %.3f ms, %.3f GFLOPS, %.3f GB/s\n", Arow, Acol, Brow, Bcol, Crow, Ccol, t2, gflops, gbytes);
}

// Result 

// (base) eeepc@eeepc-Legion-T7-34IMZ5:~/Documents/CUDA$ ./build/gpumult0 
// A[1024][1024] * B[1024][1024] = C[1024][1024]: time 3.133 ms, 4520.390 GFLOPS, 27122.341 GB/s