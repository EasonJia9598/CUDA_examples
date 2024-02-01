#include "thrust/host_vector.h"
#include "../include/cxtimers.h"
#include <random>

int hostmult0(float *C, float *A, float *B, int Ay, int Ax, int Bx){
    // compute C=A8B for matrices (assumes Ax = By)
    for(int i = 0; i <Ay;i++){
        for (int j = 0; j < Bx; j++){
            C[i*Bx +j] = 0.0; // declare 0 for start
            // row.col dot product  
            for(int k = 0; k<Ax;k++){
                C[i*Bx + j] += A[i*Ax + k] * B[k*Bx + j];
                // one change in the row
                // the other change in the column
            }
        }
    }
    return 0;
} // end hostmult0


int main(int argc, char *argv[]){
    int Arow = (argc > 1) ? atoi(argv[1]) : 1024; // 1024
    int Acol = (argc > 2) ? atoi(argv[2]) : Arow; // 1024

    int Brow = Acol;
    int Bcol = (argc > 3) ? atoi(argv[3]) : Brow; // 1024
    int Crow = Arow;
    int Ccol = Bcol;



    thrust::host_vector<float> A(Arow*Acol);
    thrust::host_vector<float> B(Brow*Bcol);
    thrust::host_vector<float> C(Crow*Ccol);


    // initialize A and B with random numbers
    std::default_random_engine gen(12345678);
    std::uniform_real_distribution<float> fran(0.0, 1.0);
    for(int k = 0; k < Arow*Acol; k++) A[k] = fran(gen);
    for(int k = 0; k < Brow*Bcol; k++) B[k] = fran(gen);

    cx::timer tim;
    hostmult0(C.data(), A.data(), B.data(), Arow, Acol, Bcol);
    double t1 = tim.lap_ms();
    double flops = 2.0 * (double)Arow * (double)Acol * (double)Bcol;
    double gflops = flops / (t1*1.0e6);
    double gbytes = gflops * 6.0;
    printf("A[%d][%d] * B[%d][%d] = C[%d][%d]: time %.3f ms, %.3f GFLOPS, %.3f GB/s\n", Arow, Acol, Brow, Bcol, Crow, Ccol, t1, gflops, gbytes);
}

// Result 

// (base) eeepc@eeepc-Legion-T7-34IMZ5:~/Documents/CUDA$ ./build/hostmult0 
// A[1024][1024] * B[1024][1024] = C[1024][1024]: time 4532.459 ms, 0.474 GFLOPS, 2.843 GB/s