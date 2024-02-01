

// Not Runnable Code

/*
    Here we assume that the reqiured size of each array is the number of threads in the thread block. threadDIm.x for 1D thread grids.


*/
__global__ void shared_example(float *x, float *y, int m){
    // notice order of declarations
    // longest variable type first
    // shortest variable type last
    // NB sx is a pointer to the start of the shared

    /*
        A single dynamically allocated shared memory array sx of type float is declared
    */
    extern __shared__ float sx[];

    ushort* su = (ushort *)(&sx[blockDim.x]);   // start after sx
    char* sc = (char*(&su[blockDim.x]));        // start after su


    int id = threadIdx.x;

    sx[id] = 3.14159*x[id];
    su[id] = id*id;
    sc[id] = id%128;

    //do useful work here
    ... 

    int threads = (argc > 1) ? atoi(argv[1]) : 256; // 256
    int blocks = (size+threads-1)/threads; // round up
    int shared = threads*(sizeof(float) + sizeof(ushort) + sizeof(char));

    shared_example<<<blocks, threads, shared>>>(dx_ptr, dy_prt, size)

    }
}// end kernel function
