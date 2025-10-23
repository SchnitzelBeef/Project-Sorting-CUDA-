#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#include "pbb_kernels.cuh"
#include "helper.h"
#include "kernels.cuh"

#define GPU_RUNS 300
#define ELEMENTS_PER_THREAD 10
#define BITS 4 

// WRONG CODE, COPIED FROM ASSIGNMENT 1
int main(int argc, char** argv) {
    uint32_t N;
    
    { // reading the number of elements 
        if (argc != 2) { 
            printf("Num Args is: %d instead of 1. Exiting!\n", argc); 
            exit(1);
        }
        
        N = (uint32_t)atoi(argv[1]);
        printf("N is: %d\n", N);
        
        const uint32_t maxN = 500000000;
        if(N > maxN) {
            printf("N is too big; maximal value is %d. Exiting!\n", maxN);
            exit(2);
        }
    }
    
    // use the first CUDA device:
    cudaSetDevice(0);
    
    uint32_t H = 1 << 4;
    uint32_t Q = 1;
    unsigned int B = 16;
    unsigned int numblocks = (N + (Q * B - 1)) / (Q * B);
    printf("Num blocks: %d \n", numblocks);
    unsigned int mask = 0xF; // 4 bits for radix 16

    uint32_t mem_size = N * sizeof(uint32_t);
    uint32_t hist_size = numblocks * RADIX * sizeof(uint32_t);
    printf("Mem size: %d: ", mem_size);
    printf("Hist size: %d: ", hist_size);

    // allocate host memory for both CPU and GPU
    uint32_t* h_in  = (uint32_t*) malloc(mem_size);
    uint32_t* gpu_res = (uint32_t*) malloc(hist_size);
    
    
    // initialize the memory
    srand(time(NULL));
    for(unsigned int i=0; i<N; ++i) {
        // h_in[i] = (uint32_t)rand() % 1024; // values between 0 and 1023
        h_in[i] = i; 
    }

    // allocate device memory
    uint32_t* d_in;
    uint32_t* d_hist;
    cudaMalloc((void**)&d_in,  mem_size);
    cudaMalloc((void**)&d_hist, hist_size);

    // copy host memory to device
    cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, hist_size);
    
    // a small number of dry runs
    // for(int r = 0; r < 1; r++) {
    //     dim3 block(B, 1, 1), grid(numblocks, 1, 1);
    //     histogramKer<<< grid, block>>>(d_in, d_hist, mask, Q, N);
    // }

    {


        //The cpu does the following:
        //Holds the outer loop over passes (for pass in [0..num_passes))
        // Calculates mask and shift for each bit group
        // Launches the three GPU kernels per pass (histogram → scan → scatter) 
        // Swaps input/output pointers between passes

        //Allocates global memory buffers on device:
        // d_in, d_out for the arrays being sorted
        // d_histograms (size = numBlocks × H)
        // d_prefixes (prefix sums of histograms)
        // Performs small global memory resets (e.g. cudaMemset)
        // Does NOT touch shared or register memory (that’s only inside kernels)

        // for(int r = 0; r < 1; r++) {
        histogramKer<<<numblocks, B>>>(d_in, d_hist, mask, Q, N, H);
        cudaDeviceSynchronize();
        mask = mask << BITS;
        // }
    }
        
    // check for errors
    gpuAssert( cudaPeekAtLastError() );

    // copy result from device to host
    cudaMemcpy(gpu_res, d_hist, hist_size, cudaMemcpyDeviceToHost);

    // print result
    // for(unsigned int i=0; i<N; ++i) printf("GPU at %d: %.6f\n", i, gpu_res[i]);
    // for(unsigned int i=0; i<N; ++i) printf("CPU at %d: %.6f\n", i, cpu_res[i]);

    // element-wise compare of CPU and GPU execution
   for (int b = 0; b < numblocks; b++) {
    printf("Block %d histogram:\n", b);
    for (int i = 0; i < H; i++)
        printf("%u ", gpu_res[b * H + i]);
    printf("\n");
    }

    printf("Reached the end! ^_^ \n");

    // clean-up memory
    free(h_in);       free(gpu_res); 
    cudaFree(d_in);   cudaFree(d_hist);
}



// Pizza:             Pepsi:`
//---------------------------------------
//    __________           
//  // ^   .  O \\           _____
// ||..   O      ||         /_____\ 
// || O    . ^   ||         |     |
// ||   ^    . O ||         |Pepsi|
// ||.   ^  O    ||         |_____|
//  \\__________//          \_____/
//``````````````````````````````````````