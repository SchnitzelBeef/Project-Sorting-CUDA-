#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#define GPU_RUNS 300
#define ELEMENTS_PER_THREAD 10
#define NUM_BITS 4
#define H (1 << NUM_BITS)

#include "host_skel.cuh"
#include "helper.h"
#include "kernels.cuh"

// *Very* beautiful binary printer:
void binaryPrinter(int val, unsigned int decimal_points) {
    for (int i = decimal_points-1; i >= 0; i--) {
        if (val & (1 << i)) {
            printf("1");
        }
        else {
            printf("0");
        }
    }
}

// Modified From assignment 2:
void scanIncAddI32(const uint32_t B     // desired CUDA block size ( <= 1024, multiple of 32)
                 , const size_t   N     // length of the input array
                 , uint32_t* d_in            // device input  of size: N * sizeof(uint32_t)
                 , uint32_t* d_out           // device result of size: N * sizeof(uint32_t)
) {
    uint32_t* d_tmp;
    cudaMalloc((void**)&d_tmp, MAX_BLOCK*sizeof(uint32_t));
    cudaMemset(d_out, 0, N*sizeof(uint32_t));

    scanInc<Add<uint32_t>> ( B, N, d_out, d_in, d_tmp );

    cudaFree(d_tmp);
}

// Modified from assignment 3-4:
/**
 * Input:
 *   inp_d : [height][width]uint32_t
 * Result:
 *   out_d : [width][height]uint32_t
 *   (the transpose of inp_d.)
 */
template<int T>
void callTransposeKer( uint32_t*          inp_d,  
                       uint32_t*          out_d, 
                       const uint32_t height, 
                       const uint32_t width
) {
    // 1. setup block and grid parameters
    int  dimy = (height+T-1) / T; 
    int  dimx = (width +T-1) / T;
    dim3 block(T, T, 1);
    dim3 grid (dimx, dimy, 1);

    //2. execute the kernel
    coalsTransposeKer<T> <<< grid, block >>>(inp_d, out_d, height, width);
}

int main(int argc, char** argv) {
    uint32_t N;

    initHwd();
    
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
    
    uint32_t Q = 1;
    unsigned int B = 16;
    unsigned int numblocks = (N + (Q * B - 1)) / (Q * B);
    printf("Num blocks: %d \n", numblocks);
    unsigned int mask = (1 << NUM_BITS) - 1; // 4 bits = 0xF for radix 16

    uint32_t mem_size = N * sizeof(uint32_t);
    uint32_t hist_size = numblocks * H * sizeof(uint32_t);
    printf("Mem size: %d: ", mem_size);
    printf("Hist size: %d: ", hist_size);

    // allocate host memory for both CPU and GPU
    uint32_t* h_in  = (uint32_t*) malloc(mem_size);
    uint32_t* gpu_res = (uint32_t*) malloc(hist_size);
    
    
    // initialize the memory
    srand(time(NULL));
    printf("Input: (N):\n");
    for(unsigned int i=0; i<N; ++i) {
        h_in[i] = (uint32_t)rand() % N; // values between 0 and N 
        binaryPrinter(h_in[i], NUM_BITS);
        printf(", ");
    }

    // allocate device memory
    uint32_t* d_in;
    uint32_t* d_hist;
    uint32_t* d_hist_buffer;
    cudaMalloc((void**)&d_in,  mem_size);
    cudaMalloc((void**)&d_hist, hist_size);
    cudaMalloc((void**)&d_hist_buffer, hist_size);

    // copy host memory to device
    cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, hist_size);
    
    // a small number of dry runs
    for(int r = 0; r < 1; r++) {
        dim3 block(B, 1, 1), grid(numblocks, 1, 1);
        histogramKer<<< grid, block>>>(d_in, d_hist, mask, Q, N);
    }

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
        histogramKer<<<numblocks, B>>>(d_in, d_hist, mask, Q, N);
        cudaDeviceSynchronize();
        callTransposeKer<32>(d_hist, d_hist_buffer, numblocks, H); //Maybe use other B value here
        cudaDeviceSynchronize();
        scanIncAddI32(B, numblocks * H, d_hist_buffer, d_hist);
        cudaDeviceSynchronize();
        callTransposeKer<32>(d_hist, d_hist_buffer, H, numblocks);
        cudaDeviceSynchronize();
        mask = mask << NUM_BITS;
        // }
    }
        
    // check for errors
    gpuAssert( cudaPeekAtLastError() );

    // copy result from device to host
    cudaMemcpy(gpu_res, d_hist_buffer, hist_size, cudaMemcpyDeviceToHost);
    
    // element-wise compare of CPU and GPU execution
   for (int b = 0; b < numblocks; b++) {
    printf("\nBlock %d histogram:\n", b);
    for (int i = 0; i < H; i++)
        printf("%u ", gpu_res[b * H + i]);
    }

    printf("\nReached the end! ^_^ \n");

    // clean-up memory
    free(h_in);       free(gpu_res); 
    cudaFree(d_in);   cudaFree(d_hist);
    cudaFree(d_hist_buffer);
}



/**
 Pizza:             Pepsi:`
 ---------------------------------------
      __________           
    // ^   .  O \\           ___;_
   ||..   O      ||         /_____\ 
   || O    . ^   ||         |     |
   ||   ^    . O ||         |Pepsi|
   ||.   ^  O    ||         |_____|
    \\__________//          \_____/
 ````````````````````````````````````````
 */