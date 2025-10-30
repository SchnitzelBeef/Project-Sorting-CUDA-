#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define GPU_RUNS 500
#define NUM_BITS 2 // number of bits processed per pass
#define H (1 << NUM_BITS) // histogram size or amount of numbers you can make with NUM_BITS bits
#define VERBOSE true

#include "host_skel.cuh"
#include "helper.h"
#include "kernels.cuh"

void cubRadixSort(uint32_t* d_in, uint32_t* d_out, size_t N, timeval& t_start, timeval& t_end);

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
    cudaSetDevice(1);
    
    uint32_t Q = 5; // number of elements per thread
    unsigned int B = 32; // number of threads per block
    unsigned int numblocks = (N + (Q * B - 1)) / (Q * B);
    printf("Pred. Q: %d \n", Q);
    printf("Pred. B: %d \n", B);
    printf("Pred. b: %d \n", NUM_BITS);
    printf("Num blocks: ceil(N / QB) = %d \n", numblocks);
    printf("H (RADIX): 2 ** b = %d \n", H);

    uint32_t mem_size = N * sizeof(uint32_t);
    uint32_t hist_mem_size = numblocks * H * sizeof(uint32_t);
    printf("Mem size: %d: ", mem_size);
    printf("Hist size: %d: ", hist_mem_size);

    // allocate host memory for both CPU and GPU
    uint32_t* h_in  = (uint32_t*) malloc(mem_size);
    uint32_t* h_out = (uint32_t*) malloc(mem_size);
    uint32_t* gpu_res = (uint32_t*) malloc(hist_mem_size); // This can be removed later
    uint32_t* h_in_ref = (uint32_t*) malloc(mem_size);
    uint32_t* h_out_ref = (uint32_t*) malloc(mem_size);
    
    
    // initialize the memory
    srand(time(NULL));
    h_in_ref[0] = 1;
    h_in[0] = 1;
    if (VERBOSE) {
        printf("\nInput:\n");
        binaryPrinter(h_in[0], NUM_BITS);
        printf(", ");
    }
    for(unsigned int i=1; i<N; ++i) {
        // Chaining 4 rands to get 32-bit integer.
        h_in[i] = (rand() & 0xFF)
                | ((rand() & 0xFF) << 8)
                | ((rand() & 0xFF) << 16)
                | ((rand() & 0xFF) << 24); 
        h_in_ref[i] = h_in[i];
        if (VERBOSE) {
            binaryPrinter(h_in[i], NUM_BITS);       
            printf(", ");
        }
    }

    // allocate device memory
    uint32_t* d_in;
    uint32_t* d_out;
    uint32_t* d_hist;
    uint32_t* d_hist_scan;
    uint32_t* d_tmp; //REMOVE ME later (only used for shifting)
    uint32_t* d_in_ref;
    uint32_t* d_out_ref;
    cudaMalloc((void**)&d_in,  mem_size);
    cudaMalloc((void**)&d_out, mem_size);
    cudaMalloc((void**)&d_hist, hist_mem_size);
    cudaMalloc((void**)&d_hist_scan, hist_mem_size);
    cudaMalloc((void**)&d_tmp, hist_mem_size);
    cudaMalloc((void**)&d_in_ref,  mem_size);
    cudaMalloc((void**)&d_out_ref, mem_size);
    // Allocating space for transposed histograms
    uint32_t* d_hist_T;
    cudaMalloc((void**)&d_hist_T, hist_mem_size);

    // copy host memory to device
    cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, mem_size);
    cudaMemset(d_hist, 0, hist_mem_size);
    cudaMemset(d_hist_scan, 0, hist_mem_size);
    cudaMemcpy(d_in_ref, h_in_ref, mem_size, cudaMemcpyHostToDevice);
    cudaMemset(d_out_ref, 0, mem_size);

    // running Cub radix sort for reference
    struct timeval t_start, t_end, t_diff;
    uint64_t elapsed_cub = 0.0;
    for (int i = 0; i < GPU_RUNS; i++) {
        cubRadixSort(d_in_ref, d_out_ref, N, t_start, t_end);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed_cub += (t_diff.tv_sec*1e6+t_diff.tv_usec);
    }
    elapsed_cub /= GPU_RUNS;
    
    // a small number of dry runs
    // for(int r = 0; r < 1; r++) {
    //     dim3 block(B, 1, 1), grid(numblocks, 1, 1);
    //     histogramKer<<< grid, block>>>(d_in, d_hist, mask, Q, N);
    // }
    const int W = sizeof(int) * 8; 
    const int num_passes = (W + NUM_BITS - 1) / NUM_BITS;
    unsigned int mask;

    uint64_t elapsed_cuda = 0.0;
    // We need to process numblocks * H elements in total
    // We have B threads per block
    // Therefore (numblocks * H) / B blocks are needed
    // (Ceil division)
    int hist_grid = (numblocks * H + B - 1) / B;
    uint32_t shift;
    for (int i = 0; i < GPU_RUNS; i++) {
        cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);
        cudaMemset(d_out, 0, mem_size);
        cudaMemset(d_hist, 0, hist_mem_size);
        cudaMemset(d_hist_scan, 0, hist_mem_size);
        mask = (1 << NUM_BITS) - 1; // 4 bits = 0xF for radix 16
        gettimeofday(&t_start, NULL);

        for (int r = 0; r < num_passes; r++) {
            //The cpu does the following:
            //Holds the outer loop over passes (for pass in [0..num_passes))
            // Calculates mask and shift for each bit group
            // Launches the three GPU kernels per pass (histogram → scan → scatter) 
            // Swaps input/output pointers between passes

            // We also need to shift the bits accordingly after masking
            shift = r * NUM_BITS;

            //Allocates global memory buffers on device:
            // d_in, d_out for the arrays being sorted
            // d_histograms (size = numBlocks × H)
            // d_prefixes (prefix sums of histograms)
            // Performs small global memory resets (e.g. cudaMemset)
            // Does NOT touch shared or register memory (that’s only inside kernels)
            histogramKer<<<numblocks, B>>>(d_in, d_hist, mask, shift, Q, N);
            
            callTransposeKer<32>(d_hist, d_hist_T, numblocks, H);

            // d_tmp is used as a temporary buffer to make d_hist ready for simulated exclusive scan
            // Should be removed
            shiftKer<<<hist_grid, B>>>(d_hist_T, d_tmp, numblocks * H);
            cudaDeviceSynchronize();
              
            scanIncAddI32(B, numblocks * H, d_tmp, d_hist_T);

            callTransposeKer<32>(d_hist_T, d_hist_scan, H, numblocks);
            
            scatterKer<<<numblocks, B>>>(d_in, d_hist_scan, d_out, Q, N, mask, shift);
            cudaDeviceSynchronize();

            // swap input and output arrays
            uint32_t* temp = d_in;
            d_in = d_out;
            d_out = temp;
        }
        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed_cuda += (t_diff.tv_sec*1e6+t_diff.tv_usec);
    }
    elapsed_cuda /= GPU_RUNS;
    
        
    // check for errors
    gpuAssert( cudaPeekAtLastError() );

    // copy result from device to host
    cudaMemcpy(gpu_res, d_hist, hist_mem_size, cudaMemcpyDeviceToHost);

    // element-wise compare of CPU and GPU execution
    printf("\n\n-- Original histogram (transposed) -- ");
    for (int b = 0; b < H; b++) {
        if (VERBOSE) {
            printf("\n");
            for (int i = 0; i < numblocks; i++)
                printf("%u ", gpu_res[b * numblocks + i]);
        }
    }

    cudaMemcpy(gpu_res, d_hist_scan, hist_mem_size, cudaMemcpyDeviceToHost);
    
    // element-wise compare of CPU and GPU execution
    if (VERBOSE) {
        printf("\n\n-- Scanned histogram -- ");
        for (int b = 0; b < numblocks; b++) {
            printf("\n");
            for (int i = 0; i < H; i++)
                printf("%u ", gpu_res[b * H + i]);
        }
    }

    cudaMemcpy(h_out, d_in, mem_size, cudaMemcpyDeviceToHost);

    // element-wise compare of CPU and GPU execution
    if (VERBOSE) {
        printf("\n\n-- Result -- ");
            for (int i = 0; i < N; i++) 
            printf("%d ", h_out[i]);
    }

    // Verify correctness against Cub result
    cudaMemcpy(h_out_ref, d_out_ref, mem_size, cudaMemcpyDeviceToHost);
    bool validated = true;
    for (int i = 0; i < N; i++) {
        if (h_out[i] != h_out_ref[i]) {
            validated = false;
            break;
        }
    }
    if (validated) {
        printf("\nVALIDATED: Result matches CUB result\n");
    } else {
        printf("\nDID NOT VALIDATE: Result dont match CUB result!\nEXITING!\n");
        return -1;
    }

    printf("\nCUB Radix Sort Time: %lu microseconds\n", elapsed_cub);
    printf("\nCUDA Radix Sort Time: %lu microseconds\n", elapsed_cuda);
    
    // clean-up memory
    free(h_in);
    free(h_out);
    free(gpu_res); 
    free(h_in_ref);
    free(h_out_ref);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_hist);
    cudaFree(d_hist_scan);
    cudaFree(d_tmp);
    cudaFree(d_in_ref);
    cudaFree(d_out_ref);
    cudaFree(d_hist_T);
}

// Taken from CUB library examples
// https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html

void cubRadixSort(uint32_t* d_in, uint32_t* d_out, size_t N, timeval& t_start, timeval& t_end) {
    //temporary storage
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_in, d_out, N);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // sorting operation
    gettimeofday(&t_start, NULL);
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_in, d_out, N);
    cudaDeviceSynchronize();
    gettimeofday(&t_end, NULL);

    cudaFree(d_temp_storage);
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