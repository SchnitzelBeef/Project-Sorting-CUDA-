#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "host_skel.cuh"
#include "helper.h"

#define GPU_RUNS 1
#define NUM_BITS 4 // number of bits processed per pass
#define H (1 << NUM_BITS) // histogram size or amount of numbers you can make with NUM_BITS bits
#define VERBOSE false

#define Q 4
#define B 8

#include "kernels.cuh"

//void handleArguments(int argc, char** argv, uint32_t& N, uint32_t& Q, uint32_t& B, uint32_t& NUM_BITS, uint32_t& useFile);

void cubRadixSort(uint32_t* d_in, uint32_t* d_out, size_t N, timeval& t_start, timeval& t_end);

// void scanExcAddI32(const size_t N, uint32_t* d_in, uint32_t* d_out);

void scanIncAddI32(const size_t N, uint32_t* d_in, uint32_t* d_out);


template<int T>
void callTransposeKer(uint32_t* inp_d, uint32_t* out_d, const uint32_t height, const uint32_t width);

void getInputFromFile(const char* filename, uint32_t* h_in, const uint32_t N);

void generateRandomInput(uint32_t* h_in, const uint32_t N);

void copyvals(uint32_t* dest, uint32_t* src, const uint32_t N);

void printVerbose(uint32_t* d_hist, uint32_t* d_hist_scan, uint32_t* d_hist_sgm_scan, uint32_t* gpu_res, uint32_t* h_out, uint32_t N, unsigned int numblocks, uint32_t hist_mem_size);

void handleArguments(int argc, char** argv, uint32_t& N, uint32_t& useFile);

void binaryPrinter(int val, unsigned int decimal_points);

int validate(uint32_t* h_out, uint32_t* h_out_ref, const uint32_t N);

int main(int argc, char** argv) {
    // Arg1: N - number of elements (Required)
    // Arg5: flag - use external input file (Optional) Default: 0 (false)
    uint32_t N;
    // Default parameters
    uint32_t useFile = 0;
    
    handleArguments(argc, argv, N, useFile);

    initHwd();
    
    // use the first CUDA device:
    cudaSetDevice(1);
    
    unsigned int numblocks = (N + (Q * B - 1)) / (Q * B);
    uint32_t mem_size = N * sizeof(uint32_t);
    uint32_t hist_mem_size = numblocks * H * sizeof(uint32_t);
    printf("N is: %d\n", N);
    printf("Pred. Q: %d\n", Q);
    printf("Pred. B: %d\n", B);
    printf("Pred. b: %d\n", NUM_BITS);
    printf("====\n");
    printf("Num blocks: ceil(N / Q*B) = %d\n", numblocks);
    printf("H (RADIX): 2 ** b = %d\n", H);
    printf("====\n");
    printf("Memory size: %d\n", mem_size);
    printf("Histogram memory size: %d\n", hist_mem_size);
    printf("====\n");

    // allocate host memory for both CPU and GPU
    uint32_t* h_in  = (uint32_t*) malloc(mem_size);
    uint32_t* h_out = (uint32_t*) malloc(mem_size);
    uint32_t* gpu_res = (uint32_t*) malloc(hist_mem_size); // This can be removed later
    uint32_t* h_out_ref = (uint32_t*) malloc(mem_size);
    // Flag memory
    char* h_inp_flag = (char*)malloc(numblocks * H);
    memset(h_inp_flag, 0, numblocks * H);

    // initialize the memory
    if (useFile) {
        getInputFromFile("input.txt", h_in, N);
    } else {
        generateRandomInput(h_in, N);
    }


    // print input array
    if (VERBOSE) {
        printf("Input Array: \n");
        for (uint32_t i = 0; i < N; i++) {
            printf("%u ", h_in[i]);
        }
        printf("\n");
        for (uint32_t i = 0; i < N; i++) {
            binaryPrinter(h_in[i], NUM_BITS);       
            printf(", ");
        }
        printf("\n");
    }

    // allocate device memory
    uint32_t* d_in;
    uint32_t* d_out;
    uint32_t* d_hist;
    uint32_t* d_hist_scan;
    uint32_t* d_hist_sgm_scan;
    char* d_flags;
    uint32_t* d_tmp; //REMOVE ME later (only used for shifting)
    uint32_t* d_in_ref;
    uint32_t* d_out_ref;
    uint32_t* d_hist_T; // Transposed histogram
    uint32_t*  d_tmp_vals; // temporary values for flag
    char* d_tmp_flag;      // flag temporary values

    
    
    cudaMalloc((void**)&d_in,  mem_size);
    cudaMalloc((void**)&d_out, mem_size);
    cudaMalloc((void**)&d_hist, hist_mem_size);
    cudaMalloc((void**)&d_hist_scan, hist_mem_size);
    cudaMalloc((void**)&d_hist_sgm_scan, hist_mem_size);
    cudaMalloc((void**)&d_flags, numblocks * H * sizeof(char));
    cudaMalloc((void**)&d_tmp, MAX_BLOCK*sizeof(uint32_t));
    cudaMalloc((void**)&d_in_ref,  mem_size);
    cudaMalloc((void**)&d_out_ref, mem_size);
    cudaMalloc((void**)&d_hist_T, hist_mem_size);
    cudaMalloc((void**)&d_tmp_vals, MAX_BLOCK*sizeof(int)); // temporary values for flag
    cudaMalloc((void**)&d_tmp_flag, MAX_BLOCK*sizeof(char)); // flag temporary values

    // copy host memory to device
    cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, mem_size);
    cudaMemset(d_hist, 0, hist_mem_size);
    cudaMemset(d_hist_scan, 0, hist_mem_size);
    cudaMemset(d_hist_sgm_scan, 0, hist_mem_size);
    cudaMemset(d_flags, 0, numblocks * H * sizeof(char));
    cudaMemcpy(d_in_ref, h_in, mem_size, cudaMemcpyHostToDevice);
    cudaMemset(d_out_ref, 0, mem_size);

    //Set flag array once:

    for(uint32_t i = 0; i < numblocks * H; i += H) {
        h_inp_flag[i] = 1;
    }
    printf("\n");
    cudaMemcpy(d_flags, h_inp_flag, numblocks * H * sizeof(char), cudaMemcpyHostToDevice);
    
    const int W = sizeof(int) * 8; 
    const int num_passes = (W + NUM_BITS - 1) / NUM_BITS;
    unsigned int mask;
    uint32_t shift;

    //a small number of dry runs
    printf("==== DRY RUNS ====== \n");
    for(int r = 0; r < 100; r++) {
        mask = (1 << NUM_BITS) - 1; // 4 bits = 0xF for radix 16
        shift = r * NUM_BITS;
        histogramKer<<<numblocks, B>>>(d_in, d_hist, mask, shift, N);
    }

    struct timeval t_start, t_end, t_diff;
    // running Cub radix sort for reference
    uint64_t elapsed_cub = 0.0;
    for (int i = 0; i < GPU_RUNS; i++) {
        cubRadixSort(d_in_ref, d_out_ref, N, t_start, t_end);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed_cub += (t_diff.tv_sec*1e6+t_diff.tv_usec);
    }
    elapsed_cub /= GPU_RUNS;
    // Verify correctness against Cub result
    cudaMemcpy(h_out_ref, d_out_ref, mem_size, cudaMemcpyDeviceToHost);

    // timing the GPU computation
    uint64_t elapsed_cuda = 0.0;
    // We need to process numblocks * H elements in total
    // We have B threads per block
    // Therefore (numblocks * H) / B blocks are needed
    // (Ceil division)
    for (int i = 0; i < GPU_RUNS; i++) {
        cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);
        cudaMemset(d_out, 0, mem_size);
        cudaMemset(d_hist, 0, hist_mem_size);
        cudaMemset(d_hist_scan, 0, hist_mem_size);
        cudaMemset(d_hist_sgm_scan, 0, hist_mem_size);
        mask = (1 << NUM_BITS) - 1; // 4 bits = 0xF for radix 16
        gettimeofday(&t_start, NULL);

        for (int r = 0; r < num_passes; r++) { //num_passes
            shift = r * NUM_BITS;

            histogramKer<<<numblocks, B>>>(d_in, d_hist, mask, shift, N);
        
            sgmScanInc< Add<uint32_t> >( B, numblocks * H, d_hist_sgm_scan, d_flags, d_hist, d_tmp_vals, d_tmp_flag);

            callTransposeKer<32>(d_hist, d_hist_T, numblocks, H);
            scanIncAddI32(numblocks * H, d_hist_T, d_hist_T);
            callTransposeKer<32>(d_hist_T, d_hist_scan, H, numblocks);
            scatterKer<<<numblocks, B>>>(d_in, d_hist_scan, d_hist_sgm_scan, d_out, N, NUM_BITS, shift);
            
            // swap input and output arrays
            uint32_t* temp = d_in;
            d_in = d_out;
            d_out = temp;
            
            if (VERBOSE) {
                printf("\n----- Iteration: %d -----", r+1);
                cudaMemcpy(h_out, d_in, mem_size, cudaMemcpyDeviceToHost);
                printVerbose(d_hist, d_hist_scan, d_hist_sgm_scan, gpu_res, h_out, N, numblocks, hist_mem_size);
                
                // for (int i = 0; i<N; i++) {
                //     printf("%u - %u", h_out[i], h_out_ref[i]);
                //     printf("\n");
                // }
            }
        }
        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed_cuda += (t_diff.tv_sec*1e6+t_diff.tv_usec);
    }
    elapsed_cuda /= GPU_RUNS;
    
    
    // check for errors
    gpuAssert( cudaPeekAtLastError() );
    
    // Copying sorted array back to host
    cudaMemcpy(h_out, d_in, mem_size, cudaMemcpyDeviceToHost);
    
    // Print result (REMOVE ME LATER)
    printf("\nRadix - CUB\n");
    for (int i = 0; i < N; i++) {
        printf("%u - %u", h_out[i], h_out_ref[i]);
        printf("\n");
    }    
    

    bool validated = validate(h_out, h_out_ref, N);    

    if (validated) {
        printf("\nVALIDATED: Result matches CUB result\n");
    } else {
        validated = validate(h_in, h_out_ref, N);
        if (validated) {
            printf("\nVALIDATED: Result matches CUB result (after even number of passes)\n");
        } else {
            printf("\nDID NOT VALIDATE: Result dont match CUB result!\nEXITING!\n");
            return 4;
        }
    }

    printf("====\n");
    printf("CUB Radix Sort Time: %lu microseconds\n", elapsed_cub);
    printf("CUDA Radix Sort Time: %lu microseconds\n", elapsed_cuda);   
    printf("====\n");
    
    // clean-up memory
    free(h_in);
    free(h_out);
    free(gpu_res); 
    free(h_out_ref);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_hist);
    cudaFree(d_hist_scan);
    cudaFree(d_hist_sgm_scan);
    cudaFree(d_tmp);
    cudaFree(d_in_ref);
    cudaFree(d_out_ref);
    cudaFree(d_hist_T);
    
    //Free flag memory:
    free(h_inp_flag);
    cudaFree(d_flags);
    cudaFree(d_tmp_vals);
    cudaFree(d_tmp_flag);
    return 0;
}

int validate(uint32_t* h_out, uint32_t* h_out_ref, const uint32_t N) {
    for (uint32_t i = 0; i < N; i++) {
        if (h_out[i] != h_out_ref[i]) {
            return 0; // Not valid
        }
    }
    return 1; // Valid
}

// Modified from assignment 2:
void scanIncAddI32(const size_t N, uint32_t* d_in, uint32_t* d_out) {
    // B: Desired CUDA block size (<= 1024, multiple of 32)
    // N: Length of the input array
    // d_in: Device input of size: N * sizeof(uint32_t)
    // d_out: device result of size: N * sizeof(uint32_t)
    uint32_t* d_tmp;
    cudaMalloc((void**)&d_tmp, MAX_BLOCK*sizeof(uint32_t));

    scanInc<Add<uint32_t>> ( B, N, d_out, d_in, d_tmp );

    cudaFree(d_tmp);
}


void handleArguments(int argc, char** argv, uint32_t& N, uint32_t& useFile) {
    // Reading the number of elements 
    if (argc < 2) { 
        printf("Missing N (number of elements) Exiting!\n");
        exit(1);
    }
    
    N = (uint32_t)atoi(argv[1]);

    const uint32_t maxN = 500000000;
    if(N > maxN) {
        printf("N is too big; maximal value is %d. Exiting!\n", maxN);
        exit(2);
    }

    if (argc >= 2) {
        useFile = (uint32_t)atoi(argv[5]);
    }
}


template<int T>
void callTransposeKer(uint32_t* inp_d, uint32_t* out_d, const uint32_t height, const uint32_t width) {
    // inp_d : [height][width]uint32_t
    // out_d : [width][height]uint32_t (the transpose of inp_d.)

    // 1. setup block and grid parameters
    int  dimy = (height+T-1) / T; 
    int  dimx = (width +T-1) / T;
    dim3 block(T, T, 1);
    dim3 grid (dimx, dimy, 1);

    //2. execute the kernel
    coalsTransposeKer<T> <<< grid, block >>>(inp_d, out_d, height, width);
}

void getInputFromFile(const char* filename, uint32_t* h_in, const uint32_t N) {
    FILE* f = fopen(filename, "r");
    if (f == NULL) {
        printf("Error opening file %s\n", filename);
        exit(5);
    }
    // read first '[' chararcter
    char ch = fgetc(f);

    for (uint64_t i = 0; i < N; ++i) {
        fscanf(f, "%u", &h_in[i]);
        // Skip 'u32'
        fscanf(f, "%c", &ch); // 'u'
        fscanf(f, "%c", &ch); // '3'
        fscanf(f, "%c", &ch); // '2'
        if (i < N - 1) {
            // read the comma
            fscanf(f, "%c", &ch); // ','
        }
    }
}

void printVerbose(uint32_t* d_hist, uint32_t* d_hist_scan, uint32_t* d_hist_sgm_scan, uint32_t* gpu_res, uint32_t* h_out, uint32_t N, unsigned int numblocks, uint32_t hist_mem_size) {
    cudaMemcpy(gpu_res, d_hist, hist_mem_size, cudaMemcpyDeviceToHost);

    // Print original histogram for debugging
    printf("\n\n-- Original histogram -- ");
    for (int b = 0; b < numblocks; b++) {
        printf("\n");
        for (int i = 0; i < H; i++) {
            printf("%u ", gpu_res[b * H + i]);
        }
    }

    cudaMemcpy(gpu_res, d_hist_scan, hist_mem_size, cudaMemcpyDeviceToHost);
    
    // Print scanned histogram for debugging
    printf("\n\n-- Scanned (transposed) histogram -- ");
    for (int b = 0; b < numblocks; b++) {
        printf("\n");
        for (int i = 0; i < H; i++) {
            printf("%u ", gpu_res[b * H + i]);
        }
    }

    cudaMemcpy(gpu_res, d_hist_sgm_scan, hist_mem_size, cudaMemcpyDeviceToHost);

    // Print histogram individual scan
    printf("\n\n-- Scanned individual histogram -- ");
    for (int b = 0; b < numblocks; b++) {
        printf("\n");
        for (int i = 0; i < H; i++) {
            printf("%u ", gpu_res[b * H + i]);
        }
    }

    // Print result array
    printf("\n\n-- Result -- \n");
    for (int i = 0; i < N; i++) {
        printf("%u ", h_out[i]);
    }
    printf("\n");
}

void generateRandomInput(uint32_t* h_in, const uint32_t N) {
    srand(time(NULL));
    if (VERBOSE) {
        printf("Input:\n");
        binaryPrinter(h_in[0], NUM_BITS);
        printf(", ");
    }
    for(unsigned int i=0; i<N; ++i) {
        // Chaining 4 rands to get 32-bit integer.
        h_in[i] = (rand() & 0xFF)
                | ((rand() & 0xFF) << 8)
                | ((rand() & 0xFF) << 16)
                | ((rand() & 0xFF) << 24); 
        if (VERBOSE) {
            binaryPrinter(h_in[i], NUM_BITS);       
            printf(", ");
        }
    }
}

void copyvals(uint32_t* dest, uint32_t* src, const uint32_t N) {
    for (uint32_t i = 0; i < N; i++) {
        dest[i] = src[i];
    }
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

// Binary printer for debugging:
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