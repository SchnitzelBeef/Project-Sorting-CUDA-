#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "host_skel.cuh"
#include "helper.h"

#define VERBOSE false     // printing for debugging
#define GPU_RUNS 500      // number of gpu-runs

       
#define H (1 << NUM_BITS) // histogram size or amount of numbers you can make with NUM_BITS bits

const int Q = RADIX_Q;      // amount of elements each thread processes
const int B = RADIX_B;      // block size
const int NUM_BITS = BITS;   // number of bits processed per pass


#include "kernels.cuh"

//void handleArguments(int argc, char** argv, uint32_t& N, uint32_t& Q, uint32_t& B, uint32_t& NUM_BITS, uint32_t& useFile);

void cubRadixSort(cub::DoubleBuffer<uint32_t>& d_keys, size_t N, timeval& t_start, timeval& t_end);

template<int T>
void callTransposeKer(uint32_t* inp_d, uint32_t* out_d, const uint32_t height, const uint32_t width);

void getInputFromFile(const char* filename, uint32_t* h_in, const uint32_t N);

void generateRandomInput(uint32_t* h_in, const uint32_t N);

void copyvals(uint32_t* dest, uint32_t* src, const uint32_t N);

void printVerbose(uint32_t* d_hist, uint32_t* d_hist_scan, uint32_t* d_hist_sgm_scan, uint32_t* h_gpu_res, uint32_t* h_out, uint32_t N, unsigned int numblocks, uint32_t hist_mem_size);

void handleArguments(int argc, char** argv, uint32_t& N, uint32_t& useFile);

void binaryPrinter(int val, unsigned int decimal_points);

int comp(const void *a, const void *b);

int main(int argc, char** argv) {
    // Arg1: N - number of elements (Required)
    // Arg5: flag - use external input file (Optional) Default: 0 (false)
    uint32_t N;
    // Default parameters
    uint32_t useFile = 0;
    
    handleArguments(argc, argv, N, useFile);

    initHwd();
    
    // use the first CUDA device:
    cudaSetDevice(0);
    
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
    printf("Reading from file: %d\n", useFile);
    printf("====\n");
    printf("Memory size (bytes): %d\n", mem_size);
    printf("Histogram memory size (bytes): %d\n", hist_mem_size);
    printf("====\n");

    // allocate host memory for both CPU and GPU
    uint32_t* h_in  = (uint32_t*) malloc(mem_size);
    uint32_t* h_out = (uint32_t*) malloc(mem_size);
    uint32_t* h_gpu_res = (uint32_t*) malloc(hist_mem_size); // This can be removed later
    uint32_t* h_out_ref = (uint32_t*) malloc(mem_size);

    // initialize the memory
    if (useFile) getInputFromFile("../../input.txt", h_in, N);
    else generateRandomInput(h_in, N);

    // print input array for debugging
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
    uint32_t* d_in;             // input memeory
    uint32_t* d_out;            // output memory
    uint32_t* d_hist;           // histograms per block
    uint32_t* d_hist_scan;      // scanned histogram global 
    uint32_t* d_hist_sgm_scan;  // per histogram scan (also used as buffer for transposed histogram) 
    uint32_t* d_tmp_vals;       // temporary values for flag
    char*     d_tmp_flag;       // flag temporary values
    char*     d_flags;          // flag array
    
    cudaMalloc((void**)&d_in,  mem_size);
    cudaMalloc((void**)&d_out, mem_size);
    cudaMalloc((void**)&d_hist, hist_mem_size);
    cudaMalloc((void**)&d_hist_scan, hist_mem_size);
    cudaMalloc((void**)&d_hist_sgm_scan, hist_mem_size);
    cudaMalloc((void**)&d_flags, numblocks * H * sizeof(char));
    cudaMalloc((void**)&d_tmp_vals, MAX_BLOCK*sizeof(int)); // memory used for scan and sgm scan
    cudaMalloc((void**)&d_tmp_flag, MAX_BLOCK*sizeof(char)); // memory used for flag array in sgm scan

    // copy host memory to device
    cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, mem_size);
    cudaMemset(d_hist, 0, hist_mem_size);
    cudaMemset(d_hist_scan, 0, hist_mem_size);
    cudaMemset(d_hist_sgm_scan, 0, hist_mem_size);
    cudaMemset(d_flags, 0, numblocks * H * sizeof(char));
    
    const int W = sizeof(int) * 8; 
    const int num_passes = (W + NUM_BITS - 1) / NUM_BITS;
    uint32_t shift = 0;

    // dry run to exercise device allocations
    printf("\n==== DRY RUN ====== \n");
    {
        histogramKer<<<numblocks, B>>>(d_in, d_hist, shift, N);
        callTransposeKer<32>(d_hist, d_hist_sgm_scan, numblocks, H);
        scanInc<Add<uint32_t>>(B, numblocks * H, d_hist_sgm_scan, d_hist_sgm_scan, d_tmp_vals);
        callTransposeKer<32>(d_hist_sgm_scan, d_hist_scan, H, numblocks);
        createFlagKer<<<numblocks, B>>>(d_flags, N);
        sgmScanInc<Add<uint32_t>>(B, numblocks * H, d_hist_sgm_scan, d_flags, d_hist, d_tmp_vals, d_tmp_flag);
        scatterKer<<<numblocks, B>>>(d_in, d_hist_scan, d_hist_sgm_scan, d_out, N, NUM_BITS, shift);
    }

    struct timeval t_start, t_end, t_diff;

    // running Cub radix sort for reference (DOES NOT SEEM TO WORK)
    uint64_t elapsed_cub = 0.0;
    cub::DoubleBuffer<uint32_t> d_keys(d_in, d_out);
    // Dry run for CUB
    { 
        cubRadixSort(d_keys, N, t_start, t_end);
    }
    for (int i = 0; i < GPU_RUNS; i++) {
        cubRadixSort(d_keys, N, t_start, t_end);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed_cub += (t_diff.tv_sec * 1e6 + t_diff.tv_usec);
    }
    elapsed_cub /= GPU_RUNS;
    // result of cub is not used (qsort for validation)
    //cudaMemcpy(h_out_ref, d_keys.Current(), N * sizeof(uint32_t), cudaMemcpyDeviceToHost);


    // Temporary qsort just so we can validate
    memcpy(h_out_ref, h_in, N*sizeof(uint32_t));
    qsort(h_out_ref, N, sizeof(uint32_t), comp);

    // timing the GPU computation
    uint64_t elapsed_cuda = 0.0;
    // We need to process numblocks * H elements in total
    // We have B threads per block
    // Therefore (numblocks * H) / B blocks are needed
    // (Ceil division)
    printf("==== %d GPU RUNS ====== \n", GPU_RUNS);
    for (int i = 0; i < GPU_RUNS; i++) {
        cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);
        cudaMemset(d_out, 0, mem_size);
        cudaMemset(d_hist, 0, hist_mem_size);
        cudaMemset(d_hist_scan, 0, hist_mem_size);
        cudaMemset(d_hist_sgm_scan, 0, hist_mem_size);
        gettimeofday(&t_start, NULL);

        for (int r = 0; r < num_passes; r++) {
            shift = r * NUM_BITS;

            // Produce a histogram of size H for each Q*B elements 
            histogramKer<<<numblocks, B>>>(d_in, d_hist, shift, N);

            // Transpose histogram for coalesced access 
            callTransposeKer<32>(d_hist, d_hist_sgm_scan, numblocks, H);

            // Perform inclusive scan of histogram
            scanInc<Add<uint32_t>>(B, numblocks * H, d_hist_sgm_scan, d_hist_sgm_scan, d_tmp_vals);

            // Transpose back
            callTransposeKer<32>(d_hist_sgm_scan, d_hist_scan, H, numblocks);

            // Create flag array for per block histogram scan
            createFlagKer<<<numblocks, B>>>(d_flags, N);

            // Calculate per block hisogram scan
            sgmScanInc<Add<uint32_t>>(B, numblocks * H, d_hist_sgm_scan, d_flags, d_hist, d_tmp_vals, d_tmp_flag);

            // Sort in each block and scatter to final position
            scatterKer<<<numblocks, B>>>(d_in, d_hist_scan, d_hist_sgm_scan, d_out, N, NUM_BITS, shift);
            
            // Swap input and output arrays
            uint32_t* temp = d_in;
            d_in = d_out;
            d_out = temp;
            
            // Print intermediate values for debugging
            if (VERBOSE) {
                printf("\n----- Iteration: %d -----", r+1);
                cudaMemcpy(h_out, d_in, mem_size, cudaMemcpyDeviceToHost);
                printVerbose(d_hist, d_hist_scan, d_hist_sgm_scan, h_gpu_res, h_out, N, numblocks, hist_mem_size);
            }
        }
        cudaDeviceSynchronize();
        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed_cuda += (t_diff.tv_sec*1e6+t_diff.tv_usec);
    }
    elapsed_cuda /= GPU_RUNS;
    
    // check for errors
    gpuAssert( cudaPeekAtLastError() );
    
    // Copying sorted array back to host
    cudaMemcpy(h_out, d_in, mem_size, cudaMemcpyDeviceToHost);
    
    // Print result for debugging
    if (VERBOSE) {
        printf("\n ------- RESULTS ------- \n\nRadix - qsort \n");
        for (int i = 0; i < N; i++) {
            printf("%10u - %10u", h_out[i], h_out_ref[i]);
            printf("\n");
        }    
    }
    
    // Validation
    printf("Against qsort: ");
    if (!validateExact(h_out, h_out_ref, N)) return 4;

    // Timings
    printf("CUB Radix Sort Time (maybe correct): %lu microseconds\n", elapsed_cub);
    printf("CUDA Radix Sort Time: %lu microseconds\n", elapsed_cuda);   
    printf("Speedup: CUDA / CUB = %f times faster\n", (double)elapsed_cuda/elapsed_cub);    
    printf("====\n");
    
    // clean-up memory
    free(h_in);
    free(h_out);
    free(h_gpu_res); 
    free(h_out_ref);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_hist);
    cudaFree(d_hist_scan);
    cudaFree(d_hist_sgm_scan);

    //Free flag memory:
    cudaFree(d_flags);
    cudaFree(d_tmp_vals);
    cudaFree(d_tmp_flag);
    return 0;
}

// TComparator function for qsort in validation
// (overflow/underflow safe)
int comp(const void *a, const void *b) {
    uint32_t arg1 = *(const uint32_t*)a;
    uint32_t arg2 = *(const uint32_t*)b;
    return (arg1 > arg2) - (arg1 < arg2);
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

    if (argc >= 3) {
        useFile = (uint32_t)atoi(argv[2]);
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
    if (!f) {
        printf("Error: Cannot open file %s\n", filename);
        exit(5);
    }

    // Expect first '['
    char ch;
    if (fscanf(f, " %c", &ch) != 1 || ch != '[') {
        printf("Error: Input file does not start with '['\n");
        exit(6);
    }

    uint32_t val;
    uint32_t count = 0;

    while (count < N && fscanf(f, "%u", &val) == 1) {
        h_in[count++] = val;

        fscanf(f, "%*c%*c%*c"); // skip u32

        // skip optional comma
        fscanf(f, " %c", &ch);
        if (ch != ',') {
            // If it's the closing bracket, stop
            if (ch == ']') break;
            // Otherwise put it back
            ungetc(ch, f);
        }
    }

    fclose(f);

    if (count < N) {
        printf("Error: File only contained %u values, expected %u\n", count, N);
        exit(7);
    }

    printf("Loaded %u elements from %s\n", count, filename);
}


void printVerbose(uint32_t* d_hist, uint32_t* d_hist_scan, uint32_t* d_hist_sgm_scan, uint32_t* h_gpu_res, uint32_t* h_out, uint32_t N, unsigned int numblocks, uint32_t hist_mem_size) {
    cudaMemcpy(h_gpu_res, d_hist, hist_mem_size, cudaMemcpyDeviceToHost);

    // Print original histogram for debugging
    printf("\n\n-- Original histogram -- ");
    for (int b = 0; b < numblocks; b++) {
        printf("\n");
        for (int i = 0; i < H; i++) {
            printf("%u ", h_gpu_res[b * H + i]);
        }
    }

    cudaMemcpy(h_gpu_res, d_hist_scan, hist_mem_size, cudaMemcpyDeviceToHost);
    
    // Print scanned histogram for debugging
    printf("\n\n-- Scanned (transposed) histogram -- ");
    for (int b = 0; b < numblocks; b++) {
        printf("\n");
        for (int i = 0; i < H; i++) {
            printf("%u ", h_gpu_res[b * H + i]);
        }
    }

    cudaMemcpy(h_gpu_res, d_hist_sgm_scan, hist_mem_size, cudaMemcpyDeviceToHost);

    // Print histogram individual scan
    printf("\n\n-- Scanned individual histogram -- ");
    for (int b = 0; b < numblocks; b++) {
        printf("\n");
        for (int i = 0; i < H; i++) {
            printf("%u ", h_gpu_res[b * H + i]);
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
        // printf("Input:\n");
        // binaryPrinter(h_in[0], NUM_BITS);
        // printf(", ");
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

void cubRadixSort(cub::DoubleBuffer<uint32_t>& d_keys, size_t N, timeval& t_start, timeval& t_end) {

    //temporary storage
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, N);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // sorting operation
    gettimeofday(&t_start, NULL);
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, N);
    cudaDeviceSynchronize();
    gettimeofday(&t_end, NULL);

    cudaFree(d_temp_storage);
}

// Binary printer for debugging
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