#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#include "helper.h"

#define GPU_RUNS 300
 
__global__ void mul2Kernel(float* X, float *Y, int N) {
    const unsigned int gid_x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int gid_y = gid_x + N;
    if (gid_x < N) Y[gid_x] = X[gid_x] * X[gid_y];
}

// WRONG CODE, COPIED FROM ASSIGNMENT 1
int main(int argc, char** argv) {
    unsigned int N;
    
    { // reading the number of elements 
      if (argc != 2) { 
        printf("Num Args is: %d instead of 1. Exiting!\n", argc); 
        exit(1);
      }

      N = atoi(argv[1]);
      printf("N is: %d\n", N);

      const unsigned int maxN = 500000000;
      if(N > maxN) {
          printf("N is too big; maximal value is %d. Exiting!\n", maxN);
          exit(2);
      }
    }

    // use the first CUDA device:
    cudaSetDevice(0);

    unsigned int mem_size = N*sizeof(float);
    unsigned int gpu_mem_size_in = mem_size*2;

    // allocate host memory for both CPU and GPU
    float* h_in  = (float*) malloc(gpu_mem_size_in);
    float* gpu_res = (float*) malloc(mem_size);
    float* cpu_res = (float*) malloc(mem_size);

    // initialize the memory
    for(unsigned int i=0; i<N; ++i) {
        h_in[i] = (float)i;
        h_in[N+i] = (float)(i+2);
    }

    // sequential map on CPU with time measured
    double cpu_elapsed; double cpu_gigabytespersec; struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);

    for (unsigned int i = 0; i < N; i++){
        cpu_res[i] = h_in[i] * h_in[i+N];
    }

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    cpu_elapsed = (1.0 * (t_diff.tv_sec*1e6+t_diff.tv_usec));
    cpu_gigabytespersec = (2.0 * N * 4.0) / (cpu_elapsed * 1000.0);
    printf("The cpu took on average %f microseconds. GB/sec: %f \n", cpu_elapsed, cpu_gigabytespersec);

    // allocate device memory
    float* d_in;
    float* d_out;
    cudaMalloc((void**)&d_in,  gpu_mem_size_in);
    cudaMalloc((void**)&d_out, mem_size);

    // copy host memory to device
    cudaMemcpy(d_in, h_in, gpu_mem_size_in, cudaMemcpyHostToDevice);

    unsigned int B = 256;
    unsigned int numblocks = (N + B - 1) / B;
    // a small number of dry runs
    for(int r = 0; r < 1; r++) {
        dim3 block(B, 1, 1), grid(numblocks, 1, 1);
        mul2Kernel<<< grid, block>>>(d_in, d_out, N);
    }

    {
        double gpu_elapsed; double gpu_gigabytespersec; struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        for(int r = 0; r < GPU_RUNS; r++) {
            dim3 block(B, 1, 1), grid(numblocks, 1, 1);
            mul2Kernel<<< grid, block>>>(d_in, d_out, N);
        }
        cudaDeviceSynchronize();
        
        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        gpu_elapsed = (1.0 * (t_diff.tv_sec*1e6+t_diff.tv_usec)) / GPU_RUNS;
        gpu_gigabytespersec = (3.0 * N * 4.0) / (gpu_elapsed * 1000.0);
        printf("The kernel took on average %f microseconds. GB/sec: %f \n", gpu_elapsed, gpu_gigabytespersec);
    
        double speedup = cpu_elapsed / gpu_elapsed;
        printf("Speedup := cpu_elapsed / gpu_elapsed  = %f \n", speedup);
    }
        
    // check for errors
    gpuAssert( cudaPeekAtLastError() );

    // copy result from device to host
    cudaMemcpy(gpu_res, d_out, mem_size, cudaMemcpyDeviceToHost);

    // print result
    // for(unsigned int i=0; i<N; ++i) printf("GPU at %d: %.6f\n", i, gpu_res[i]);
    // for(unsigned int i=0; i<N; ++i) printf("CPU at %d: %.6f\n", i, cpu_res[i]);

    // element-wise compare of CPU and GPU execution
    for (unsigned int i = 0; i < N; i++) {
        float expected = cpu_res[i];
        float actual = gpu_res[i];
        if (fabs(actual - expected >= 0.000001)) {
            printf("Invalid result at index %d, actual: %f, expected: %f. \n", i, actual, expected);
            exit(3);
        }
    }

    printf("Successful Validation.\n");

    // clean-up memory
    free(h_in);       free(gpu_res);
    cudaFree(d_in);   cudaFree(d_out);
}
