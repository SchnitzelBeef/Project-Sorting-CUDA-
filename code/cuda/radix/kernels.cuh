#include "pbb_kernels.cuh"

#define NUM_BITS 4
#define RADIX 1 << NUM_BITS

// To be implemented
__global__ void histogramKer(uint32_t* input
                            , uint32_t* histogram // Global set of histograms
                            , uint32_t mask
                            , uint32_t Q
                            , uint32_t N

) {

  // Copy memory from global to shared memory here??

  __shared__ uint32_t sh_hist[RADIX];
  
  // Zeroing shared histogram
 if (threadIdx.x == 0) {  
  for (int i = 0; i < RADIX; i ++) {
    sh_hist[i] = 0;
  }
 }
  __syncthreads();

  // Global thread ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  

  // i jumping over all threads for every iteration
  for (uint32_t i = 0; i < Q; i ++) {
    uint32_t idx = i * blockDim.x + tid;
    if (idx >= N) break;
    uint32_t curr_val = input[idx];
    uint32_t bucket = curr_val & mask;
    atomicAdd((uint32_t*)&sh_hist[bucket], 1);
  }

  __syncthreads();

  // Copy back to global memory via the first thread
  // If at some point RADIX becomes large, consider changing this to be parallel
  if (threadIdx.x == 0) { //  First Thread of the Block
    for (int i = 0; i < RADIX; i++) {
      histogram[i + blockIdx.x * RADIX] = sh_hist[i];
    }
  }
}