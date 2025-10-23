#include "pbb_kernels.cuh"

#define NUM_BITS 4

// To be implemented
__global__ void histogramKer(uint32_t* input
                            , uint32_t* histogram // Global set of histograms
                            , uint32_t mask
                            , uint32_t Q
                            , uint32_t N
                            , uint32_t H

) {

  // Shared memory buffer
  __shared__ uint32_t sh_hist[H];

  // Global thread ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  
  // Zeroing shared histogram
  // for (uint32_t i = 0; i < Q; i ++) {
  //   uint32_t idx = i * blockDim.x + tid;
  //   if (idx >= N) break;
  //   uint32_t curr_val = input[idx];
  //   uint32_t bucket = curr_val & mask;
  //   sh_hist[i] = 0;
  // }

  // Zeroing shared histogram
 if (threadIdx.x == 0) {  
  for (int i = 0; i < H; i ++) {
    sh_hist[i] = 0;
  }
 }
  __syncthreads();


  // i jumping over all threads for every iteration
  for (uint32_t i = 0; i < Q; i ++) {
    uint32_t idx = tid * Q + i;
    if (idx >= N) break;
    uint32_t curr_val = input[idx];
    uint32_t bucket = curr_val & mask;
    atomicAdd((uint32_t*)&sh_hist[bucket], 1);
  }

  __syncthreads();

  // Copy back to global memory via the first thread
  // If at some point H becomes large, consider changing this to be parallel
  if (threadIdx.x == 0) { //  First Thread of the Block
    for (int i = 0; i < H; i++) {
      histogram[i] = i;
      // histogram[i + blockIdx.x * H] = tid; 
    }
  }
}