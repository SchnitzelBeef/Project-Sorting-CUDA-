#include "pbb_kernels.cuh"

__global__ void
histogramKer( uint32_t* input
            , uint32_t* histogram // Global set of histograms
            , uint32_t shift
            , uint32_t N
) {

  // Shared memory buffer
  __shared__ uint32_t sh_hist[H];

  // Zeroing shared histogram (only works for H >= B)
  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    sh_hist[i] = 0;
  }

  //coalesced access on block level - should be warp level??
  uint32_t block_start = blockIdx.x * (blockDim.x * Q);
  for (uint32_t i = 0; i < Q; i++) {
      uint32_t idx = block_start + threadIdx.x + (i * blockDim.x);
      if (idx < N) {
        uint32_t curr_val = input[idx];
        uint32_t bucket = (curr_val >> shift) & ((1 << NUM_BITS) - 1);
        atomicAdd(&sh_hist[bucket], 1);
      }
  }

  __syncthreads();

  // Copy shared histogram to global memory (only works for H >= B)
  for (int i = threadIdx.x; i < H; i += blockDim.x) {
      histogram[blockIdx.x * H + i] = sh_hist[i];
  }
}

// Modified from assignment 3-4:
// blockDim.y = T; blockDim.x = T
// each block transposes a square T
template <int T> 
__global__ void
coalsTransposeKer(uint32_t* A, uint32_t* C, int heightA, int widthA) {
  __shared__ uint32_t tile[T][T+1];

  int x = blockIdx.x * T + threadIdx.x;
  int y = blockIdx.y * T + threadIdx.y;

  if( x < widthA && y < heightA )
      tile[threadIdx.y][threadIdx.x] = A[y*widthA + x];

  __syncthreads();

  x = blockIdx.y * T + threadIdx.x; 
  y = blockIdx.x * T + threadIdx.y;

  if( x < heightA && y < widthA )
      C[y*heightA + x] = tile[threadIdx.x][threadIdx.y];
}

// Creates a flag array representing the (static) per block histogram size
__global__ void createFlagKer(char* d_out, const size_t N) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) d_out[tid] = tid % H == 0;
}

// Partition and scatter kernel 
__global__ void scatterKer(uint32_t* input,
                           uint32_t* histogram_scan,
                           uint32_t* histogram_sgm_scan,
                           uint32_t* output,
                           const size_t N,
                           const size_t bits,
                           const size_t shift) {

  // Local shared elements and histogram across the block
  __shared__ uint32_t elms[Q*B];
  __shared__ uint32_t elms_shared[Q*B];
  __shared__ uint32_t histogram_block[B];
  __shared__ uint32_t histogram_block_scan[B];

  uint32_t block_start = blockIdx.x * (Q * blockDim.x);
  uint32_t thread_offset = Q * threadIdx.x;
  
  // Load elements into shared memory:
  for (int q = 0; q < Q; q++) {
    uint32_t idx = q + block_start + thread_offset;
    if (idx < N) {
      elms[q + thread_offset] = input[idx];
    }
  }

  // Sort elements in block (same as performing partition2 on each bit starting with least sig.)
  for (int b = 0; b < bits; b++){
    uint32_t sum = 0;

    for (int q = thread_offset; q < thread_offset + Q; q++) {
      // Adds one to the sum if bit is zero
      uint32_t idx = block_start + q;
      // Only count if within bounds
      if (idx < N) {
          sum += ((elms[q] & (1 << (b + shift))) == 0);
      }
    }

    histogram_block[ threadIdx.x] = sum;
    histogram_block_scan[ threadIdx.x] = sum; 
    
    // Scan excl block    
    int offset = 1;
    // Up sweep in logB steps
    for (int d = B >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if ( threadIdx.x < d)
            histogram_block_scan[offset * (2 * threadIdx.x + 2) - 1] += histogram_block_scan[offset * (2 * threadIdx.x + 1) - 1];
        offset <<= 1;
    }

    // Clear the last element for exclusive scan
    if ( threadIdx.x == 0)
        histogram_block_scan[B - 1] = 0;

    // Down sweep in logB steps
    for (int d = 1; d < B; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if ( threadIdx.x < d) {
            int ai = offset * (2 * threadIdx.x + 1) - 1;
            int bi = offset * (2 * threadIdx.x + 2) - 1;
            int t = histogram_block_scan[ai];
            histogram_block_scan[ai] = histogram_block_scan[bi];
            histogram_block_scan[bi] += t;
        }
    }

    __syncthreads();
    // Turn exclusive scan inclusive by adding all elements (not pretty)
    histogram_block_scan[ threadIdx.x] += histogram_block[ threadIdx.x];

    __syncthreads();

    int splitpoint = histogram_block_scan[ blockDim.x - 1];
    if ( threadIdx.x == 0) sum = 0;
    else sum = histogram_block_scan[ threadIdx.x - 1];
    
    for (int q = 0; q < Q; q++) {
      // Adds one to the sum if bit is zero
      uint32_t idx = q + block_start + thread_offset;
      // Only count if within bounds
      if (idx < N) {
        uint32_t elm = elms[q + thread_offset];
        if ((elm & (1 << (b + shift))) == 0) {
          sum += 1;
          elms_shared[sum - 1] = elm;
        } else {
          elms_shared[splitpoint + thread_offset + q - sum] = elm;
        } 
      }
    }

    __syncthreads();

    // Load elements back
    for (int q = 0; q < Q; q++) {
      elms[q + thread_offset] = elms_shared[q + thread_offset];
    }
  }

  // Scatter
  for (int q = thread_offset; q < thread_offset + Q; q++) {
    int idx = q + block_start;
    if (idx < N) {
      int bucket = (elms[q] >> shift) & ((1 << NUM_BITS) - 1);
      //Find final position
      int pos = histogram_scan[H * blockIdx.x + bucket] - histogram_sgm_scan[H * blockIdx.x + bucket] + q;
      output[pos] = elms[q];
    }
  }
}
