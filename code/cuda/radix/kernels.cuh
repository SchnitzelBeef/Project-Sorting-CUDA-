#include "pbb_kernels.cuh"

__global__ void
histogramKer( uint32_t* input
            , uint32_t* histogram // Global set of histograms
            , uint32_t shift
            , uint32_t N
) {

  // Shared memory buffer
  __shared__ uint32_t sh_hist[H];

  // Zeroing shared histogram
  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    sh_hist[i] = 0;
  }
  __syncthreads();


  //coalesced access on block level - should be warp level??
  uint32_t block_start = blockIdx.x * (blockDim.x * Q);
  for (uint32_t i = 0; i < Q; i++) {
      uint32_t idx = block_start + threadIdx.x + (i * blockDim.x);
      if (idx >= N) break;
      uint32_t curr_val = input[idx];
      uint32_t bucket = (curr_val >> shift) & ((1 << NUM_BITS) - 1);
      atomicAdd(&sh_hist[bucket], 1);
  }

  __syncthreads();

  // Copy shared histogram to global memory
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

// Temporary fix to make scan exclusive and not inclusive 
__global__ void
shiftKer(uint32_t* input
                        , uint32_t* output
                        , uint32_t N
) {
  // Global thread ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < N) {

    // Creates a copy of the input array shifted one element
    if (tid > 0) {
      output[tid] = input[tid-1];
    }
    else {
      output[tid] = 0;
    }
  }
}


// Last kernel 
__global__ void scatterKer(uint32_t* input,
                           uint32_t* histogram_scan,
                           uint32_t* histogram_sgm_scan,
                           uint32_t* output,
                           const size_t N,
                           const size_t bits,
                           const size_t start_bit) {

  // Local shared histogram across the block
  // USES STATIC MEMORY SO ONLY WORKS WHEN HARDCODED Q = 4 and B = 2

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

  __syncthreads();

  // Sort elements in block
  for (int b = 0; b < bits; b++){
    uint32_t sum = 0;
    
    for (int q = 0; q < Q; q++) {
      // Adds one to the sum if bit is zero
      if ((elms[q + thread_offset] & (1 << (b + start_bit))) == 0) {
        sum += 1;
      }
    }

    histogram_block[ threadIdx.x] = sum;
    histogram_block_scan[ threadIdx.x] = sum; 


    __syncthreads();
    // histogram_block_scan[1] += histogram_block_scan[0];
    // histogram_block_scan[2] += histogram_block_scan[1];
    // histogram_block_scan[3] += histogram_block_scan[2];

    // ------------------------------
    // Scan excl block
    
    int offset = 1;
    // Up-sweep (reduce) phase
    for (int d = B >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (threadIdx.x < d)
            histogram_block_scan[offset * (2 * threadIdx.x + 2) - 1] += histogram_block_scan[offset * (2 * threadIdx.x + 1) - 1];
        offset *= 2;
    }

    // Clear the last element for exclusive scan
    if (threadIdx.x == 0)
        histogram_block_scan[B - 1] = 0;

    // Down-sweep phase
    for (int d = 1; d < B; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (threadIdx.x < d) {
            int ai = offset * (2 * threadIdx.x + 1) - 1;
            int bi = offset * (2 * threadIdx.x + 2) - 1;

            int t = histogram_block_scan[ai];
            histogram_block_scan[ai] = histogram_block_scan[bi];
            histogram_block_scan[bi] += t;
        }
    }



    // Up sweep: (4 := B)
    // for (int d = 0; d <= 4; d = 1 << d) { //change 2 to log og B
    //   // Can maybe make this loop go with one less iteration because last iteration is overwritten in line below
    //   if (( threadIdx.x + 1) % d == 0) {
    //     // Entered first only by thread 1, 3, 5, ...
    //     // Then by 3, 7, 11, ... in the second iteration of the loop
    //     int half_index = (( threadIdx.x + 1) >> 1) - 1;
    //     int tmp = histogram_block_scan[ threadIdx.x];
    //     histogram_block_scan[ threadIdx.x] = tmp + histogram_block_scan[half_index];
    //   }
    //   __syncthreads();

    // }
    // histogram_block_scan[ blockDim.x - 1] = 0;
    // __syncthreads();

    // // Down sweep: (4 := B)
    // for (int d = 2-1; d >= 0; d = d >> 1) {
    //   if (( threadIdx.x + 1) % (1 << (d + 1)) == 0) {
    //     int half_index = (( threadIdx.x + 1) >> 1) - 1;
    //     uint32_t tmp = histogram_block_scan[half_index];
    //     int tmp2 = histogram_block_scan[ threadIdx.x];
    //     histogram_block_scan[half_index] = histogram_block_scan[ threadIdx.x];
    //     histogram_block_scan[ threadIdx.x] = tmp + tmp2;
    //   }
    //   __syncthreads();

    // }

    __syncthreads();
    // Turn exclusive scan inclusive by adding all elements (not pretty)
    histogram_block_scan[ threadIdx.x] += histogram_block[ threadIdx.x];

    // ------------------------------

    __syncthreads();

    int splitpoint = histogram_block_scan[ blockDim.x - 1];
    if ( threadIdx.x == 0) {
      sum = 0;
    } else {
      sum = histogram_block_scan[ threadIdx.x - 1];
    }

    __syncthreads();
    
    
    for (int q = 0; q < Q; q++) {
        // Adds one to the sum if bit is zero
        uint32_t elm = elms[q + thread_offset];
        if ((elm & (1 << (b + start_bit))) == 0) {
          sum += 1;
          elms_shared[sum - 1] = elm;
        } else {
          elms_shared[splitpoint + thread_offset + q - sum] = elm;
        } 
    }

    __syncthreads();

    // Load elements back
    for (int q = 0; q < Q; q++) {
      elms[q + thread_offset] = elms_shared[q + thread_offset];
    }
    __syncthreads();
    
  }
  // for (int q = 0; q < Q; q++) {
  //   uint32_t idx = q + block_start + thread_offset;
  //   if (idx < N) {
  //     output[idx] = elms[q + thread_offset];
  //   }
  // }
  // return;
  
  // for (int q = 0; q < Q; q++) {
  //   uint32_t idx = q + block_start + thread_offset;
  //   if (idx < N) {
  //     output[idx] = elms[q + thread_offset];
  //   }
  // }
  // return;


  // Scatter
  for (int q = 0; q < Q; q++) {
    uint32_t bin = elms[thread_offset + q];
    
    //Find final position
    int pos = histogram_scan[(1 << bits) * blockIdx.x + bin] - histogram_sgm_scan[(1 << bits) * blockIdx.x + bin] + (q + threadIdx.x * Q);
    output[pos] = elms[thread_offset + q];
  }
}