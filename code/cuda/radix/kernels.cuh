#include "pbb_kernels.cuh"

__global__ void
histogramKer( uint32_t* input
            , uint32_t* histogram // Global set of histograms
            , uint32_t mask
            , uint32_t shift
            , uint32_t Q
            , uint32_t N
            , uint32_t H
            , uint32_t NUM_BITS
) {

  // Shared memory buffer
  extern __shared__ uint32_t sh_hist[];

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

  if (threadIdx.x < H) //first H threads copy
    histogram[blockIdx.x * H + threadIdx.x] = sh_hist[threadIdx.x];
}

// Modified from assignment 3-4:
// blockDim.y = T; blockDim.x = T
// each block transposes a square T
template <int T> 
__global__ void
coalsTransposeKer(uint32_t* A, uint32_t* B, int heightA, int widthA) {
  __shared__ uint32_t tile[T][T+1];

  int x = blockIdx.x * T + threadIdx.x;
  int y = blockIdx.y * T + threadIdx.y;

  if( x < widthA && y < heightA )
      tile[threadIdx.y][threadIdx.x] = A[y*widthA + x];

  __syncthreads();

  x = blockIdx.y * T + threadIdx.x; 
  y = blockIdx.x * T + threadIdx.y;

  if( x < heightA && y < widthA )
      B[y*heightA + x] = tile[threadIdx.x][threadIdx.y];
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

// // Better version of Scatter Ker
// __global__ void scatterKer(uint32_t* input,
//                            uint32_t* histogram_scan,
//                            uint32_t* output,
//                            uint32_t Q,
//                            uint32_t N,
//                            uint32_t mask,
//                            uint32_t shift,
//                            uint32_t H,
//                            uint32_t NUM_BITS) {
//     // Shared memory buffer
//     extern __shared__ uint32_t rank[];

//     // Zero shared ranks cooperatively (first H threads do it)
//     for (int i = threadIdx.x; i < H; i += blockDim.x) {
//       rank[i] = 0;
//     }

//     __syncthreads();  // Sync after init

//     uint32_t block_start = blockIdx.x * (blockDim.x * Q);
//     // Loop over elements assigned to this thread
//     for (uint32_t i = 0; i < Q; i++) {
//          uint32_t idx = block_start + threadIdx.x + i * blockDim.x;
//         if (idx >= N) break;

//         // (d) SCATTER: place each number into correct position
//         uint32_t elem = input[idx];
//         int d = (int)((elem >> shift) & ((1 << NUM_BITS) - 1));
//         uint32_t rank_before = atomicAdd(&rank[d], 1u);
//         int pos = histogram_scan[blockIdx.x * H + d] + rank_before;
//         output[pos] = elem;
//     }
//     __syncthreads();
// }


// Even better version of last kernel 
__global__ void scatterKer(uint32_t* input,
                           uint32_t* histogram_scan,
                           uint32_t* histogram_sgm_scan,
                           uint32_t* output,
                           const size_t N,
                           const size_t Q,
                           const size_t bits,
                           const size_t start_bit) {

    // Local shared histogram across the block
    // USES STATIC MEMORY SO ONLY WORKS WHEN Q = 5 and B = 32
    __shared__ uint32_t elms[5*32];
    __shared__ uint32_t elms_shared[5*32];
    __shared__ uint32_t histogram_block[32];
    __shared__ uint32_t histogram_block_scan[32];

    uint32_t block_start = blockIdx.x * (Q * blockDim.x);
    uint32_t thread_offset = Q * threadIdx.x;
    
    // Load elements into shared memory:
    for (int q = 0; q < Q; q++) {
      uint32_t idx = q + block_start + thread_offset;
      if (idx >= N) break;
      elms[q] = input[idx]; 
    }

    __syncthreads();

    // Sort elements in block
    for (int b = 0; b < bits; b++){
      uint32_t sum = 0;
      
      for (int q = 0; q < Q; q++) {
        // Adds one to the sum if bit is zero
        sum += elms[q + thread_offset] ^ (b + start_bit);
      }

      histogram_block[threadIdx.x] = sum;
      histogram_block_scan[threadIdx.x] = sum; 

      // ------------------------------
      // Scan excl block
      
      // Up sweep:
      for (int d = 0; d < bits; d++) {
        // Can maybe make this loop go with one less iteration because last iteration is overwritten in line below
        if ((threadIdx.x + 1) % (1 << (d + 1)) == 0) {
          // Entered first only by thread 1, 3, 5, ...
          // Then by 3, 7, 11, ... in the second iteration of the loop
          int half_index = ((threadIdx.x + 1) >> 1) - 1;
          histogram_block_scan[threadIdx.x] += histogram_block_scan[half_index];
          // Can most likely remove:
          __syncthreads();
        }
      }
      histogram_block_scan[ blockDim.x - 1] = 0;
      __syncthreads();

      // Down sweep:
      for (int d = bits-1; d >= 0; d++) {
        if ((threadIdx.x + 1) % (1 << (d + 1)) == 0) {
          int half_index = ((threadIdx.x + 1) >> 1) - 1;
          uint32_t tmp = histogram_block_scan[half_index];
          histogram_block_scan[half_index] = histogram_block_scan[threadIdx.x];
          histogram_block_scan[threadIdx.x] += tmp;
        }
        // Can most likely remove:
        __syncthreads();
      }
      // Turn exclusive scan inclusive by adding all elements (not pretty)
      histogram_block_scan[threadIdx.x] += histogram_block[threadIdx.x];

      // ------------------------------
      
      int splitpoint = histogram_block_scan[ blockDim.x - 1];
      if (threadIdx.x == 0) {
        sum = 0;
      } else {
        sum = histogram_block_scan[threadIdx.x];
      }

      // This syncronization can maybe be removed
      __syncthreads();

      for (int q = 0; q < Q; q++) {
          // Adds one to the sum if bit is zero
          uint32_t elm = elms[q + thread_offset];
          int isZero = elm ^ (b + start_bit);
          sum += isZero;
          if (isZero) {
            elms_shared[sum - 1] = elm;
          } else {
            elms_shared[splitpoint + thread_offset + q - sum] = elm;
          } 
      }

      // This syncronization can maybe be removed
      __syncthreads();

      // Load elements back
      if (b < bits-1) {
        for (int q = 0; q < Q; q++) {
          elms[q] = elms_shared[q];
        }
      } else {
        for (int q = 0; q < Q; q++) {
          elms[q] = elms_shared[q * blockDim.x + threadIdx.x];
        }
      }
    }

    // Scatter
    for (int q = 0; q < Q; q++) {
      int index = thread_offset + q;
      uint32_t bin = elms[index];

      //Find final position
      int pos = histogram_scan[block_start + bin] - histogram_sgm_scan[block_start + bin] + (q + threadIdx.x * blockDim.x);
      output[pos] = bin;
    }
}