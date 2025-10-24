#include "pbb_kernels.cuh"

// To be implemented
__global__ void
histogramKer(uint32_t* input
                            , uint32_t* histogram // Global set of histograms
                            , uint32_t mask
                            , uint32_t Q
                            , uint32_t N

) {

  // Shared memory buffer
  __shared__ uint32_t sh_hist[H];

  // Global thread ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

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
      histogram[i + blockIdx.x * H] = sh_hist[i]; 
    }
  }
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

// Better version of Scatter Ker
__global__ void scatterKer(uint32_t* input,
                           uint32_t* histogram_scan,
                           uint32_t* output,
                           uint32_t Q,
                           uint32_t N,
                           uint32_t mask) {
    // Shared memory buffer
    __shared__ uint32_t rank[H];

    // Global thread ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Zero shared ranks cooperatively (first H threads do it)
    if (threadIdx.x == 0) {
        for (int i = 0; i < H; i++) {
            rank[i] = 0;
        }
    }
    __syncthreads();  // Sync after init

    // Loop over elements assigned to this thread
    for (uint32_t i = 0; i < Q; i++) {
        uint32_t idx = tid * Q + i;
        if (idx >= N) break;

        // (d) SCATTER: place each number into correct position
        uint32_t elem = input[idx];
        int d = (int)(elem & mask);
        uint32_t rank_before = atomicAdd(&rank[d], 1u);
        int pos = histogram_scan[blockIdx.x * H + d] + rank_before;
        output[pos] = elem;
    }

    __syncthreads();
}