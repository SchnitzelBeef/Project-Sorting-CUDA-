// General helper file from the course written by Cosmin  

#ifndef HELPER
#define HELPER

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

int gpuAssert(cudaError_t code) {
  if(code != cudaSuccess) {
    printf("GPU Error: %s\n", cudaGetErrorString(code));
    return -1;
  }
  return 0;
}

template<class T>
void randomInit(T* data, uint64_t size) {
    for (uint64_t i = 0; i < size; i++)
        data[i] = rand() / (T)RAND_MAX;
}

/**
 * Initialize the `data` array, which has `size` elements:
 * frac% of them are NaNs and (1-frac)% are random values.
 * 
 */
void randomMask(char* data, uint64_t size, float frac) {
    for (uint64_t i = 0; i < size; i++) {
        float r = rand() / (float)RAND_MAX;
        data[i] = (r >= frac) ? 1 : 0;
    }
}

// error for matmul: 0.02
template<class T>
bool validate(T* A, T* B, const uint64_t sizeAB, const T ERR){
    for(uint64_t i = 0; i < sizeAB; i++) {
        T curr_err = fabs( (A[i] - B[i]) / max(A[i], B[i]) ); 
        if (curr_err >= ERR) {
            printf("INVALID RESULT at flat index %llu: %f vs %f\n", i, A[i], B[i]);
            return false;
        }
    }
    printf("VALID RESULT!\n");
    return true;
}

template<class T>
bool validateExact(T* A, T* B, uint64_t sizeAB){
    for(uint64_t i = 0; i < sizeAB; i++) {
        if ( A[i] != B[i] ) {
            printf("INVALID RESULT at flat index %llu: %u vs %u\n", i, (float)A[i], (float)B[i]);
            return false;
        }
    }
    printf("VALID RESULT!\n");
    return true;
}

#endif
