#include <stdio.h>
#include <math.h>

#define N 10
#define RADIX 16

void print_binary(int n) {
    // Handle 0 case
    if (n == 0) {
        printf("0");
        return;
    }
    
    // For positive numbers only; adjust for negatives if needed
    if (n < 0) {
        printf("-");  // Prefix for negative
        n = -n;       // Work with absolute value (simplified)
    }
    
    char binary[33] = {0};  // Buffer for 32 bits + null terminator
    int i = 31;             // Start from MSB (31st bit for 32-bit int)
    
    while (i >= 0) {
        binary[i] = (n & 1) ? '1' : '0';
        n >>= 1;  // Right-shift to check next bit
        i--;
    }
    
    printf("%s", binary);
}

int main() {
    //int input[N] = {0b0001, 0b1111, 0b0000, 0b0101};
    int input[N] = {1234, 5678, 9012, 3456, 7890, 2345, 6789, 1122, 3344, 5566};
    int output[N];
    int digits[N];
    int mask = 0xF;

    for (int iteration = 0; iteration < 4; iteration++) {

        int bucket[RADIX] = {0};
        int offsets[RADIX] = {0};
        int rank[RADIX] = {0};
       
        // (a) MAP: extract digit corresponding to current iteration
        for (int i = 0; i < N; i++) {
           
            digits[i] = (input[i] >> (iteration * 4)) & 0xF;
             
            // digits[i] = input[i] & 0xF;
            // input[i] = input[i] & 0xF;
        }
       // mask = mask << iteration * 4;
        // input[i] & 0xF   --> get last  4 bits of input[i]
        
        printf("a) Map (last digit): ");
        for (int i = 0; i < N; i++) printf("%d ", digits[i]);
        printf("\n");

        // (b) REDUCE: count occurrences per bucket
        for (int i = 0; i < N; i++) {
            bucket[digits[i]]++;
        }

        printf("b) Reduce (buckets): ");
        for (int i = 0; i < RADIX; i++) printf("%d ", bucket[i]);
        printf("\n");

        // (c) SCAN: prefix sum to compute offsets
        int sum = 0;
        for (int i = 0; i < RADIX; i++) {
            offsets[i] = sum;
            sum += bucket[i];
        }

        printf("c) Scan (offsets): ");
        for (int i = 0; i < RADIX; i++) printf("%d ", offsets[i]);
        printf("\n");

        // (d) SCATTER: place each number into correct position
        for (int i = 0; i < N; i++) {
            int d = digits[i];
            int pos = offsets[d] + rank[d]++;
            output[pos] = input[i];
        }

        printf("d) Scatter (output): ");
        for (int i = 0; i < N; i++) printf("%d ", output[i]);
        printf("\n");

        // printf("d) Scatter (output): ");
        // for (int i = 0; i < N; i++) {
        //     print_binary(output[i]);
        //     printf("\n");    
        // }
            
        printf("\n");

        for (int i = 0; i < N; i++) {
            input[i] = output[i];
        }
    }

    // (e) Next iteration (if multi-digit sort)
    printf("\nSorted by last digit:\n");
    for (int i = 0; i < N; i++) printf("%d ", output[i]);
    printf("\n");

    return 0;
}
