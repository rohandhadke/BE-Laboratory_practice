// Matrix Multiplication using CUDA C
#include <stdio.h>
#include <stdlib.h>

#define N 3  // Size of the matrix (N x N)

// CUDA kernel for matrix multiplication
__global__ void matrixMulKernel(int *A, int *B, int *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index

    if (row < n && col < n) {
        int sum = 0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main() {
    int size = N * N * sizeof(int);

    int *h_A, *h_B, *h_C; // Host pointers
    int *d_A, *d_B, *d_C; // Device pointers

    // Allocate memory on host
    h_A = (int *)malloc(size);
    h_B = (int *)malloc(size);
    h_C = (int *)malloc(size);

    // Initialize matrices A and B
    printf("Matrix A:\n");
    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() % 10; // Random numbers between 0 and 9
        printf("%d ", h_A[i]);
        if ((i + 1) % N == 0) printf("\n");
    }

    printf("\nMatrix B:\n");
    for (int i = 0; i < N * N; i++) {
        h_B[i] = rand() % 10;
        printf("%d ", h_B[i]);
        if ((i + 1) % N == 0) printf("\n");
    }

    // Allocate memory on device
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch kernel
    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Synchronize to ensure kernel completion
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Display result
    printf("\nResultant Matrix C (A x B):\n");
    for (int i = 0; i < N * N; i++) {
        printf("%d ", h_C[i]);
        if ((i + 1) % N == 0) printf("\n");
    }

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
