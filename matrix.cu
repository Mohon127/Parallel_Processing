%%writefile matrix.cu
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
using namespace std;

__global__ void matrixMul(float *A, float *B, float *R, int M, int N, int P, int batchOffset) {
    int k = threadIdx.x + batchOffset;   // one thread per matrix
    if (k >= gridDim.x * blockDim.x) return;

    float *a = A + k * M * N;
    float *b = B + k * N * P;
    float *r = R + k * M * P;

    // compute matrix multiplication
    for (int i = 0; i < M; i++) {
        for (int l = 0; l < P; l++) {
            r[i * P + l] = 0.0f;
            for (int j = 0; j < N; j++) {
                r[i * P + l] += a[i * N + j] * b[j * P + l];
            }
        }
    }
}

// print one matrix at given index
void printMatrixAtIndex(float *A, int index, int M, int N) {
    int offset = index * M * N;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            cout << A[offset + i * N + j] << " ";
        }
        cout << endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 6) {
        cout << "Usage: ./matrix <threads> <k> <m> <n> <p>" << endl;
        return 1;
    }

    int threads = atoi(argv[1]); // threads per block
    int K = atoi(argv[2]);
    int M = atoi(argv[3]);
    int N = atoi(argv[4]);
    int P = atoi(argv[5]);

    int sizeA = K * M * N;
    int sizeB = K * N * P;
    int sizeR = K * M * P;

    // Host memory
    float *h_A = (float*)malloc(sizeA * sizeof(float));
    float *h_B = (float*)malloc(sizeB * sizeof(float));
    float *h_R = (float*)malloc(sizeR * sizeof(float));

    // Initialize random matrices
    for (int i = 0; i < sizeA; i++) h_A[i] = rand() % 10;
    for (int i = 0; i < sizeB; i++) h_B[i] = rand() % 10;

    // Device memory
    float *d_A, *d_B, *d_R;
    cudaMalloc(&d_A, sizeA * sizeof(float));
    cudaMalloc(&d_B, sizeB * sizeof(float));
    cudaMalloc(&d_R, sizeR * sizeof(float));

    cudaMemcpy(d_A, h_A, sizeA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_R, 0, sizeR * sizeof(float));

    int remaining = K;
    int batchOffset = 0;
    while (remaining > 0) {
        int currentBatchSize = min(remaining, threads);
        matrixMul<<<1, currentBatchSize>>>(d_A, d_B, d_R, M, N, P, batchOffset);
        cudaDeviceSynchronize();
        remaining -= currentBatchSize;
        batchOffset += currentBatchSize;
    }

    

    // Copy result back
    cudaMemcpy(h_R, d_R, sizeR * sizeof(float), cudaMemcpyDeviceToHost);

    // Output
    if (K > 9) {
        cout << "Matrix A[9]:" << endl;
        printMatrixAtIndex(h_A, 9, M, N);

        cout << "Matrix B[9]:" << endl;
        printMatrixAtIndex(h_B, 9, N, P);

        cout << "Matrix C[9]:" << endl;
        printMatrixAtIndex(h_R, 9, M, P);
    } else {
        cout << "Error: K <= 9, so A[9], B[9], C[9] do not exist." << endl;
    }

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_R);
    free(h_A); free(h_B); free(h_R);
    return 0;
}


//!nvcc -arch=sm_75 matrix.cu -o matrix
//!time ./matrix 400 2 2 2 2 > output.txt
