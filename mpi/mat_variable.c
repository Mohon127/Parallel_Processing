#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// Function to print a matrix
void display(int rows, int cols, int matrix[rows][cols]) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%3d ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Expect 4 arguments: K M N P
    if (argc < 5) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s K M N P\n", argv[0]);
            fprintf(stderr, "Example: mpirun -np 4 %s 500 100 100 100\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int K = atoi(argv[1]); // number of matrix pairs
    int M = atoi(argv[2]); // rows of A
    int N = atoi(argv[3]); // cols of A / rows of B
    int P = atoi(argv[4]); // cols of B

    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&P, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int (*A)[M][N] = NULL;
    int (*B)[N][P] = NULL;
    int (*R)[M][P] = NULL;

    if (rank == 0) {
        A = malloc(K * sizeof(*A));
        B = malloc(K * sizeof(*B));
        R = malloc(K * sizeof(*R));

        // Initialize random matrices
        for (int k = 0; k < K; k++) {
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    A[k][i][j] = rand() % 100;
                }
            }
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < P; j++) {
                    B[k][i][j] = rand() % 100;
                }
            }
        }
    }

    // Compute counts and displacements
    int baseK = K / size;
    int remainder = K % size;

    int *sendcountsA = malloc(size * sizeof(int));
    int *displsA     = malloc(size * sizeof(int));
    int *sendcountsB = malloc(size * sizeof(int));
    int *displsB     = malloc(size * sizeof(int));
    int *sendcountsR = malloc(size * sizeof(int));
    int *displsR     = malloc(size * sizeof(int));

    int offsetA = 0, offsetB = 0, offsetR = 0;
    for (int i = 0; i < size; i++) {
        int myK = baseK + (i < remainder ? 1 : 0);
        sendcountsA[i] = myK * M * N;
        sendcountsB[i] = myK * N * P;
        sendcountsR[i] = myK * M * P;
        displsA[i] = offsetA;
        displsB[i] = offsetB;
        displsR[i] = offsetR;
        offsetA += sendcountsA[i];
        offsetB += sendcountsB[i];
        offsetR += sendcountsR[i];
    }

    int localK = baseK + (rank < remainder ? 1 : 0);

    // Allocate local arrays
    int (*localA)[M][N] = malloc(localK * sizeof(*localA));
    int (*localB)[N][P] = malloc(localK * sizeof(*localB));
    int (*localR)[M][P] = malloc(localK * sizeof(*localR));

    // Scatter with variable counts
    MPI_Scatterv(A, sendcountsA, displsA, MPI_INT,
                 localA, sendcountsA[rank], MPI_INT,
                 0, MPI_COMM_WORLD);

    MPI_Scatterv(B, sendcountsB, displsB, MPI_INT,
                 localB, sendcountsB[rank], MPI_INT,
                 0, MPI_COMM_WORLD);

    double startTime = MPI_Wtime();

    // Local multiplication
    for (int k = 0; k < localK; k++) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < P; j++) {
                localR[k][i][j] = 0;
                for (int l = 0; l < N; l++) {
                    localR[k][i][j] += localA[k][i][l] * localB[k][l][j];
                }
                localR[k][i][j] %= 100; // optional modulo
            }
        }
    }

    double endTime = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);

    // Gather results back
    MPI_Gatherv(localR, sendcountsR[rank], MPI_INT,
                R, sendcountsR, displsR, MPI_INT,
                0, MPI_COMM_WORLD);

    // Print timing
    printf("Process %d: Time taken = %f seconds\n", rank, endTime - startTime);

    /*
    if (rank == 0) {
        for (int k = 0; k < K; k++) {
            printf("Result Matrix R%d:\n", k);
            display(M, P, R[k]);
        }
    }
    */

    // Free memory
    free(localA);
    free(localB);
    free(localR);
    free(sendcountsA);
    free(sendcountsB);
    free(sendcountsR);
    free(displsA);
    free(displsB);
    free(displsR);

    if (rank == 0) {
        free(A);
        free(B);
        free(R);
    }

    MPI_Finalize();
    return 0;
}


/*
mpicc mat_variable.c -o mat_var
mpirun -np 4 ./mat_var 244 100 100 100
*/
