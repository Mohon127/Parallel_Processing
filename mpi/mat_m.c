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

    int K = 120, M = 100, N = 100, P = 100;
    /*
    // Check conmand Line argunents
    if (argc < 5) {
    if (rank == 0) {
        printf("mpirun -np 4 ./fileName  <K> <M> <N> <P>");
        MPI_Finalize();
        return 1;
    }

    int K = atoi(argv[1]);
    int M = atoi(argv[2]);
    int N = atoi(argv[3]);
    int P = atoi(argv[4]);

    */



    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&P, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (K % size != 0) {
        if (rank == 0)
            printf("Error: K (%d) must be divisible by number of processes (%d).\n", K, size);
        MPI_Finalize();
        return 1;
    }

    int localK = K / size;   // number of matrices per process


    int (*A)[M][N] = NULL;
    int (*B)[N][P] = NULL;
    int (*R)[M][P] = NULL;

    if (rank == 0) {
        A = malloc(K * sizeof(*A));
        B = malloc(K * sizeof(*B));
        R = malloc(K * sizeof(*R));

        // Initialize
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

        // If you want to see the initialization of matrix
        // for (int k = 0; k < 1; k++) {
        //     for (int i = 0; i < M; i++) {
        //         for (int j = 0; j < N; j++) {
        //             printf("%d ", B[k][i][j]);
                    
        //         } printf("\n");
        //     }
            
        // }
    }

    // Allocate local arrays (only what each process needs)
    int (*localA)[M][N] = malloc(localK * M * N * sizeof(int));
    int (*localB)[N][P] = malloc(localK * M * N * sizeof(int));
    int (*localR)[M][P] = malloc(localK * M * N * sizeof(int));

    // Distribute data
    MPI_Scatter(A, localK * M * N, MPI_INT, localA, localK * M * N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(B, localK * N * P, MPI_INT, localB, localK * N * P, MPI_INT, 0, MPI_COMM_WORLD);

    //MPI_Barrier(MPI_COMM_WORLD);
    double startTime = MPI_Wtime();

    // Local multiplication
    for (int k = 0; k < localK; k++) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < P; j++) {
                localR[k][i][j] = 0;
                for (int l = 0; l < N; l++) {
                    localR[k][i][j] += (localA[k][i][l] * localB[k][l][j]) ;
                }
                localR[k][i][j] %= 100;
            }
        }
    }

    double endTime = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);

    // Gather results back
    MPI_Gather(localR, localK * M * P, MPI_INT, R, localK * M * P, MPI_INT, 0, MPI_COMM_WORLD);

    // Print timing
    printf("Process %d: Time taken = %f seconds\n", rank, endTime - startTime);

    // Uncomment to print results
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
    if (rank == 0) {
        free(A);
        free(B);
        free(R);
    }

    MPI_Finalize();
    return 0;
}



// mpicc helllo.c -o hello
// mpirun -np 4 ./hello 