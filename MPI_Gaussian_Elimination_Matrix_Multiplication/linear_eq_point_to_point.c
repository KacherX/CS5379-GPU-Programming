#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void distribute_data_p2p(double *A, double *b,
                         double *A_local, double *b_local,
                         int N, int local_n, int rank)
{
    if (rank == 0) {
        for (int p = 1; p < N / local_n; p++) {
            int offset = p * local_n;
            MPI_Send(&A[offset * N], local_n * N, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
            MPI_Send(&b[offset],     local_n,     MPI_DOUBLE, p, 1, MPI_COMM_WORLD);
        }

        for (int i = 0; i < local_n; i++) {
            for (int j = 0; j < N; j++)
                A_local[i * N + j] = A[i * N + j];
            b_local[i] = b[i];
        }
    } else {
        MPI_Recv(A_local, local_n * N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(b_local, local_n,     MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

void gaussian_elimination_p2p(double *A_local, double *b_local,
                              int N, int local_n, int rank, int size)
{
    double *pivot_row = malloc(N * sizeof(double));
    double pivot_b;

    for (int k = 0; k < N; k++) {

        int pivot_owner = k / local_n;
        int pivot_local = k % local_n;

        if (rank == pivot_owner) {
            // Pivot Row Gathering
            for (int j = 0; j < N; j++)
                pivot_row[j] = A_local[pivot_local * N + j];
            pivot_b = b_local[pivot_local];

            // Pivot Normalization
            double diag = pivot_row[k];
            if (fabs(diag) < 1e-12 && rank == 0)
                printf("Zero pivot encountered at k=%d\n", k);

            for (int j = k; j < N; j++)
                pivot_row[j] /= diag;
            pivot_b /= diag;

            // Pivot Write-Back
            for (int j = 0; j < N; j++)
                A_local[pivot_local * N + j] = pivot_row[j];
            b_local[pivot_local] = pivot_b;

            // Point-To-Point operation
            for (int p = 0; p < size; p++) {
                if (p == pivot_owner) continue;
                MPI_Send(pivot_row, N, MPI_DOUBLE, p, 2, MPI_COMM_WORLD);
                MPI_Send(&pivot_b, 1, MPI_DOUBLE, p, 3, MPI_COMM_WORLD);
            }
        } 
        else {
            MPI_Recv(pivot_row, N, MPI_DOUBLE, pivot_owner, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&pivot_b, 1, MPI_DOUBLE, pivot_owner, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Row Elimination
        for (int i = 0; i < local_n; i++) {
            int gi = rank * local_n + i;
            if (gi <= k) continue;

            double factor = A_local[i * N + k];
            if (fabs(factor) < 1e-12) continue;

            for (int j = k; j < N; j++)
                A_local[i * N + j] -= factor * pivot_row[j];

            b_local[i] -= factor * pivot_b;
            A_local[i * N + k] = 0.0;
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    free(pivot_row);
}

void gather_results_p2p(double *A_local, double *b_local,
                        double *A, double *b,
                        int N, int local_n, int rank, int size)
{
    if (rank == 0) {
        // Local Block Copy
        for (int i = 0; i < local_n; i++) {
            for (int j = 0; j < N; j++)
                A[i * N + j] = A_local[i * N + j];
            b[i] = b_local[i];
        }

        // Receive Remaining Data
        for (int p = 1; p < size; p++) {
            int offset = p * local_n;
            MPI_Recv(&A[offset * N], local_n * N, MPI_DOUBLE, p, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&b[offset],     local_n,     MPI_DOUBLE, p, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(A_local, local_n * N, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD);
        MPI_Send(b_local, local_n,     MPI_DOUBLE, 0, 5, MPI_COMM_WORLD);
    }
}

void back_substitution(double *A, double *b, double *x, int N)
{
    for (int i = N - 1; i >= 0; i--) {
        double sum = b[i];
        for (int j = i + 1; j < N; j++)
            sum -= A[i * N + j] * x[j];
        x[i] = sum;
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) printf("Usage: %s <matrix-size>\n", argv[0]);
        MPI_Finalize();
        return 0;
    }

    int N = atoi(argv[1]);
    int local_n = N / size;
    
    // Allocation
    double *A_local = malloc((size_t)local_n * N * sizeof(double));
    double *b_local = malloc(local_n * sizeof(double));

    double *A = NULL, *b = NULL, *x = NULL;

    if (rank == 0) {
        A = malloc((size_t)N * N * sizeof(double));
        b = malloc(N * sizeof(double));
        x = malloc(N * sizeof(double));

        srand(1);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++)
                A[i * N + j] = rand() % 10 + 1;
            b[i] = rand() % 10 + 1;
        }
    }

    distribute_data_p2p(A, b, A_local, b_local, N, local_n, rank);

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    gaussian_elimination_p2p(A_local, b_local, N, local_n, rank, size);

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();

    gather_results_p2p(A_local, b_local, A, b, N, local_n, rank, size);

    if (rank == 0) {
        back_substitution(A, b, x, N);
        printf("Matrix Size (N)=%d, Processors=%d, Execution Time=%f sec\n", N, size, end - start);
    }

    free(A_local);
    free(b_local);
    if (rank == 0) {
        free(A);
        free(b);
        free(x);
    }

    MPI_Finalize();
    return 0;
}
