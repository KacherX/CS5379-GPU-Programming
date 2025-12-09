/*
 * mpi_pi.c
 * Parallel computation of π using numerical integration.
 * Uses the midpoint rule and MPI to distribute work across processes.
 */

#include "mpi.h"
#include <stdio.h>
#include <math.h>

int main(int argc, char *argv[]) {
    int done = 0, n, myid, numprocs, i;
    double PI25DT = 3.141592653589793238462643;
    double mypi, pi, width, sum, x;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    while (!done) {
        if (myid == 0) {
            printf("Enter the number of intervals: (0 quits) ");
            fflush(stdout);
            if (scanf("%d", &n) != 1) n = 0;
        }

        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (n == 0) break;

        width = 1.0 / (double)n;
        sum = 0.0;

        for (i = myid + 1; i <= n; i += numprocs) {
            x = width * ((double)i - 0.5);
            sum += 4.0 / (1.0 + x * x);
        }

        mypi = width * sum;
        MPI_Reduce(&mypi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (myid == 0) {
            printf("pi is approximately %.16f, Error is %.16f\n",
                   pi, fabs(pi - PI25DT));
        }
    }

    MPI_Finalize();
    return 0;
}
