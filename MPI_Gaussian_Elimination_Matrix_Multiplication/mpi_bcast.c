#include "mpi.h"
#include <stdio.h>
 
int main(int argc, char** argv) {

  MPI_Init(&argc, &argv);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
 
  int number=0;
  if (world_rank == 0) {
    number = -1;
  } 
  
  MPI_Bcast(&number, 1, MPI_INT, 0, MPI_COMM_WORLD);

  printf("Process %d received number %d from process 0\n",
            world_rank, number);

  MPI_Finalize();
}
