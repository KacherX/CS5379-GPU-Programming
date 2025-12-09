#include "mpi.h"
#include <stdio.h>
 
int main(int argc, char** argv) {

  MPI_Init(&argc, &argv);
 
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
 
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
 
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
 
  printf("Hello world from node %s, process rank %d"
         " out of %d processes\n",
         processor_name, world_rank, world_size);
 
  MPI_Finalize();
}
