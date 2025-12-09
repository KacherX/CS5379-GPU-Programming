#!/bin/bash
#SBATCH --job-name=MPI_TEST_JOB
#SBATCH --partition=zen4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --time=01:00:00
#SBATCH --output=out.%j
#SBATCH --error=err.%j

module load mpich/4.1.2

#srun ./mpi_hello.exe &
#echo -e "###\n"

srun ./mpi_hello_hostname.exe
echo -e "###\n"

srun ./mpi_send.exe
echo -e "###\n"

srun ./mpi_bcast.exe
echo -e "###\n"

srun ./mpi_matrixmul.exe 2048
echo -e "###\n"
#wait
