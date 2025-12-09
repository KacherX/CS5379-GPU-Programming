CC = mpicc
EXE0 = mpi_hello.exe
EXE1 = mpi_send.exe
EXE2 = mpi_bcast.exe
EXE3 = mpi_hello_hostname.exe
EXE4 = mpi_matrixmul.exe
EXE5 = linear_eq_point_to_point.exe
EXE6 = linear_eq_collective.exe 

%.exe : %.o
	$(CC) -o $@ $^

%.o : %.c
	$(CC) -c -o $@ $^

all: $(EXE0) $(EXE1) $(EXE2) $(EXE3) $(EXE4) $(EXE5) $(EXE6)

clean:
	rm -f *.exe *.o
