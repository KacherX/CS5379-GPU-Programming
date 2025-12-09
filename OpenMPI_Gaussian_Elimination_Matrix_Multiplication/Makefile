CC = gcc
CFLAGS = -Wall -Wextra -O2 -fopenmp
TARGETS = omp-gaussian-elimination omp-gaussian-elimination-v1

all: $(TARGETS)

omp-gaussian-elimination: omp-gaussian-elimination.c
	$(CC) $(CFLAGS) -o $@ $<

omp-gaussian-elimination-v1: omp-gaussian-elimination-v1.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS) *.o
