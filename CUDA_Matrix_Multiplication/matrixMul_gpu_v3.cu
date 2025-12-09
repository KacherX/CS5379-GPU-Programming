/*
 * MATRIXMUL_GPU.CU - GPU Matrix Multiplication Implementation
 *
 * CS5379 Parallel Processing - GPU Programming Lecture
 * Texas Tech University - Fall 2025
 * Author: Mert SIDE
 *
 * Multiplying two matrices on the GPU
 *
 */

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_error_check.h"

// ------------------------------------------------------------------ GPUmatmul
__global__
void GPUmatmul(int N, double *x, double *y, double *ans) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = 0; i < N; i++) {
    for (int j = index; j < N; j += stride) {
      double sum = 0;
      for (int k = 0; k < N; k++) {
        sum += x[i*N+k] * y[k*N+j];
      }
      ans[i*N+j] += sum;
    }
  }
}

// ---------------------------------------------------------------------- check
bool check(int N, double *ans) {
  for(int i = 0; i < N; i++) {
    for(int j = 0; j < N; j++) {
      if(ans[i*N+j] != 20.0) {
	printf("x = %f\n", ans[i*N+j]);
	return false;
      }
    }
  }
  return true;
}

// ----------------------------------------------------------------------- MAIN
int main(void) {
  // size of matrix
  int N = 1<<10; // binary left-shift: 1 * 2^10 = 1024
  printf("Size of matrix (N) is %d by %d.\n", N, N);
  int iter = 3;
  clock_t t;
  
  // Martices
  double *x, *y, *ans;

  // TODO: Allocate unified Memory - accessible from both CPU and GPU
  // ...
  // ...
  // ...
  CUDA_CHECK(cudaMallocManaged(&x, N*N*sizeof(double)));
  CUDA_CHECK(cudaMallocManaged(&y, N*N*sizeof(double)));
  CUDA_CHECK(cudaMallocManaged(&ans, N*N*sizeof(double)));

  // ..........................................................................
  // initialize x,y and ans arrays on the host
  for (int i = 0; i < N; i++) {
    for(int j = 0; j < N; j++) {
      x[i*N+j] = 5; 
      y[i*N+j] = (i==j?1:0); 
      ans[i*N+j] = (double)0.000000000000;
    }
  }

  // ..........................................................................
  double avg=0;
  std::cout<<"Starting GPU computation with multiple blocks"<<std::endl;
  
  int blockSize = 256;
  int numBlocks = (N*N + blockSize - 1)/blockSize;
  // Run kernel on GPU
  for(int i = 0; i <= iter; i++) {
    t = clock();
    GPUmatmul<<<numBlocks,blockSize>>>(N, x, y,ans);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    t = clock() - t;
    if(i) avg += t; //we will ignore the first run
    // printf ("It took GPU-%d %f ms.\n",i,(((double)t)/CLOCKS_PER_SEC)*1000);
  }

  avg /= iter;
  avg /= CLOCKS_PER_SEC;
  avg *= 1000;
  printf("It took %lf ms on avg.\n", avg);
  if(check(N,ans)) std::cout<<"RUN OK."<<std::endl;
  else std::cout<<"RUN NOT OK."<<std::endl;

  // ..........................................................................
  
  // TODO: Free memory
  // ...
  // ...
  // ...
  CUDA_CHECK(cudaFree(x));
  CUDA_CHECK(cudaFree(y));
  CUDA_CHECK(cudaFree(ans));

  return 0;
}
/* EOF */
