#ifndef CUDA_ERROR_CHECK_H
#define CUDA_ERROR_CHECK_H

#include <iostream>
#include <cstdlib>

// CUDA error checking macro
#define CUDA_CHECK(call) \
  do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                << " - " << cudaGetErrorString(error) << std::endl; \
      exit(EXIT_FAILURE); \
    } \
  } while(0)

#endif // CUDA_ERROR_CHECK_H