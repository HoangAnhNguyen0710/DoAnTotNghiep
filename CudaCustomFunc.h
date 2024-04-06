
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <cudnn.h>
#include <cassert>
#include <cstdlib>
#include <iostream>

#define KERNEL_SIZE 3
#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

void CudnnRuntimeAlgoGemn(char* imgName, char* outputImg, float kernel_template[][KERNEL_SIZE], FILE* outputFile);

void CudnnRuntimeAlgoWinograd(char* imgName, char* outputImg, float kernel_template[][KERNEL_SIZE], FILE* outputFile);

void Convolution_Calculation_CUDA(char* inputImgName, char* outputImgName, const float* h_kernel,
    int kernel_size, int stride, FILE* outputFile);