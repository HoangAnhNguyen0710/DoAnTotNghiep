
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

void CudnnRuntimeAlgoGemn(char* imgName, char* outputImg, float kernel_template[][KERNEL_SIZE]);

void CudnnRuntimeAlgoWinograd(char* imgName, char* outputImg, float kernel_template[][KERNEL_SIZE]);

void Convolution_Calculation_CUDA(float* h_input, const float* h_kernel, float* h_output,
    int input_width, int input_height, int kernel_size, int stride,
    int output_width, int output_height, int channels);