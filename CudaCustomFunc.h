
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <cudnn.h>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#define KERNEL_SIZE 4
#define BLOCK_SIZE 32
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

__global__ void im2colKernel(const float* __restrict__ input, float* output, int input_height, int input_width, int kernel_size, int stride);
__global__ void MatrixMultiply(const float* __restrict__ d_kernel, const float* __restrict__ input, float* output, const int input_width, int kernel_size);
void Self_Gemm_Convolution(char* inputImgName, char* outputImgName, const float* h_kernel,
    int kernel_size, int stride, FILE* outputFile);