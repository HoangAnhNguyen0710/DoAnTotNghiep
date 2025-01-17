
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

#define TOTAL_CHANNELS 3
#define KERNEL_SIZE 3
#define BLOCK_SIZE 16 
// #define BLOCK_SIZE 32
#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

float* CudnnRuntimeAlgoGemn(char* imgName, char* outputImg, float kernel_template[][KERNEL_SIZE], FILE* outputFile);

float* CudnnRuntimeAlgoWinograd(char* imgName, char* outputImg, float kernel_template[][KERNEL_SIZE], FILE* outputFile);

float* Self_Direct_Convolution_CUDA(char* inputImgName, char* outputImgName, const float kernel_template[][KERNEL_SIZE],
    const int kernel_size, int stride, FILE* outputFile);

__global__ void im2colKernel(const float* __restrict__ input, float* output, const int input_height, const int input_width, const int kernel_size, const int stride, const int channels);
__global__ void MatrixMultiply(const float* __restrict__ input, float* output, const int input_width, const int output_width, const int kernel_size, const int channels);
float* Self_Gemm_Convolution(char* inputImgName, char* outputImgName, const float kernel_template[][KERNEL_SIZE],
    int kernel_size, int stride, FILE* outputFile);