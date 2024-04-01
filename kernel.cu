
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
#include "CudaCustomFunc.h"
#include "ImageInOut.h"

#define BLOCK_SIZE 32
__global__ void Convolution(float* input, const float* kernel, float* output,
    int input_width, int input_height, int kernel_size, int stride,
    int output_width, int output_height);

void Convolution_Calculation_CUDA(float* h_input, const float* h_kernel, float* h_output,
    int input_width, int input_height, int kernel_size, int stride,
    int output_width, int output_height, int channels) {

    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);
    float* d_input = NULL;
    float* d_kernel = NULL;
    float* d_output = NULL;
    cudaMalloc((void**)&d_input, input_width * input_height * channels * sizeof(float));
    cudaMalloc((void**)&d_kernel, kernel_size * kernel_size * sizeof(float));
    cudaMalloc((void**)&d_output, output_width * output_height * channels * sizeof(float));
    // Copy input and kernel data from CPU to GPU
    cudaMemcpy(d_input, h_input, input_width * input_height * channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, output_width * output_height * channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions for GPU computation
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((input_width + BLOCK_SIZE - 1) / BLOCK_SIZE, (input_height + BLOCK_SIZE - 1) / BLOCK_SIZE);

     // Launch the kernel
    Convolution << <blocksPerGrid, threadsPerBlock >> > (d_input, d_kernel, d_output, input_width, input_height, kernel_size, 1, output_width, output_height);

    // Copy output data from GPU to CPU
    cudaMemcpy(h_output, d_output, output_width * output_height * channels * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start1, stop1);
    printf("self run duration : %f s\n", milliseconds / 1000);
    printf("self h_output");

    save_image("self_convolution.jpg", h_output, output_height, output_width);
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    return;
}




__global__ void Convolution(float* input, const float* kernel, float* output,
    int input_width, int input_height, int kernel_size, int stride,
    int output_width, int output_height)
{

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < input_width && row < input_height) {
        int pixelIndex = row * input_width + col;
        float sum = 0.0f;

        // Iterate over the kernel
        for (int i = 0; i < kernel_size; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                int currentRow = row - kernel_size / 2 + i;
                int currentCol = col - kernel_size / 2 + j;

                // Check boundary conditions
                if (currentRow >= 0 && currentRow < input_height && currentCol >= 0 && currentCol < input_width) {
                    int currentPixelIndex = currentRow * input_width + currentCol;
                    sum += kernel[i * kernel_size + j] * input[currentPixelIndex];
                }
            }
        }
        output[pixelIndex] = sum;
    }
}


int main(int argc, const char* argv[]) {

    float kernel_template[KERNEL_SIZE][KERNEL_SIZE] = {
        // {0.111111, 0.111111, 0.111111},
        // {0.111111, 0.111111, 0.111111},
        // {0.111111, 0.111111, 0.111111}
        //  {0, -1, 0},
        // {-1, 5, -1},
        // {0, -1, 0}
         {0, 1, 0},
         {1, -0.5, 1},
         {0, 1, 0}

    };

   // Algo GEMM Testing
   CudnnRuntimeAlgoGemn("input/Test_Image4.jpg", "output/Test_Image4.jpg", kernel_template);
   CudnnRuntimeAlgoGemn("output/Test_Image4.jpg", "output/Test_Image4.jpg", kernel_template);
   CudnnRuntimeAlgoGemn("output/Test_Image4.jpg", "output/Test_Image4.jpg", kernel_template);
   CudnnRuntimeAlgoGemn("output/Test_Image4.jpg", "output/Test_Image4.jpg", kernel_template);
   CudnnRuntimeAlgoGemn("output/Test_Image4.jpg", "output/Test_Image4.jpg", kernel_template);
   CudnnRuntimeAlgoGemn("output/Test_Image4.jpg", "output/Test_Image4.jpg", kernel_template);
   CudnnRuntimeAlgoGemn("output/Test_Image4.jpg", "output/Test_Image4.jpg", kernel_template);
   CudnnRuntimeAlgoGemn("output/Test_Image4.jpg", "output/Test_Image4.jpg", kernel_template);
   CudnnRuntimeAlgoGemn("output/Test_Image4.jpg", "output/Test_Image4.jpg", kernel_template);
   CudnnRuntimeAlgoGemn("output/Test_Image4.jpg", "output/Test_Image4.jpg", kernel_template);
   CudnnRuntimeAlgoGemn("output/Test_Image4.jpg", "output/Test_Image4.jpg", kernel_template);
   CudnnRuntimeAlgoGemn("output/Test_Image4.jpg", "output/Test_Image4.jpg", kernel_template);
   CudnnRuntimeAlgoGemn("output/Test_Image4.jpg", "output/Test_Image4.jpg", kernel_template);
   CudnnRuntimeAlgoGemn("output/Test_Image4.jpg", "output/Test_Image4.jpg", kernel_template);
   CudnnRuntimeAlgoGemn("output/Test_Image4.jpg", "output/Test_Image4.jpg", kernel_template);
   // Algo Winograd Testing
   CudnnRuntimeAlgoWinograd("input/Test_Image4.jpg", "output/Test_Image4.jpg", kernel_template);
   CudnnRuntimeAlgoWinograd("output/Test_Image4.jpg", "output/Test_Image4.jpg", kernel_template);
   CudnnRuntimeAlgoWinograd("output/Test_Image4.jpg", "output/Test_Image4.jpg", kernel_template);
   CudnnRuntimeAlgoWinograd("output/Test_Image4.jpg", "output/Test_Image4.jpg", kernel_template);
   CudnnRuntimeAlgoWinograd("output/Test_Image4.jpg", "output/Test_Image4.jpg", kernel_template);
   CudnnRuntimeAlgoWinograd("output/Test_Image4.jpg", "output/Test_Image4.jpg", kernel_template);
   CudnnRuntimeAlgoWinograd("output/Test_Image4.jpg", "output/Test_Image4.jpg", kernel_template);
   CudnnRuntimeAlgoWinograd("output/Test_Image4.jpg", "output/Test_Image4.jpg", kernel_template);
   CudnnRuntimeAlgoWinograd("output/Test_Image4.jpg", "output/Test_Image4.jpg", kernel_template);
   CudnnRuntimeAlgoWinograd("output/Test_Image4.jpg", "output/Test_Image4.jpg", kernel_template);
   CudnnRuntimeAlgoWinograd("output/Test_Image4.jpg", "output/Test_Image4.jpg", kernel_template);
   CudnnRuntimeAlgoWinograd("output/Test_Image4.jpg", "output/Test_Image4.jpg", kernel_template);
   CudnnRuntimeAlgoWinograd("output/Test_Image4.jpg", "output/Test_Image4.jpg", kernel_template);
   CudnnRuntimeAlgoWinograd("output/Test_Image4.jpg", "output/Test_Image4.jpg", kernel_template);
   CudnnRuntimeAlgoWinograd("output/Test_Image4.jpg", "output/Test_Image4.jpg", kernel_template);
   // CudnnRuntimeAlgoGemn("input/Test_Image4.jpg");
    // clang-format 
    float h_kernel[3][3];
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            h_kernel[row][column] = kernel_template[row][column];
        }
    }
    // self convolution
    float* new_h_kernel = new float[10] {0};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            new_h_kernel[i * 3 + j] = h_kernel[i][j];
        }
    }
    /*
    float* new_h_output = new float[image_bytes] { 0 };
    // clock_t start1 = clock();
    Convolution_Calculation_CUDA(pixelData, new_h_kernel, new_h_output, width, height, 3, 1, width, height, channels);
    float milliseconds = 0;
    // printf("self run duration : %f s\n", milliseconds /1000);
    printf("Happy new year 2024 !!!");
    */
}

