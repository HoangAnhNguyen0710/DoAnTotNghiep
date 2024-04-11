
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

void Convolution_Calculation_CUDA(char* inputImgName, char* outputImgName, const float* h_kernel,
    int kernel_size, int stride, FILE* outputFile) {
    cv::Mat image = load_image(inputImgName);
    float* h_input = image.ptr<float>(0);
    cudaEvent_t start1, stop1;

    float* d_input = NULL;
    float* d_kernel = NULL;
    float* d_output = NULL;
    int input_height = image.rows;
    int input_width = image.cols;
    int channels = image.channels();

    int output_width = input_width - kernel_size + 3;
    int output_height = input_height - kernel_size + 3;
    long image_bytes = channels * output_height * output_width * sizeof(float);
    float* h_output = new float[image_bytes] { 0 };

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

    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);
     // Launch the convolution
    Convolution << <blocksPerGrid, threadsPerBlock >> > (d_input, d_kernel, d_output, input_width, input_height, kernel_size, 1, output_width, output_height);
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    // Copy output data from GPU to CPU
    cudaMemcpy(h_output, d_output, output_width * output_height * channels * sizeof(float), cudaMemcpyDeviceToHost);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start1, stop1);
    printf("self run duration : %f s\n", milliseconds / 1000);
    //save data to file
    fprintf(outputFile, "%f\n", milliseconds / 1000);
    //save image
    save_image(outputImgName, h_output, output_height, output_width);
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
    FILE* Algo_Gemm_Data_File = fopen("data/Algo_GEMM_1024x1024_input_&3x3filter2.txt", "w+");
    FILE* Algo_Winograd_Data_File = fopen("data/Algo_Winograd_1024x1024_input_&3x3filter2.txt", "w+");
    FILE* Algo_Direct_Data_File = fopen("data/Algo_Direct_1024x1024_input_&3x3filter2.txt", "w+");
    float kernel_template[KERNEL_SIZE][KERNEL_SIZE] = {
        // {0.111111, 0.111111, 0.111111},
        // {0.111111, 0.111111, 0.111111},
        // {0.111111, 0.111111, 0.111111}
        // Laplacian
         {0, 1, 0},
         {1, -4, 1},
         {0, 1, 0}
        // Sharpen
        // {0, -1, 0},
        // {-1, 5, -1},
        // {0, -1, 0}
        // Gauss
        // {1, 2, 1},
        // {2, 4, 2},
        // {1, 2, 1}
        // Sobel by x
        // {-1, 0, 1},
        // {-2, 0, 2},
        // {-1, 0, 1}
        // Sobel by y
        // {-1, -2, -1},
        // {0, 0, 0},
        // {1, 2, 1}
    };
   printf("GEMM impl:\n");
   // Algo GEMM Testing
   for (int i = 0; i < 101; i++) {
       CudnnRuntimeAlgoGemn("input/1024x1024.jpg", "output/1024x1024_Gemm.jpg", kernel_template, Algo_Gemm_Data_File);
   }
   printf("Winograd impl:\n");
   // Algo Winograd Testing
   for (int i = 0; i < 100; i++) {
       CudnnRuntimeAlgoWinograd("input/1024x1024.jpg", "output/1024x1024_Winograd.jpg", kernel_template, Algo_Winograd_Data_File);
   }

    // clang-format 
    float h_kernel[KERNEL_SIZE][KERNEL_SIZE];
    for (int row = 0; row < KERNEL_SIZE; ++row) {
        for (int column = 0; column < KERNEL_SIZE; ++column) {
            h_kernel[row][column] = kernel_template[row][column];
        }
    }
    // self convolution
    float* new_h_kernel = new float[10] {0};
    for (int i = 0; i < KERNEL_SIZE; i++) {
        for (int j = 0; j < KERNEL_SIZE; j++) {
            new_h_kernel[i * KERNEL_SIZE + j] = h_kernel[i][j];
        }
    }
    // direct testing
    printf("Direct impl:\n");
    for (int i = 0; i < 100; i++) {
        Convolution_Calculation_CUDA("input/1024x1024.jpg", "output/1024x1024_Direct.jpg", new_h_kernel, KERNEL_SIZE, 1, Algo_Direct_Data_File);
    }

    fclose(Algo_Direct_Data_File);
    fclose(Algo_Gemm_Data_File);
    fclose(Algo_Winograd_Data_File);
}

