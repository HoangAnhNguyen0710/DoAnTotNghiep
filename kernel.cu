
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

__global__ void Convolution(float* input, const float* kernel, float* output,
    int input_width, int input_height, int kernel_size, int stride,
    int output_width, int output_height);

__global__ void Im2winConvolution(float* input, const float* kernel, float* output,
    int input_width, int input_height, int kernel_size, int stride,
    int output_width, int output_height);

//__constant__ float d_kernel_const[KERNEL_SIZE * KERNEL_SIZE];

__global__ void MatrixMultiply(const float* __restrict__ d_kernel, const float* __restrict__ input, float* output, const int input_width, int kernel_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if (row < kernel_size * kernel_size) {
        for (int i = 0; i < (kernel_size * kernel_size); i++) {
            sum += d_kernel[i] * input[i * input_width + col];
        }
        output[col] = sum;
    }
}

__global__ void im2colKernel(const float* __restrict__ input, float* output, int input_height, int input_width, int kernel_size, int stride) {
    int total_output_cols = (input_width - kernel_size) / stride + 1;
    int total_output_rows = (input_height - kernel_size) / stride + 1;
    int total_cols = total_output_cols * total_output_rows;

    int col_idx = blockIdx.x * blockDim.x + threadIdx.x; // Chỉ số hàng trong ma trận im2col
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y; // Chỉ số cột trong ma trận im2col

    if (row_idx < kernel_size * kernel_size && col_idx < total_cols) {
        int k_row = row_idx / kernel_size; // Dòng trong kernel
        int k_col = row_idx % kernel_size; // Cột trong kernel

        int origin_y = (col_idx % total_output_cols) * stride + k_col;
        int origin_x = (col_idx / total_output_cols) * stride + k_row;

        if (origin_y < input_height && origin_x < input_width) {
            output[row_idx * total_cols + col_idx] = input[origin_x * input_width + origin_y];
            //  printf("in %d\n", input[origin_y * input_width + origin_x]);

        }
        else {
            output[row_idx * total_cols + col_idx] = 0.0f; // Padding with zero if out of bounds
        }
    }
}


void Self_Gemm_Convolution(char* inputImgName, char* outputImgName, const float* h_kernel,
    int kernel_size, int stride, FILE* outputFile) {
    cv::Mat image = load_image(inputImgName);

    float* h_input = image.ptr<float>(0);
    cudaEvent_t start1, stop1;
    float* d_input = NULL;
    float* d_col = NULL;
    float* d_kernel = NULL;
    float* d_output = NULL;
    const int input_height = image.rows;
    const int input_width = image.cols;
    const int channels = image.channels();

    int output_width = input_width - kernel_size + 1;
    int output_height = input_height - kernel_size + 1;
    long image_bytes = channels * output_height * output_width * sizeof(float);
    float* h_output = new float[image_bytes] { 0 };

    float* h_col = new float[(input_width - kernel_size + 1) * (input_width - kernel_size + 1) * kernel_size * kernel_size * channels * sizeof(float)] { 0 };
    int h_col_width = (input_width - kernel_size + 1) * (input_width - kernel_size + 1);
    int h_col_height = kernel_size * kernel_size;
    cudaMalloc((void**)&d_col, (input_width - kernel_size + 1) * (input_width - kernel_size + 1) * kernel_size * kernel_size * channels * sizeof(float));
    cudaMalloc((void**)&d_input, input_width * input_height * channels * sizeof(float));
    cudaMalloc((void**)&d_kernel, kernel_size * kernel_size * sizeof(float));
    cudaMalloc((void**)&d_output, output_width * output_height * channels * sizeof(float));
    // Copy input and kernel data from CPU to GPU
    cudaMemcpy(d_col, h_col, (input_width - kernel_size + 1) * (input_width - kernel_size + 1) * kernel_size * kernel_size * channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, h_input, input_width * input_height * channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, output_width * output_height * channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    //  cudaMemcpyToSymbol(&d_kernel_const, h_kernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);

      // Define grid and block dimensions for GPU computation
    dim3 threadsPerBlock(BLOCK_SIZE, h_col_height);
    dim3 blocksPerGrid((input_width + BLOCK_SIZE - 1) / BLOCK_SIZE, (input_height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int sharedMemSize = (threadsPerBlock.x + KERNEL_SIZE) * (threadsPerBlock.y + KERNEL_SIZE) * sizeof(float);
    dim3 im2ColBlocksPerGrid((h_col_width + BLOCK_SIZE - 1) / BLOCK_SIZE, (h_col_height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 im2ColBlocksPerGrid2((h_col_width + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);
    // Launch the convolution
    im2colKernel << <im2ColBlocksPerGrid, threadsPerBlock >> > (d_input, d_col, input_height, input_width, kernel_size, 1);
    cudaDeviceSynchronize();
    MatrixMultiply << <im2ColBlocksPerGrid2, threadsPerBlock >> > (d_kernel, d_col, d_output, h_col_width, kernel_size);
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


void Convolution_Calculation_CUDA(char* inputImgName, char* outputImgName, const float* h_kernel,
    int kernel_size, int stride, FILE* outputFile) {
    cv::Mat image = load_image(inputImgName);

    float* h_input = image.ptr<float>(0);
    cudaEvent_t start1, stop1;
    float* d_input = NULL;

    float* d_kernel = NULL;
    float* d_output = NULL;
    const int input_height = image.rows;
    const int input_width = image.cols;
    const int channels = image.channels();

    int output_width = input_width - kernel_size + 1;
    int output_height = input_height - kernel_size + 1;
    long image_bytes = channels * output_height * output_width * sizeof(float);
    float* h_output = new float[image_bytes] { 0 };

    cudaMalloc((void**)&d_input, input_width * input_height * channels * sizeof(float));
    cudaMalloc((void**)&d_kernel, kernel_size * kernel_size * sizeof(float));
    cudaMalloc((void**)&d_output, output_width * output_height * channels * sizeof(float));
    // Copy input and kernel data from CPU to GPU
    cudaMemcpy(d_input, h_input, input_width * input_height * channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, output_width * output_height * channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpyToSymbol(&d_kernel_const, h_kernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions for GPU computation
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((input_width + BLOCK_SIZE - 1) / BLOCK_SIZE, (input_height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    // int sharedMemSize = (threadsPerBlock.x + KERNEL_SIZE) * (threadsPerBlock.y + KERNEL_SIZE) * sizeof(float);
    
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);
     // Launch the convolution
    Convolution << <blocksPerGrid, threadsPerBlock >> > (d_input, d_kernel, d_output, input_width, input_height, kernel_size, 1, output_width, output_height);
    cudaDeviceSynchronize();
    // Im2winConvolution << <blocksPerGrid, threadsPerBlock, sharedMemSize >> > (d_input, d_kernel, d_output, input_width, input_height, kernel_size, 1, output_width, output_height);
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

__global__ void Im2winConvolution(float* input, const float* kernel, float* output,
    int input_width, int input_height, int kernel_size, int stride,
    int output_width, int output_height)
{
    // PENDING
}

__global__ void Convolution(float* input, const float* kernel, float* output,
        int input_width, int input_height, int kernel_size, int stride,
        int output_width, int output_height)
    {
    extern __shared__ float tile[];

    // int col_in_block = threadIdx.x;
    // int row_in_block = threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // int row_i = row - kernel_size / 2;
    // int col_i = col - kernel_size / 2;

    int pixelIndex = row * output_width + col;
    float sum = 0.0f;

    // Iterate over the kernel
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            int currentRow = row + i - kernel_size/2;
            int currentCol = col + j - kernel_size/2;

            // Check boundary conditions
            if (currentRow >= 0 && currentRow < input_height && currentCol >= 0 && currentCol < input_width) {
                int currentPixelIndex = currentRow * input_width + currentCol;
                sum += kernel[i * kernel_size + j] * input[currentPixelIndex];
            }
        }
    }

    output[pixelIndex] = sum;
}




int main(int argc, const char* argv[]) {
    FILE* Algo_Gemm_Data_File = fopen("data/Algo_GEMM_1024x1024_input_&4x4filter.txt", "w+");
    FILE* Algo_Winograd_Data_File = fopen("data/Algo_Winograd_128x128_input_&3x3filter.txt", "w+");
    FILE* Algo_Direct_Data_File = fopen("data/Algo_Direct_1024x1024_input_&4x4filter.txt", "w+");
    float kernel_template[KERNEL_SIZE][KERNEL_SIZE] = {
        //Emboss
      {0, 1, 1, 0},
      {1, -2, -2, 1},
      {1, -2, -2, 1},
      {0, 1, 1, 0}
        // {0.111111, 0.111111, 0.111111},
        // {0.111111, 0.111111, 0.111111},
        // {0.111111, 0.111111, 0.111111}
        // Laplacian
      //  {0, 1, 0},
      //  {1, -4, 1},
      //  {0, 1, 0}
        // Sharpen
        // {0, -1, 0},
        // {-1, 5, -1},
        // {0, -1, 0}
        // Gauss
        // {1, 2, 1},
        // {2, 4, 2},
        // {1, 2, 1}
        // Sobel by x
         //{-1, 0, 1},
        // {-2, 0, 2},
        // {-1, 0, 1}
        // Sobel by y
        // {-1, -2, -1},
        // {0, 0, 0},
        // {1, 2, 1}
    };
   //printf("GEMM impl:\n");
   // Algo GEMM Testing

  // printf("Winograd impl:\n");
   // Algo Winograd Testing
  // for (int i = 0; i < 50; i++) {
   // CudnnRuntimeAlgoWinograd("input/128x128.jpg", "output/128x128_Winograd.jpg", kernel_template, Algo_Winograd_Data_File);
  // }

    // clang-format 
    float h_kernel[KERNEL_SIZE][KERNEL_SIZE];
    for (int row = 0; row < KERNEL_SIZE; ++row) {
        for (int column = 0; column < KERNEL_SIZE; ++column) {
            h_kernel[row][column] = kernel_template[row][column];
        }
    }
    // self convolution
    float* new_h_kernel = new float[KERNEL_SIZE * KERNEL_SIZE] {0};
    for (int i = 0; i < KERNEL_SIZE; i++) {
        for (int j = 0; j < KERNEL_SIZE; j++) {
            new_h_kernel[i * KERNEL_SIZE + j] = h_kernel[i][j];
        }
    }

    // direct testing
   // printf("Direct impl:\n");
    for (int i = 0; i < 50; i++) {
        CudnnRuntimeAlgoGemn("input/1024x1024.jpg", "output/1024x1024_Gemm.jpg", kernel_template, Algo_Gemm_Data_File);
    }
    for (int i = 0; i < 51; i++) {
   //     printf("%d\n", i);
        Convolution_Calculation_CUDA("input/1024x1024.jpg", "output/1024x1024_Direct.jpg", new_h_kernel, KERNEL_SIZE, 1, Algo_Direct_Data_File);
    }
    fclose(Algo_Direct_Data_File);
    fclose(Algo_Gemm_Data_File);
    fclose(Algo_Winograd_Data_File);
}

