
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
#include "dirent.h"
#include <stdlib.h>
#include "CudaCustomFunc.h"
#include "ImageInOut.h"

__global__ void Convolution(const float* __restrict__  input, float* output,
    int input_width, int input_height, int kernel_size, int stride,
    int output_width, int output_height, int channels);

__global__ void SelfWinogradConvolution(float* input, float* output,
    int input_width, int input_height, int kernel_size, int stride,
    int output_width, int output_height, int channels);

__constant__ float d_kernel_const_gemm[3][3][KERNEL_SIZE * KERNEL_SIZE];

__constant__ float d_kernel_const_direct[3][3][KERNEL_SIZE][KERNEL_SIZE];


__global__ void MatrixMultiply(const float* __restrict__ input, float* output, const int input_width, const int kernel_size, const int channels) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int kernel = 0; kernel < channels; kernel++) {
        float totalSum = 0.0f;
        for (int ch = 0; ch < channels; ch++) {
            float sum = 0.0f;
            if (row < kernel_size * kernel_size) {
                for (int i = 0; i < (kernel_size * kernel_size); i++) {
                    sum += d_kernel_const_gemm[kernel][ch][i] * input[(i * input_width + col) * channels + ch];
                }
                totalSum += sum;
            }
        }
        output[col * channels + kernel] = totalSum;
    }
}

__global__ void im2colKernel(const float* __restrict__ input, float* output, const int input_height, const int input_width, const int kernel_size, const int stride, const int channels) {
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
        for (int ch = 0; ch < channels; ch++){
            if (origin_y < input_height && origin_x < input_width) {
                output[(row_idx * total_cols + col_idx) * channels + ch] = input[(origin_x * input_width + origin_y) * channels + ch];
                //  printf("in %d\n", input[origin_y * input_width + origin_x]);
            }
            else {
                output[(row_idx * total_cols + col_idx) * channels + ch] = 0.0f; // Padding with zero if out of bounds
            }
        }
    }
}


void Self_Gemm_Convolution(char* inputImgName, char* outputImgName, const float kernel_template[][KERNEL_SIZE],
    int kernel_size, int stride, FILE* outputFile) {
    cv::Mat image = load_image(inputImgName);

    size_t beforeFreeBytes, beforeTotalBytes;
    cudaMemGetInfo(&beforeFreeBytes, &beforeTotalBytes);

    float* h_input = image.ptr<float>(0);
    cudaEvent_t start1, stop1;
    float* d_input = NULL;
    float* d_col = NULL;
    // float* d_kernel = NULL;
    float* d_output = NULL;
    const int input_height = image.rows;
    const int input_width = image.cols;
    const int channels = image.channels();

    int output_width = input_width - kernel_size + 1;
    int output_height = input_height - kernel_size + 1;
    long image_bytes = channels * output_height * output_width * sizeof(float);
    float* h_output = new float[image_bytes] { 0 };

    float h_kernel[3][3][KERNEL_SIZE * KERNEL_SIZE];
    for (int kernel = 0; kernel < 3; kernel++) {
        for (int ch = 0; ch < 3; ch++) {
            for (int row = 0; row < kernel_size; row++) {
                for (int column = 0; column < kernel_size; column++) {
                    if (kernel == ch)
                        h_kernel[kernel][ch][row * KERNEL_SIZE + column] = kernel_template[row][column];
                    else h_kernel[kernel][ch][row * KERNEL_SIZE + column] = 0.0f;
                }
            }
        }
    }

    float* h_col = new float[output_height * output_width * kernel_size * kernel_size * channels * sizeof(float)] { 0 };
    int h_col_width = output_height * output_width;
    int h_col_height = kernel_size * kernel_size;
    cudaMalloc((void**)&d_col, output_height * output_width * kernel_size * kernel_size * channels * sizeof(float));
    cudaMalloc((void**)&d_input, input_width * input_height * channels * sizeof(float));
    // cudaMalloc((void**)&d_kernel, kernel_size * kernel_size * sizeof(float));
    cudaMalloc((void**)&d_output, output_width * output_height * channels * sizeof(float));

    // Copy input and kernel data from CPU to GPU
    cudaMemcpy(d_col, h_col, output_height * output_width * kernel_size * kernel_size * channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, h_input, input_width * input_height * channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, output_width * output_height * channels * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_kernel, h_kernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_kernel_const_gemm, h_kernel, sizeof(h_kernel), 0, cudaMemcpyHostToDevice);

    // Define grid and block dimensions for GPU computation
    dim3 threadsPerBlock(BLOCK_SIZE, h_col_height);
    dim3 blocksPerGrid((input_width + BLOCK_SIZE - 1) / BLOCK_SIZE, (input_height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    dim3 im2ColBlocksPerGrid((h_col_width + BLOCK_SIZE - 1) / BLOCK_SIZE, (h_col_height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 im2ColBlocksPerGrid2((h_col_width + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);
    // Launch the convolution
    im2colKernel << <im2ColBlocksPerGrid, threadsPerBlock >> > (d_input, d_col, input_height, input_width, kernel_size, 1, channels);
    cudaDeviceSynchronize();
    MatrixMultiply << <im2ColBlocksPerGrid2, threadsPerBlock >> > (d_col, d_output, h_col_width, kernel_size, channels);


    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);

    /*
    size_t afterFreeBytes, afterTotalBytes;
    cudaMemGetInfo(&afterFreeBytes, &afterTotalBytes);
    size_t usedBytes = beforeFreeBytes - afterFreeBytes;

    std::cout << " Free Memory (MB): " << (afterFreeBytes / 1024.0 / 1024.0) << std::endl;
    std::cout << " Used Memory (MB): " << (usedBytes / 1024.0 / 1024.0) << std::endl;
    */

    //calculate used memory
    
    size_t d_col_mem = output_height * output_width * kernel_size * kernel_size * channels * sizeof(float);
    size_t d_input_mem = input_width * input_height * channels * sizeof(float);
    size_t d_output_mem = output_width * output_height * channels * sizeof(float);
    size_t global_mem = d_col_mem + d_input_mem + d_output_mem;
    size_t const_mem = KERNEL_SIZE * KERNEL_SIZE * 3 * 3 * sizeof(float);

    size_t total_mem = global_mem + const_mem;

    printf("total mem usage: %zu MB\n", total_mem / 1024 / 1024);

    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(cudaError));
    }

    // Copy output data from GPU to CPU
    cudaMemcpy(h_output, d_output, output_width * output_height * channels * sizeof(float), cudaMemcpyDeviceToHost);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start1, stop1);
    printf("self gemm run duration : %f ms\n", milliseconds);

    // save data to file
    fprintf(outputFile, "%f\n", milliseconds);

    // save image
    save_image(outputImgName, h_output, output_height, output_width);

    // Cleanup
    // cudaFree(d_kernel);
    cudaFree(d_col);
    cudaFree(d_input);
    cudaFree(d_output);
    return;
}
const float BT[4][4] = {
    {1, 0, -1, 0},
    {0, 1, 1, 0},
    {0, -1, 1, 0},
    {0, 1, 0, -1}
};

const float B[4][4] = {
    {1, 0, 0, 0},
    {0, 1, -1, 1},
    {-1, 1, 1, 0},
    {0, 0, 0, -1}
};

const float A[4][2] = {
    {1, 0,},
    {1, 1},
    {1, -1},
    {0, -1}
};

const float AT[2][4] = {
    {1, 1, 1, 0},
    {0, 1, -1, -1},
};

const float G[4][3] = {
    {1, 0, 0},
    {0.5, 0.5, 0.5},
    {0.5, -0.5, 0.5},
    {0, 0, 1}
};

const float Gt[3][4] = {
    {1, 0.5, 0.5, 0},
    {0, 0.5, -0.5, 0},
    {0, 0.5, 0.5, 1}
};


// Khai báo các ma trận cần sao chép vào constant

// Khai báo ma trận BT là constant
__constant__ float BT_constant[4][4];
// Khai báo ma trận B là constant
__constant__ float B_constant[4][4];
// Khai báo ma trận A là constant
__constant__ float A_constant[4][2];
// Khai báo ma trận AT là constant
__constant__ float AT_constant[2][4];
// Khai báo ma trận G là constant
__constant__ float G_constant[4][3];
// Khai báo ma trận Gt là constant
__constant__ float Gt_constant[3][4];
// Khai báo ma trận biến đổi U của kernel (U = G * F * GT)
__constant__ float U_constant[3][3][4][4];

void Self_Direct_Convolution_CUDA(char* inputImgName, char* outputImgName, const float kernel_template[][KERNEL_SIZE],
    const int kernel_size, int stride, FILE* outputFile) {
    cv::Mat image = load_image(inputImgName);
    /*
    size_t beforeFreeBytes, beforeTotalBytes;
    cudaMemGetInfo(&beforeFreeBytes, &beforeTotalBytes);
    */
    float* h_input = image.ptr<float>(0);
    
    cudaEvent_t start1, stop1;
    float* d_input = NULL;
        
    // float* d_kernel = NULL;
    float* d_output = NULL;
    const int input_height = image.rows;
    const int input_width = image.cols;
    const int channels = image.channels();

    int output_width = input_width - kernel_size + 1;
    int output_height = input_height - kernel_size + 1;

    long image_bytes = channels * output_height * output_width * sizeof(float);
    float* h_output = new float[image_bytes] { 0 };

    float h_kernel[3][3][KERNEL_SIZE][KERNEL_SIZE];
    for (int kernel = 0; kernel < 3; kernel++) {
        for (int ch = 0; ch < 3; ch++) {
            for (int row = 0; row < kernel_size; row++) {
                for (int column = 0; column < kernel_size; column++) {
                    if (kernel == ch)
                        h_kernel[kernel][ch][row][column] = kernel_template[row][column];
                    else h_kernel[kernel][ch][row][column] = 0.0f;
                }
            }
        }
    }

    cudaMalloc((void**)&d_input, input_width * input_height * channels * sizeof(float));
    // cudaMalloc((void**)&d_kernel, kernel_size * kernel_size * sizeof(float));
    cudaMalloc((void**)&d_output, output_width * output_height * channels * sizeof(float));
    // Copy input and kernel data from CPU to GPU
    cudaMemcpy(d_input, h_input, input_width * input_height * channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, output_width * output_height * channels * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_kernel, h_kernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_kernel_const_direct, h_kernel, sizeof(h_kernel), 0, cudaMemcpyHostToDevice);

    // Define grid and block dimensions for GPU computation
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((input_width + BLOCK_SIZE - 1) / BLOCK_SIZE, (input_height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    // int sharedMemSize = (threadsPerBlock.x + KERNEL_SIZE) * (threadsPerBlock.y + KERNEL_SIZE) * sizeof(float);
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);
     // Launch the convolution
    Convolution << <blocksPerGrid, threadsPerBlock >> > (d_input, d_output, input_width, input_height, kernel_size, 1, output_width, output_height, channels);
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(cudaError));
        // Thực hiện các xử lý khác hoặc thoát khỏi chương trình nếu cần
    }
    cudaDeviceSynchronize();
    // Im2winConvolution << <blocksPerGrid, threadsPerBlock, sharedMemSize >> > (d_input, d_kernel, d_output, input_width, input_height, kernel_size, 1, output_width, output_height);
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    /*
    size_t afterFreeBytes, afterTotalBytes;
    cudaMemGetInfo(&afterFreeBytes, &afterTotalBytes);
    size_t usedBytes = beforeFreeBytes - afterFreeBytes;

    std::cout << " Free Memory (MB): " << (afterFreeBytes / 1024.0 / 1024.0) << std::endl;
    std::cout << " Used Memory (MB): " << (usedBytes / 1024.0 / 1024.0) << std::endl;
    */

    size_t d_input_mem = input_width * input_height * channels * sizeof(float);
    size_t d_output_mem = output_width * output_height * channels * sizeof(float);
    size_t global_mem = d_input_mem + d_output_mem;
    size_t const_mem = (KERNEL_SIZE * KERNEL_SIZE * 3 * 3) * sizeof(float);

    size_t total_mem = global_mem + const_mem;

    printf("total mem usage: %zu MB\n", total_mem / 1024 / 1024 );
    

    // Copy output data from GPU to CPU
    cudaMemcpy(h_output, d_output, output_width * output_height * channels * sizeof(float), cudaMemcpyDeviceToHost);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start1, stop1);
    printf("self direct run duration : %f ms\n", milliseconds);
    //save data to file
    fprintf(outputFile, "%f\n", milliseconds);
    //save image
    save_image(outputImgName, h_output, output_height, output_width);
    // printf("size %d\n", output_width * output_height * channels * sizeof(float));
    // Cleanup
    cudaFree(d_input);
    // cudaFree(d_kernel);
    cudaFree(d_output);
    return;
}



void Self_Winograd_Convolution_CUDA(char* inputImgName, char* outputImgName, const float kernel_template[][KERNEL_SIZE],
    int kernel_size, int stride, FILE* outputFile) {
    cv::Mat image = load_image(inputImgName);
    /*
    size_t freeBytes, totalBytes;
    cudaMemGetInfo(&freeBytes, &totalBytes);
    */

    float F[3][3]; // F
    float GF[4][3]; // G * F 
    float U[3][3][4][4]; // G * F * GT

    float* h_input = image.ptr<float>(0);
    float* h_input_pinned;

    cudaEvent_t start1, stop1;
    float* d_input = NULL;
    
    // float* d_kernel = NULL;
    float* d_output = NULL;
    const int input_height = image.rows;
    const int input_width = image.cols;
    const int channels = image.channels();
    
    int output_width = input_width - kernel_size + 1;
    int output_height = input_height - kernel_size + 1;
    long image_bytes = channels * output_height * output_width * sizeof(float);
    float* h_output = new float[image_bytes] { 0 };
    
    cudaMalloc((void**)&d_input, input_width * input_height * channels * sizeof(float));
    cudaMalloc((void**)&d_output, output_width * output_height * channels * sizeof(float));
    // Copy input and kernel data from CPU to GPU
    
    // cudaMallocHost((void**)&h_input_pinned, image.cols * image.rows * image.channels() * sizeof(float));
    // memcpy(h_input_pinned, h_input, image.cols * image.rows * image.channels() * sizeof(float));

    cudaMemcpy(d_input, h_input, input_width * input_height * channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, output_width * output_height * channels * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions for GPU computation
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((output_width + BLOCK_SIZE - 1) /( 2 * BLOCK_SIZE ) + 1, (output_height + BLOCK_SIZE - 1) / ( 2 * BLOCK_SIZE) + 1);
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);

    //transform h_kernel
    for (int kernel = 0; kernel < channels; kernel++) {
        for (int ch = 0; ch < channels; ch++) {
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        F[i][j] = kernel_template[i][j];
                        // if(row == 1 && col == 1) printf("%d %d %f\n", i, j, F[i][j]);
                    }
                }
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 3; j++) {
                        GF[i][j] = 0.0f;
                        for (int k = 0; k < 3; ++k) {
                            GF[i][j] += G[i][k] * F[k][j];
                            // printf("Z %d %d = %f\n", i, j, Z[i][j]);
                        }
                    }
                }
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        U[kernel][ch][i][j] = 0.0f;
                        if (kernel == ch) {
                            for (int k = 0; k < 3; ++k) {
                                U[kernel][ch][i][j] += GF[i][k] * Gt[k][j];
                                // printf("Z %d %d = %f\n", i, j, Z[i][j]);
                            }
                        }
                    }
                }
            
        }
    }

    cudaMemcpyToSymbol(BT_constant, BT, sizeof(float) * 4 * 4);
    cudaMemcpyToSymbol(B_constant, B, sizeof(float) * 4 * 4);
    cudaMemcpyToSymbol(A_constant, A, sizeof(float) * 4 * 2);
    cudaMemcpyToSymbol(AT_constant, AT, sizeof(float) * 2 * 4);
    cudaMemcpyToSymbol(G_constant, G, sizeof(float) * 4 * 3);
    cudaMemcpyToSymbol(Gt_constant, Gt, sizeof(float) * 3 * 4);
    cudaMemcpyToSymbol(U_constant, U, sizeof(float) * 4 * 4 * 3 * 3);
    // Launch the convolution

    SelfWinogradConvolution << <blocksPerGrid, threadsPerBlock >> > (d_input, d_output, input_width, input_height, kernel_size, 1, output_width, output_height, channels);

    cudaDeviceSynchronize();

    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);

    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(cudaError));
        // Thực hiện các xử lý khác hoặc thoát khỏi chương trình nếu cần
    }
    
    /*
    size_t currentFreeBytes, currentTotalBytes;
    cudaMemGetInfo(&currentFreeBytes, &currentTotalBytes);
    size_t usedBytes = freeBytes - currentFreeBytes;

    std::cout << " Total Memory (MB): " << (totalBytes / 1024.0 / 1024.0) << std::endl;
    std::cout << " Current Free Memory (MB): " << (currentFreeBytes / 1024.0 / 1024.0) << std::endl;
    std::cout << " Used Memory (MB): " << (usedBytes / 1024.0 / 1024.0) << std::endl;
    */

    
    size_t d_input_mem = input_width * input_height * channels * sizeof(float);
    size_t d_output_mem = output_width * output_height * channels * sizeof(float);
    size_t global_mem = d_input_mem + d_output_mem;
    size_t const_mem = (4 * 4 + 4 * 4 + 4 * 2 + 2 * 4 + 4 * 3 + 3 * 4 + 4 * 4 * 3 * 3) * sizeof(float);

    size_t total_mem = global_mem + const_mem;


    printf("total mem usage: %zu MB\n", total_mem / 1024 / 1024);
    

    // Copy output data from GPU to CPU
    cudaMemcpy(h_output, d_output, output_width * output_height * channels * sizeof(float), cudaMemcpyDeviceToHost);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start1, stop1);
    printf("self winograd run duration : %f ms\n", milliseconds);
    //save data to file
    fprintf(outputFile, "%f\n", milliseconds);
    //save image
    save_image(outputImgName, h_output, output_height, output_width);
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    return;
}


__global__ void SelfWinogradConvolution(float* input, float* output,
    int input_width, int input_height, int kernel_size, int stride,
    int output_width, int output_height,int channels)
{
    // PENDING
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    int row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;

    float Win4x4[4][4]; // I
    float New4x4[4][4]; // BT * I
    float V[4][4]; // BT * I * B
    float M[4][4]; // U (*) V
    float ATM[2][4];
    float Out[2][2];
    for (int kernel = 0; kernel < channels; kernel++) {
        output[(row * output_width + col) * channels + kernel] = 0.0f;
        output[(row * output_width + col + 1) * channels + kernel] = 0.0f;
        output[((row + 1) * output_width + col) * channels + kernel] = 0.0f;
        output[((row + 1) * output_width + col + 1) * channels + kernel] = 0.0f;
        for (int ch = 0; ch < channels; ch++) {


            if (col < output_width && row < output_height && col > 0 && row > 0) {


                for (int i = -1; i < 3; i++) {
                    for (int j = -1; j < 3; j++) {
                        if (row + i > input_width) {
                            Win4x4[i + 1][j + 1] = 0.0f;
                        }
                        else if (col + j > input_height) {
                            Win4x4[i + 1][j + 1] = 0.0f;
                        }
                        else
                        Win4x4[i + 1][j + 1] = input[((row + i) * input_width + col + j) * channels + ch];
                        // if (row == 100 && col == 100) printf("%d %d %f\n", row + i, col + j, Win4x4[i + 1][j + 1]);
                    }
                }
                // V =  BT*I*B

                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        New4x4[i][j] = 0.0f;
                        for (int k = 0; k < 4; ++k) {
                            New4x4[i][j] += BT_constant[i][k] * Win4x4[k][j];
                            // printf("Z %d %d = %f\n", i, j, Z[i][j]);
                        }
                    }
                }

                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        V[i][j] = 0.0f;
                        for (int k = 0; k < 4; k++) {
                            V[i][j] += New4x4[i][k] * B_constant[k][j];
                            // printf("Z %d %d = %f\n", i, j, Z[i][j]);
                        }
                    }
                }
                // U = G*F*Gt

                // M = U (*) V
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        M[i][j] = U_constant[kernel][ch][i][j] * V[i][j];
                    }
                }

                // O = At * M * A
                for (int i = 0; i < 2; i++) {
                    for (int j = 0; j < 4; j++) {
                        ATM[i][j] = 0.0f;
                        for (int k = 0; k < 4; k++) {
                            ATM[i][j] += AT_constant[i][k] * M[k][j];
                            // printf("Z %d %d = %f\n", i, j, Z[i][j]);
                        }
                    }
                }

                for (int i = 0; i < 2; i++) {
                    for (int j = 0; j < 2; j++) {
                        Out[i][j] = 0.0f;
                        for (int k = 0; k < 4; k++) {
                            Out[i][j] += ATM[i][k] * A_constant[k][j];
                        }
                    }
                }
                for (int i = 0; i < 2; i++) {
                    for (int j = 0; j < 2; j++) {
                        output[((row + i) * output_width + col + j) * channels + kernel] += Out[i][j];
                    }
                }
            }
        }
    }
}

__global__ void Convolution(const float* __restrict__  input, float* output,
        int input_width, int input_height, int kernel_size, int stride,
        int output_width, int output_height, int channels)
    {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= output_width || row >= output_height) {
        return;
    }

    int kernel_radius = kernel_size / 2;

    for (int kernel = 0; kernel < channels; kernel++) {
        float totalSum = 0.0f;
        for (int ch = 0; ch < channels; ch++) {
            float sum = 0.0f;
            for (int i = -kernel_radius; i <= kernel_radius; i++) {
                for (int j = -kernel_radius; j <= kernel_radius; j++) {
                    int currentRow = row + i;
                    int currentCol = col + j;

                    // Check boundary conditions
                    if (currentRow >= 0 && currentRow < input_height && currentCol >= 0 && currentCol < input_width) {
                        int currentPixelIndex = (currentRow * input_width + currentCol) * channels + ch;
                       // int kernelIndex = (i + kernel_radius) * kernel_size + (j + kernel_radius);
                        sum += d_kernel_const_direct[kernel][ch][(i + kernel_radius)][(j + kernel_radius)] * input[currentPixelIndex];
                    }
                }
            }
            totalSum += sum;
        }
        int outputPixelIndex = (row * output_width + col) * channels + kernel;
        output[outputPixelIndex] = totalSum;
    }
}


int main(int argc, const char* argv[]) {

    cudaDeviceProp device;
    cudaGetDeviceProperties(&device, 0);

    printf("Device name: %s\n", device.name);
    printf("Device max threads per blocks: %d\n", device.maxThreadsPerBlock);
    printf("Device max threads per multi processor: %d\n", device.maxThreadsPerMultiProcessor);
    printf("Device max block per multi processor %d\n", device.maxBlocksPerMultiProcessor);
    printf("Device num of multi processor: %d\n", device.multiProcessorCount);
    printf("Device total constant memory: %zu bytes\n", device.totalConstMem);
    printf("Device total L2 cache size: %zu bytes\n", device.l2CacheSize);
    printf("Device mem pitch: %zu bytes\n", device.memPitch);
    printf("Device warp size in thread: %d\n\n", device.warpSize);
    // calculate the memory before using kernel
    size_t freeBytes, totalBytes;
    cudaMemGetInfo(&freeBytes, &totalBytes);
    size_t usedBytes = totalBytes - freeBytes;

    std::cout << " Total Memory (MB): " << (totalBytes / 1024.0 / 1024.0) << std::endl;
    std::cout << " Free Memory (MB): " << (freeBytes / 1024.0 / 1024.0) << std::endl;
    std::cout << " Used Memory (MB): " << (usedBytes / 1024.0 / 1024.0) << std::endl;

    // cifar_10 data
    /*
    FILE* Algo_Gemm_Data_File = fopen("data/Algo_GEMM_cifar_10_input_&3x3filter.txt", "a+");
    FILE* Algo_Winograd_Data_File = fopen("data/Algo_Winograd_cifar_10_input_&3x3filter.txt", "a+");
    FILE* Self_Winograd_Data_File = fopen("data/Self_Winograd_cifar_10_input_&3x3filter.txt", "a+");
    FILE* Algo_Direct_Data_File = fopen("data/Algo_Direct_cifar_10_input_&3x3filter.txt", "a+");
    FILE* Self_Gemm_Data_File = fopen("data/Self_Gemm_cifar_10_input_&3x3filter.txt", "a+");
    */
    // tampere_17 file
    /*
    FILE* Algo_Gemm_Data_File = fopen("data/Algo_GEMM_tampere_17_input_&3x3filter.txt", "a+");
    FILE* Algo_Winograd_Data_File = fopen("data/Algo_Winograd_tampere_17_input_&3x3filter.txt", "a+");
    FILE* Self_Winograd_Data_File = fopen("data/Self_Winograd_tampere_17_input_&3x3filter.txt", "a+");
    FILE* Algo_Direct_Data_File = fopen("data/Algo_Direct_tampere_17_input_&3x3filter.txt", "a+");
    FILE* Self_Gemm_Data_File = fopen("data/Self_Gemm_tampere_17_input_&3x3filter.txt", "a+");
    */
    // mnist file
    
    FILE* Algo_Gemm_Data_File = fopen("data/Algo_GEMM_mnist_input_&3x3filter.txt", "a+");
    FILE* Algo_Winograd_Data_File = fopen("data/Algo_Winograd_mnist_input_&3x3filter.txt", "a+");
    FILE* Self_Winograd_Data_File = fopen("data/Self_Winograd_mnist_input_&3x3filter.txt", "a+");
    FILE* Algo_Direct_Data_File = fopen("data/Algo_Direct_mnist_input_&3x3filter.txt", "a+ ");
    FILE* Self_Gemm_Data_File = fopen("data/Self_Gemm_mnist_input_&3x3filter.txt", "a+");
    
    // MLRS_Net file
    /*
    FILE* Algo_Gemm_Data_File = fopen("data/Algo_GEMM_MLRS_Net_input_&3x3filter.txt", "a+");
    FILE* Algo_Winograd_Data_File = fopen("data/Algo_Winograd_MLRS_Net_input_&3x3filter.txt", "a+");
    FILE* Self_Winograd_Data_File = fopen("data/Self_Winograd_MLRS_Net_input_&3x3filter.txt", "a+");
    FILE* Algo_Direct_Data_File = fopen("data/Algo_Direct_MLRS_Net_input_&3x3filter.txt", "a+");
    FILE* Self_Gemm_Data_File = fopen("data/Self_Gemm_MLRS_Net_input_&3x3filter.txt", "a+");
    */
    float kernel_template[KERNEL_SIZE][KERNEL_SIZE] = {
        //Emboss
      //{0, 1, 1, 0},
      //{1, -2, -2, 1},
      //{1, -2, -2, 1},
      //{0, 1, 1, 0}
        // Blur
        // {0.111111, 0.111111, 0.111111},
        //  {0.111111, 0.111111, 0.111111},
        // {0.111111, 0.111111, 0.111111}
        //
        // {0, 1, 0},
        // {1, 1, 1},
        /// {0, 1, 0}
        // Laplacian
         {0, 1, 0},
         {1, -4, 1},
         {0, 1, 0}
        // Sharpen
        // {0, -1, 0},
        // {-1, 8, -1},
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

    float h_kernel[KERNEL_SIZE][KERNEL_SIZE];
    for (int row = 0; row < KERNEL_SIZE; ++row) {
        for (int column = 0; column < KERNEL_SIZE; ++column) {
            h_kernel[row][column] = kernel_template[row][column];
        }
    }

    float* new_h_kernel = new float[KERNEL_SIZE * KERNEL_SIZE] {0};
    for (int i = 0; i < KERNEL_SIZE; i++) {
        for (int j = 0; j < KERNEL_SIZE; j++) {
            new_h_kernel[i * KERNEL_SIZE + j] = h_kernel[i][j];
        }
    }

    // Algo GEMM Testing
    printf("GEMM impl:\n");
     for (int i = 0; i < 2; i++) {
      //  CudnnRuntimeAlgoGemn("input/128x128_RGB.jpeg", "output/128x128_GEMM.jpeg", kernel_template, Algo_Gemm_Data_File);
    //CudnnRuntimeAlgoGemn("input/128x128_RGB.jpeg", "output/128x128_GEMM.jpeg", kernel_template, Algo_Gemm_Data_File);
    //  CudnnRuntimeAlgoGemn("input/1024x1024_RGB.jpeg", "output/1024x1024_GEMM.jpeg", kernel_template, Algo_Gemm_Data_File);
    //  CudnnRuntimeAlgoGemn("input/256x256(2).png", "output/256x256(2)_GEMM.png", kernel_template, Algo_Gemm_Data_File);
    }
  
    // Algo Winograd Testing
    printf("Winograd impl:\n");
    for (int i = 0; i < 2; i++) {
       // CudnnRuntimeAlgoWinograd("input/128x128_RGB.jpeg", "output/128x128_Winograd.jpeg", kernel_template, Algo_Winograd_Data_File);
    //  CudnnRuntimeAlgoWinograd("input/1024x1024_RGB.jpeg", "output/1024x1024_Winograd.jpeg", kernel_template, Algo_Winograd_Data_File);
     // CudnnRuntimeAlgoWinograd("input/256x256(2).png", "output/256x256(2)_Winograd.png", kernel_template, Algo_Winograd_Data_File);
    }
   
     // Direct
     printf("Direct impl:\n");
    for (int i = 0; i < 2; i++) {
       // Self_Direct_Convolution_CUDA("input/128x128_RGB.jpeg", "output/128x128_Direct.jpeg", kernel_template, KERNEL_SIZE, 1, Algo_Direct_Data_File);
     //Self_Direct_Convolution_CUDA("input/1024x1024_RGB.jpeg", "output/1024x1024_Direct.png", kernel_template, KERNEL_SIZE, 1, Algo_Direct_Data_File);
     // Self_Direct_Convolution_CUDA("input/256x256(2).png", "output/256x256(2)_Direct.png", new_h_kernel, KERNEL_SIZE, 1, Algo_Direct_Data_File);
    }
     
    // Self Winograd
    printf("Self Winograd impl:\n");
    for (int i = 0; i < 2; i++) {
        //  Self_Winograd_Convolution_CUDA("input/256x256(2).png", "output/256x256(2)_Self_Winograd.png", new_h_kernel, KERNEL_SIZE, 1, Self_Winograd_Data_File);
       // Self_Winograd_Convolution_CUDA("input/1024x1024_RGB.jpeg", "output/1024x1024_Self_Winograd.jpeg", kernel_template, KERNEL_SIZE, 1, Algo_Self_Winograd_Data_File);
       // Self_Winograd_Convolution_CUDA("input/128x128_RGB.jpeg", "output_datasets/self_wino/128x128_Self_Winograd.jpeg", kernel_template, KERNEL_SIZE, 1, Algo_Self_Winograd_Data_File);
    }

    // Self GEMM
    printf("Self GEMM impl:\n");
    for (int i = 0; i < 2; i++) {
     // Self_Gemm_Convolution("input/1024x1024_RGB.jpeg", "output/1024x1024_Self_GEMM.jpeg", kernel_template, KERNEL_SIZE, 1, Self_Gemm_Data_File);
     // Self_Gemm_Convolution("input/128x128_RGB.jpeg", "output/128x128_Self_GEMM.jpeg", kernel_template, KERNEL_SIZE, 1, Self_Gemm_Data_File);
    //Self_Gemm_Convolution("input/512x512.jpg", "output/512x512_Self_GEMM.png", new_h_kernel, KERNEL_SIZE, 1, Self_Gemm_Data_File);
    }


    size_t currentFreeBytes, currentTotalBytes;
    cudaMemGetInfo(&currentFreeBytes, &currentTotalBytes);
    size_t usedBytesAfter = currentTotalBytes - currentFreeBytes;

    std::cout << " Total Memory (MB): " << (currentTotalBytes / 1024.0 / 1024.0) << std::endl;
    std::cout << " Current Free Memory (MB): " << (currentFreeBytes / 1024.0 / 1024.0) << std::endl;
    std::cout << " Used Memory (MB): " << (usedBytesAfter / 1024.0 / 1024.0) << std::endl;


    // const char* folderPath = "input_datasets/cifar_10/"; // Thay đường dẫn đến thư mục
    // const char* folderPath = "input_datasets/tampere_17/";
    // const char* folderPath = "input_datasets/MLRS_Net/";
     const char* folderPath = "input_datasets/mnist/";
    struct dirent* entry;
    DIR* dp = opendir(folderPath);

    if (dp == NULL) {
        perror("opendir");
        return 1;
    }
    while ((entry = readdir(dp))) {
        // Bỏ qua các mục "." và ".."
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }

        // Kiểm tra xem mục có phải là tệp không
        if (entry->d_type == DT_REG) {
            // cifar_10
            // char out_path[1024] = "output_datasets/cifar_10/direct/";
            // char out_path[1024] = "output_datasets/cifar_10/self_gemm/";
            // char out_path[1024] = "output_datasets/cifar_10/gemm/";
            // char out_path[1024] = "output_datasets/cifar_10/self_winograd/";
            // char out_path[1024] = "output_datasets/cifar_10/winograd/";
            // char in_path[1024] = "input_datasets/cifar_10/";

            // tampere_17
            // char out_path[1024] = "output_datasets/tampere_17/direct/";
            // char out_path[1024] = "output_datasets/tampere_17/self_gemm/";
            // char out_path[1024] = "output_datasets/tampere_17/gemm/";
            // char out_path[1024] = "output_datasets/tampere_17/self_winograd/";
            // char out_path[1024] = "output_datasets/tampere_17/winograd/";
            // char in_path[1024] = "input_datasets/tampere_17/";

            // MLRS_Net
            // char out_path[1024] = "output_datasets/MLRS_Net/direct/";
            // char out_path[1024] = "output_datasets/MLRS_Net/self_gemm/";
            // char out_path[1024] = "output_datasets/MLRS_Net/gemm/";
            // char out_path[1024] = "output_datasets/MLRS_Net/self_winograd/";
            // char out_path[1024] = "output_datasets/MLRS_Net/winograd/";
            // char in_path[1024] = "input_datasets/MLRS_Net/";

            // mnist
            // char out_path[1024] = "output_datasets/mnist/direct/";
            // char out_path[1024] = "output_datasets/mnist/self_gemm/";
            // char out_path[1024] = "output_datasets/mnist/gemm/";
            // char out_path[1024] = "output_datasets/mnist/self_winograd/";
             char out_path[1024] = "output_datasets/mnist/winograd/";
             char in_path[1024] = "input_datasets/mnist/";
                        
            strcat(out_path, entry->d_name);
            strcat(in_path, entry->d_name);
            // Self_Winograd_Convolution_CUDA(in_path, out_path, kernel_template, KERNEL_SIZE, 1, Self_Winograd_Data_File);
            // Self_Gemm_Convolution(in_path, out_path, kernel_template, KERNEL_SIZE, 1, Self_Gemm_Data_File);
            // CudnnRuntimeAlgoGemn(in_path, out_path, kernel_template, Algo_Gemm_Data_File);
            // CudnnRuntimeAlgoWinograd(in_path, out_path, kernel_template, Algo_Winograd_Data_File);
            // Self_Direct_Convolution_CUDA(in_path, out_path, kernel_template, KERNEL_SIZE, 1, Algo_Direct_Data_File);
        }
    }

    closedir(dp);

    fclose(Algo_Direct_Data_File);
    fclose(Algo_Gemm_Data_File);
    fclose(Algo_Winograd_Data_File);
    fclose(Self_Winograd_Data_File);
    fclose(Self_Gemm_Data_File);
}
