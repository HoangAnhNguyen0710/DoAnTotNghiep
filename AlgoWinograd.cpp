
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <cudnn.h>
#include<string.h>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include "ImageInOut.h"
#include "CudaCustomFunc.h"

void CudnnRuntimeAlgoWinograd(char* imgName, char* outputImg, float kernel_template[][KERNEL_SIZE], FILE* outputFile) {
 cv::Mat image = load_image(imgName);
    const int kernel_size = KERNEL_SIZE;

    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    cudnnTensorDescriptor_t input_descriptor;
    cudnnCreateTensorDescriptor(&input_descriptor);
    cudnnSetTensor4dDescriptor(input_descriptor,
        CUDNN_TENSOR_NHWC,
        CUDNN_DATA_FLOAT,
        1,
        1,
        image.rows,
        image.cols);

    cudnnFilterDescriptor_t kernel_descriptor;
    cudnnCreateFilterDescriptor(&kernel_descriptor);
    cudnnSetFilter4dDescriptor(kernel_descriptor,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        1,
        1,
        kernel_size,
        kernel_size);

    cudnnConvolutionDescriptor_t convolution_descriptor;
    cudnnCreateConvolutionDescriptor(&convolution_descriptor);
    cudnnSetConvolution2dDescriptor(convolution_descriptor,
        1, 1, 1, 1, 1, 1,
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT);

    int batch_size, channels, output_height, output_width;
    cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
        input_descriptor,
        kernel_descriptor,
        &batch_size,
        &channels,
        &output_height,
        &output_width);
     std::cerr << "Output Image: " << output_height << " x " << output_width << " x " << channels
     << std::endl;
    cudnnTensorDescriptor_t output_descriptor;
    cudnnCreateTensorDescriptor(&output_descriptor);
    cudnnSetTensor4dDescriptor(output_descriptor,
        CUDNN_TENSOR_NHWC,
        CUDNN_DATA_FLOAT,
        batch_size,
        channels,
        output_height,
        output_width);

    size_t workspace_bytes;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn,
        input_descriptor,
        kernel_descriptor,
        convolution_descriptor,
        output_descriptor,
        CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
        &workspace_bytes);

    std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB"
        << std::endl;
    assert(workspace_bytes > 0);
  //  printf("6\n");
    void* d_workspace{ nullptr };
    float* d_input{ nullptr };
    float* pixelData = image.ptr<float>(0);
    float* d_output{ nullptr };

    long image_bytes = batch_size * channels * output_height * output_width * sizeof(float);
    long input_image_bytes = batch_size * channels * image.rows * image.cols * sizeof(float);
    cudaMalloc((void**)&d_workspace, workspace_bytes);
    cudaMalloc((void**)&d_input, input_image_bytes);
    cudaMemcpy(d_input, pixelData, input_image_bytes, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_output, image_bytes);
    cudaMemset(d_output, 0, image_bytes);

    float h_kernel[kernel_size][kernel_size];
    for (int row = 0; row < kernel_size; ++row) {
        for (int column = 0; column < kernel_size; ++column) {
            h_kernel[row][column] = kernel_template[row][column];
        }
    }
    float* d_kernel{ nullptr };
    cudaMalloc(&d_kernel, sizeof(h_kernel));
    // printf("%d %d\n", kernel_size * kernel_size * channels * sizeof(float), sizeof(h_kernel));
    cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);
    float* h_output = new float[image_bytes] {0};

    const float alpha = 1.0f, beta = 1.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    checkCUDNN(cudnnConvolutionForward(cudnn,
        &alpha,
        input_descriptor,
        d_input,
        kernel_descriptor,
        d_kernel,
        convolution_descriptor,
        CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
        d_workspace,
        workspace_bytes,
        &beta,
        output_descriptor,
        d_output));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaMemcpy(h_output, d_output, image_bytes, cudaMemcpyDeviceToHost);
    float cudnnMillisec = 0;
    cudaEventElapsedTime(&cudnnMillisec, start, stop);
    printf("CUDNN run duration : %f s\n", cudnnMillisec / 1000);
    //save data to file
    fprintf(outputFile, "%f\n", cudnnMillisec / 1000);

    //save image
    save_image(outputImg, h_output, output_height, output_width);

    //destroy cudnn
    delete[] h_output;
    cudaFree(d_kernel);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_workspace);

    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);

    cudnnDestroy(cudnn);
    return;
}
