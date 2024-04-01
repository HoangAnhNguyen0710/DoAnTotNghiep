
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
#include "ImageInOut.h"
#include "CudaCustomFunc.h"


void CudnnRuntimeAlgoGemn(char* imgName, char* outputImg, float kernel_template[][KERNEL_SIZE]) {
    cv::Mat image = load_image(imgName);
    const int kernel_size = KERNEL_SIZE;
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
        /*format=*/CUDNN_TENSOR_NHWC,
        /*data_type=*/CUDNN_DATA_FLOAT,
        /*batch_size=*/1,
        /*channels=*/1,
        /*image_height=*/image.rows,
        /*image_width=*/image.cols));
    cudnnFilterDescriptor_t kernel_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
        /*data_type=*/CUDNN_DATA_FLOAT,
        /*format=*/CUDNN_TENSOR_NCHW,
        /*out_channels=*/1,
        /*in_channels=*/1,
        /*kernel_height=*/kernel_size,
        /*kernel_width=*/kernel_size));
    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
        /*pad_height=*/1,
        /*pad_width=*/1,
        /*vertical_stride=*/1,
        /*horizontal_stride=*/1,
        /*dilation_height=*/1,
        /*dilation_width=*/1,
        /*mode=*/CUDNN_CROSS_CORRELATION,
        /*computeType=*/CUDNN_DATA_FLOAT));
    int batch_size{ 0 }, channels{ 0 }, outputHeight{ 0 }, outputWidth{ 0 };
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
        input_descriptor,
        kernel_descriptor,
        &batch_size,
        &channels,
        &outputHeight,
        &outputWidth));

   // std::cerr << "Output Image: " << outputHeight << " x " << outputWidth << " x " << channels
   //     << std::endl;
    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
        /*format=*/CUDNN_TENSOR_NHWC,
        /*data_type=*/CUDNN_DATA_FLOAT,
        /*batch_size=*/batch_size,
        /*channels=*/channels,
        /*image_height=*/outputHeight,
        /*image_width=*/outputWidth));
    size_t workspace_bytes{ 0 };
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
        input_descriptor,
        kernel_descriptor,
        convolution_descriptor,
        output_descriptor,
        CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
        &workspace_bytes));
    std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB"
        << std::endl;
    assert(workspace_bytes > 0);
    void* d_workspace{ nullptr };
    cudaMalloc((void**)&d_workspace, workspace_bytes);
    long image_bytes = batch_size * channels * outputHeight * outputWidth * sizeof(float);
    float* d_input{ nullptr };
    cudaMalloc((void**)&d_input, image_bytes);
    float* pixelData = image.ptr<float>(0);

    cudaMemcpy(d_input, pixelData, image_bytes, cudaMemcpyHostToDevice);
    float* d_output{ nullptr };
    cudaMalloc((void**)&d_output, image_bytes);
    cudaMemset(d_output, 0, image_bytes);

    float h_kernel[kernel_size][kernel_size];
    for (int row = 0; row < kernel_size; row++) {
        for (int column = 0; column < kernel_size; column++) {
            h_kernel[row][column] = kernel_template[row][column];
        }
    }
    float* d_kernel{ nullptr };
    cudaMalloc(&d_kernel, sizeof(h_kernel));
    cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);

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
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
        d_workspace,
        workspace_bytes,
        &beta,
        output_descriptor,
        d_output));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float cudnnMillisec = 0;
    cudaEventElapsedTime(&cudnnMillisec, start, stop);
    printf("CUDNN run duration : %f s\n", cudnnMillisec / 1000);

    float* h_output = new float[image_bytes] {0};
    cudaMemcpy(h_output, d_output, image_bytes, cudaMemcpyDeviceToHost);

    save_image(outputImg, h_output, outputHeight, outputWidth);

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
}
