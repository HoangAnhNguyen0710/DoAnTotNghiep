#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <cudnn.h>
#include <string.h>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include "ImageInOut.h"
#include "CudaCustomFunc.h"

float* CudnnRuntimeAlgoWinograd(char* imgName, char* outputImg, float kernel_template[][KERNEL_SIZE], FILE* outputFile) {
    cv::Mat image;
    if (TOTAL_CHANNELS > 3) {
        image = load_multi_channels_image_from_multi_images(imgName, ".jpg", TOTAL_CHANNELS);
    }
    if (TOTAL_CHANNELS == 3) {
        image = load_image(imgName, TOTAL_CHANNELS);
    }
    if (TOTAL_CHANNELS == 1) {
        image = load_image_grayscale(imgName);
    }
    const int kernel_size = KERNEL_SIZE;
    const int batch_size = 1;

    
    size_t beforeFreeBytes, beforeTotalBytes;
    cudaMemGetInfo(&beforeFreeBytes, &beforeTotalBytes);

    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    cudnnTensorDescriptor_t input_descriptor;
    cudnnCreateTensorDescriptor(&input_descriptor);
    cudnnSetTensor4dDescriptor(input_descriptor,
        CUDNN_TENSOR_NHWC,
        CUDNN_DATA_FLOAT,
        batch_size,
        image.channels(),
        image.rows / batch_size,  // We divide the rows by batch size to get the original image height
        image.cols);

    cudnnFilterDescriptor_t kernel_descriptor;
    cudnnCreateFilterDescriptor(&kernel_descriptor);
    cudnnSetFilter4dDescriptor(kernel_descriptor,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        image.channels(),
        image.channels(),
        kernel_size,
        kernel_size);

    cudnnConvolutionDescriptor_t convolution_descriptor;
    cudnnCreateConvolutionDescriptor(&convolution_descriptor);
    cudnnSetConvolution2dDescriptor(convolution_descriptor,
        0, 0, 1, 1, 1, 1,
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT);

    int output_batch_size, channels, output_height, output_width;
    cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
        input_descriptor,
        kernel_descriptor,
        &output_batch_size,
        &channels,
        &output_height,
        &output_width);

   // std::cerr << "Input Image: " << image.cols << " x " << image.rows << " x " << image.channels() << " x " << batch_size
    //    << std::endl;

    std::cerr << "Output Image: " << output_height << " x " << output_width << " x " <<  image.channels() << " x " << batch_size
        << std::endl;
    
    cudnnTensorDescriptor_t output_descriptor;
    cudnnCreateTensorDescriptor(&output_descriptor);
    cudnnSetTensor4dDescriptor(output_descriptor,
        CUDNN_TENSOR_NHWC,
        CUDNN_DATA_FLOAT,
        output_batch_size,
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
    void* d_workspace{ nullptr };
    cudaMalloc((void**)&d_workspace, workspace_bytes);

    long image_bytes = output_batch_size * channels * output_height * output_width * sizeof(float);
    long input_img_bytes = batch_size * (image.rows / batch_size) * image.cols * channels * sizeof(float);
    
    float* d_input{ nullptr };
    cudaMalloc((void**)&d_input, input_img_bytes);
    float* pixelData = image.ptr<float>(0);
    cudaMemcpy(d_input, pixelData, input_img_bytes, cudaMemcpyHostToDevice);
    
    float* d_output{ nullptr };
    cudaMalloc((void**)&d_output, image_bytes);
    cudaMemset(d_output, 0, image_bytes);
    
    float h_kernel[TOTAL_CHANNELS][TOTAL_CHANNELS][kernel_size][kernel_size];
    for (int kernel = 0; kernel < channels; kernel++) {
        for (int ch = 0; ch < channels; ch++) {
            for (int row = 0; row < kernel_size; row++) {
                for (int column = 0; column < kernel_size; column++) {
                    if (kernel == ch)
                        h_kernel[kernel][ch][row][column] = kernel_template[row][column];
                    else
                        h_kernel[kernel][ch][row][column] = 0.0f;
                }
            }
        }
    }

    float* d_kernel{ nullptr };
    cudaMalloc(&d_kernel, sizeof(h_kernel));
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

    size_t afterFreeBytes, afterTotalBytes;
    cudaMemGetInfo(&afterFreeBytes, &afterTotalBytes);
    size_t usedBytes = beforeFreeBytes - afterFreeBytes;

    std::cout << " Free Memory (MB): " << (afterFreeBytes / 1024.0 / 1024.0) << std::endl;
    std::cout << " Used Memory (MB): " << (usedBytes / 1024.0 / 1024.0) << std::endl;

    cudaMemcpy(h_output, d_output, image_bytes, cudaMemcpyDeviceToHost);
    float cudnnMillisec = 0;
    cudaEventElapsedTime(&cudnnMillisec, start, stop);
    printf("CUDNN run duration : %f ms\n", cudnnMillisec);

    // Save data to file
    fprintf(outputFile, "%f\n", cudnnMillisec);

    // Save image
    if (TOTAL_CHANNELS > 3) {
        save_multi_channels_image_to_multi_image(outputImg, h_output, output_height, output_width, channels);
    }
    else {
        save_image(outputImg, h_output, output_height, output_width, TOTAL_CHANNELS);
    }
    // Destroy cudnn resources
    // delete[] h_output;
    cudaFree(d_kernel);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_workspace);

    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);

    cudnnDestroy(cudnn);
    return h_output;
    
}
