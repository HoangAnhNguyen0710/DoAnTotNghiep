#pragma once
#include <opencv2/opencv.hpp>


#define CV_LOAD_IMAGE 1
cv::Mat load_image(const char* image_path, int channels);
cv::Mat load_image_grayscale(const char* image_path);
cv::Mat load_multi_channels_bmp_image_from_multi_images(const char* directory, const char* type, int n);
void save_image(const char* output_filename, float* buffer, int height, int width, int input_channels);
bool save_multi_channels_image_to_multi_image(const char* output_filename, float* buffer, int height, int width, int channels);
void save_image_from_n_channels_to_3_channels(const char* output_filename, float* buffer, int height, int width, int input_channels);