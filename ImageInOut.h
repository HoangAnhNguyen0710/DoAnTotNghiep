#pragma once
#include <opencv2/opencv.hpp>


#define CV_LOAD_IMAGE 1
cv::Mat load_image(const char* image_path, int channels);
void save_image(const char* output_filename, float* buffer, int height, int width, int input_channels);