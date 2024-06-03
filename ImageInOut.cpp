#include "ImageInOut.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

cv::Mat load_image(const char* image_path) {
    cv::Mat image = cv::imread(image_path, CV_LOAD_IMAGE);
    image.convertTo(image, CV_32FC3);
    cv::normalize(image, image, 0.0, 1.0, cv::NORM_MINMAX);
    return image;
}

void save_image(const char* output_filename,
    float* buffer,
    int height,
    int width) {
    cv::Mat output_image(height, width, CV_32FC3, buffer); // Sử dụng CV_32FC3 cho ảnh 3 kênh
    //for (int i = 100; i < 103; i++) {
    //  for (int j = 100; j < 103; j++)
    //  printf("i%d j %d -> %f\t", i, j, buffer[i * 256 + j]);
    //}
    // Make negative values zero.
    cv::threshold(output_image, output_image, /*threshold=*/0, /*maxval=*/0, cv::THRESH_TOZERO);
    cv::normalize(output_image, output_image, 0.0, 255.0, cv::NORM_MINMAX);
    output_image.convertTo(output_image, CV_8UC3); // Chuyển đổi sang kiểu dữ liệu CV_32FC3
    cv::imwrite(output_filename, output_image);
}