#include "ImageInOut.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

cv::Mat load_image(const char* image_path) {
    cv::Mat image = cv::imread(image_path, CV_LOAD_IMAGE);
    image.convertTo(image, CV_32FC1);
    cv::normalize(image, image, 0.0, 1.0, cv::NORM_MINMAX);
    //std::cerr << "Input Image: " << image.rows << " x " << image.cols << " x "
    //    << image.channels() << std::endl;
    return image;
}

void save_image(const char* output_filename,
    float* buffer,
    int height,
    int width) {
    cv::Mat output_image(height, width, CV_32FC1, buffer); // Sử dụng CV_32FC1 cho ảnh xám 1 kênh

    // Make negative values zero.
    cv::threshold(output_image, output_image, /*threshold=*/0, /*maxval=*/0, cv::THRESH_TOZERO);
    cv::normalize(output_image, output_image, 0.0, 255.0, cv::NORM_MINMAX);
    output_image.convertTo(output_image, CV_8UC1); // Chuyển đổi sang kiểu dữ liệu 8-bit unsigned integer
    cv::imwrite(output_filename, output_image);
    //std::cerr << "Wrote output to " << output_filename << std::endl;
}