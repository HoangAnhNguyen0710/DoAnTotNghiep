#include "ImageInOut.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

/*
cv::Mat load_image(const char* image_path) {
    cv::Mat image = cv::imread(image_path, CV_LOAD_IMAGE);
    image.convertTo(image, CV_32FC3);
    cv::normalize(image, image, 0.0, 1.0, cv::NORM_MINMAX);
    return image;
}
*/

cv::Mat load_image(const char* image_path, const int channels) {
    // Load the image with OpenCV. We use IMREAD_UNCHANGED to potentially load the alpha channel if present.
    cv::Mat image = cv::imread(image_path, cv::IMREAD_UNCHANGED);
    if (image.empty()) {
        throw std::runtime_error("Could not open or find the image");
    }

    // Convert image to float type
    image.convertTo(image, CV_32F);
    cv::normalize(image, image, 0.0, 1.0, cv::NORM_MINMAX);

    int rows = image.rows;
    int cols = image.cols;
    int originalChannels = image.channels();

    // Create a new image with the target number of channels
    cv::Mat expanded_image(rows, cols, CV_MAKETYPE(CV_32F, channels));

    // Use cv::mixChannels to copy and expand channels
    std::vector<cv::Mat> split_channels;
    cv::split(image, split_channels);

    std::vector<cv::Mat> output_channels(channels);
    for (int i = 0; i < channels; ++i) {
        // Cycle through original channels to replicate them in the output image
        output_channels[i] = split_channels[i % originalChannels];
    }

    cv::merge(output_channels, expanded_image);
    int batch_size = 1;
    // Create a new image with batch size times the rows
    cv::Mat batch_image(rows * batch_size, cols, expanded_image.type());

    // Copy the expanded image into each batch
    for (int i = 0; i < batch_size; ++i) {
        expanded_image.copyTo(batch_image(cv::Rect(0, i * rows, cols, rows)));
    }

    return batch_image;
}

/*
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
    cv::threshold(output_image, output_image, 0, 0, cv::THRESH_TOZERO);
    cv::normalize(output_image, output_image, 0.0, 255.0, cv::NORM_MINMAX);
    output_image.convertTo(output_image, CV_8UC3); // Chuyển đổi sang kiểu dữ liệu CV_32FC3
    cv::imwrite(output_filename, output_image);
}*/

void save_image(const char* output_filename, float* buffer, int height, int width, int input_channels) {
    if (buffer == nullptr) {
        throw std::runtime_error("Input buffer is null");
    }

    // Create a cv::Mat from the input buffer
    cv::Mat image(height, width, CV_32FC(input_channels), buffer);

    int targetChannels = 3; // Chỉ lấy 3 channels đầu tiên để tạo ảnh RGB
    if (input_channels < targetChannels) {
        throw std::runtime_error("Not enough channels in input buffer");
    }

    // Create a new image with 3 channels
    cv::Mat reduced_image(height, width, CV_32FC(targetChannels));

    // Trích xuất 3 channels đầu tiên từ ảnh gốc có nhiều channels
    std::vector<cv::Mat> split_channels;
    cv::split(image, split_channels);

    // Lựa chọn 3 channels đầu tiên để tạo thành ảnh RGB
    std::vector<cv::Mat> selected_channels = { split_channels[0], split_channels[1], split_channels[2] };
    cv::merge(selected_channels, reduced_image); // Hợp nhất 3 channels đã chọn

    // Normalize and convert the image to 8-bit for saving
    cv::normalize(reduced_image, reduced_image, 0, 255, cv::NORM_MINMAX);
    reduced_image.convertTo(reduced_image, CV_8UC1);

    // Save the image
    if (!cv::imwrite(output_filename, reduced_image)) {
        throw std::runtime_error("Failed to write the image");
    }
}
