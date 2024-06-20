#include "ImageInOut.h"
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>
#include <string.h>

/*
cv::Mat load_image(const char* image_path) {
    cv::Mat image = cv::imread(image_path, CV_LOAD_IMAGE);
    image.convertTo(image, CV_32FC3);
    cv::normalize(image, image, 0.0, 1.0, cv::NORM_MINMAX);
    return image;
}
*/

cv::Mat load_image_grayscale(const char* image_path) {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    image.convertTo(image, CV_32FC1);
    cv::normalize(image, image, 0.0, 1.0, cv::NORM_MINMAX);
    return image;
}

//update 18/0/2024
cv::Mat load_multi_channels_bmp_image_from_multi_images(const char* directory, const char* type, int n) {
    std::vector<cv::Mat> images;
    images.clear(); // Xóa vector ảnh nếu đã có sẵn

    for (int i = 1; i <= n; ++i) {
        // Tạo đường dẫn đầy đủ cho file ảnh bằng cách sử dụng std::string
        std::string filename = std::string(directory) + "_" + std::to_string(i) + std::string(type);
        cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE); // Đọc ảnh dưới dạng grayscale
        if (!img.empty()) {
            cv::Mat img32f;
            img.convertTo(img32f, CV_32F); // Chuyển đổi ảnh sang kiểu CV_32F
            images.push_back(img32f);
        }
        else {
            std::cerr << "Failed to read image: " << filename << std::endl;
        }
    }

    cv::Mat mergedImage;
    if (!images.empty()) {
        cv::merge(images, mergedImage); // Ghép các kênh lại với nhau
    }
    return mergedImage;
}

//update 18/0/2024
bool save_multi_channels_image_to_multi_image(const char* output_filename, float* buffer, int height, int width, int channels) {
    if (!buffer) {
        std::cerr << "Input buffer is null." << std::endl;
        return false;
    }

    try {
        // Tạo một ảnh từ buffer với dữ liệu kiểu float
        cv::Mat input_image(height, width, CV_32FC(channels), buffer);

        // Bỏ qua giá trị âm
        cv::threshold(input_image, input_image, 0, 0, cv::THRESH_TOZERO);

        // Tách ảnh thành các kênh riêng
        std::vector<cv::Mat> split_channels;
        cv::split(input_image, split_channels);

        // Lưu mỗi kênh riêng lẻ
        for (int i = 0; i < split_channels.size(); i++) {
            // Chuyển đổi mỗi kênh sang kiểu dữ liệu CV_8U để lưu
            cv::Mat channel_image;
            split_channels[i].convertTo(channel_image, CV_8UC1, 255.0 / (255.0 - 0.0), 0);

            // Tạo tên file cho mỗi kênh
            std::string filename = std::string(output_filename) + "_channel_" + std::to_string(i + 1) + ".bmp";

            // Ghi ảnh ra file
            if (!cv::imwrite(filename, channel_image)) {
                std::cerr << "Failed to write image to file: " << filename << std::endl;
                return false;
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Exception occurred: " << e.what() << std::endl;
        return false;
    }

    return true;
}

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

void save_image(const char* output_filename,
    float* buffer,
    int height,
    int width, int channels) {
    cv::Mat output_image(height, width, CV_32FC(channels), buffer); // Sử dụng CV_32FC(N) cho ảnh N kênh (N = {1, 3, 4})

    // Make negative values zero.
    cv::threshold(output_image, output_image, 0, 0, cv::THRESH_TOZERO);
    cv::normalize(output_image, output_image, 0.0, 255.0, cv::NORM_MINMAX);
    output_image.convertTo(output_image, CV_8UC(channels)); // Chuyển đổi sang kiểu dữ liệu CV_8UC(N)
    cv::imwrite(output_filename, output_image);
}

void save_image_from_n_channels_to_3_channels(const char* output_filename, float* buffer, int height, int width, int input_channels) {
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
    reduced_image.convertTo(reduced_image, CV_8UC3);

    // Save the image
    if (!cv::imwrite(output_filename, reduced_image)) {
        throw std::runtime_error("Failed to write the image");
    }
}
