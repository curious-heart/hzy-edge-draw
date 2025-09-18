#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_filename>" << std::endl;
        return 1;
    }

    std::string img_fn = argv[1];

    // 分离文件名与扩展名
    size_t dot_pos = img_fn.find_last_of('.');
    std::string bn = (dot_pos == std::string::npos) ? img_fn : img_fn.substr(0, dot_pos);
    std::string en = (dot_pos == std::string::npos) ? "" : img_fn.substr(dot_pos);

    std::string dst_fn = bn + "_sober_edge-level-his" + en;

    // 读取图像（建议灰度图）
    cv::Mat img = cv::imread(img_fn, cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << img_fn << std::endl;
        return 1;
    }

    std::cout << "shape: " << img.rows << "x" << img.cols 
              << (img.channels() > 1 ? "x" + std::to_string(img.channels()) : "")
              << std::endl;
    std::cout << "dtype: " << img.type() << std::endl;

    cv::Mat gray;
    if (img.channels() > 1) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = img.clone();
    }

    cv::Mat blur1;
    cv::GaussianBlur(gray, blur1, cv::Size(35, 35), 0);

    cv::Mat sobel_x, sobel_y;
    cv::Sobel(blur1, sobel_x, CV_64F, 1, 0, 3);
    cv::Sobel(blur1, sobel_y, CV_64F, 0, 1, 3);

    cv::Mat sobel_img;
    cv::magnitude(sobel_x, sobel_y, sobel_img);

    // 低于阈值的部分清零
    for (int r = 0; r < sobel_img.rows; ++r) {
        double* row = sobel_img.ptr<double>(r);
        for (int c = 0; c < sobel_img.cols; ++c) {
            if (row[c] < 40.0) row[c] = 0.0;
        }
    }

    // 转换为 8位图像
    cv::Mat edges;
    cv::convertScaleAbs(sobel_img, edges);

    if (!cv::imwrite(dst_fn, edges)) {
        std::cerr << "Failed to write output: " << dst_fn << std::endl;
        return 1;
    }

    std::cout << "Saved result to " << dst_fn << std::endl;

    return 0;
}
