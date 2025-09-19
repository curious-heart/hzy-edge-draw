#include "opencv2/opencv.hpp"
#include <iostream>
#include <string>
#include <algorithm>
#include <filesystem>

using namespace cv;
using namespace std;

std::string make_new_filename(const std::string& img_fn, const std::string& ap_str)
{
    // 找到最后一个路径分隔符（兼容 Windows 和 Linux）
    size_t slash_pos = img_fn.find_last_of("/\\");
    std::string dir = (slash_pos == std::string::npos) ? "" : img_fn.substr(0, slash_pos + 1);
    std::string name_ext = (slash_pos == std::string::npos) ? img_fn : img_fn.substr(slash_pos + 1);

    // 找到最后一个点，分离扩展名
    size_t dot_pos = name_ext.find_last_of('.');
    std::string base, ext;
    if (dot_pos == std::string::npos) {
        base = name_ext;      // 没有扩展名
        ext = "";
    } else {
        base = name_ext.substr(0, dot_pos);
        ext = name_ext.substr(dot_pos);  // 包含"."
    }

    return dir + base + ap_str + ext;
}

Mat detect_boundaries_block_4dir(const Mat& img, int bdr_width = 5, int slice_width = 4, 
								double bdr_diff_r = 20.0f)
{
    Mat gray;
    if (img.channels() > 1) {
        cvtColor(img, gray, COLOR_BGR2GRAY);
    } else {
        gray = img;
    }

    int h = gray.rows;
    int w = gray.cols;
    Mat img_f;
    gray.convertTo(img_f, CV_32F);

    Mat out = Mat::zeros(h, w, CV_32F);

    // ----------------- 左→右 -----------------
    for (int y0 = 0; y0 < h; y0 += slice_width) {
        int y1 = min(y0 + slice_width, h);
        bool row_set = false;
        for (int x0 = 0; x0 < w; x0 += bdr_width) {
            if (row_set) break;
            int x1 = min(x0 + bdr_width, w);

            Mat center_block = img_f(Range(y0, y1), Range(x0, x1));
            double center_mean = mean(center_block)[0];

            int xl0 = max(0, x0 - bdr_width);
            int xl1 = x0;
            double left_mean = 0;
            if (xl1 > xl0)
            {
                left_mean = mean(img_f(Range(y0, y1), Range(xl0, xl1)))[0];
            }

            //if (center_mean - left_mean > bdr_diff_r)
            if((xl1 > xl0) && (center_mean > 0) 
                && ((left_mean / center_mean) < bdr_diff_r))
            {
                out(Range(y0, y1), Range(x0, x1)) = center_mean;
                out(Range(y0, y1), Range(0, x0)) = 0;
                out(Range(y0, y1), Range(x1, w)) = 0;
                row_set = true;
            }
        }
    }

    // ----------------- 右→左 -----------------
    for (int y0 = 0; y0 < h; y0 += slice_width) {
        int y1 = min(y0 + slice_width, h);
        bool row_set = false;
        for (int x0 = w - bdr_width; x0 >= 0; x0 -= bdr_width) {
            if (row_set) break;
            int x1 = min(x0 + bdr_width, w);

            Mat center_block = img_f(Range(y0, y1), Range(x0, x1));
            double center_mean = mean(center_block)[0];

            int xr0 = x1;
            int xr1 = min(w, x1 + bdr_width);
            double right_mean = 0;
            if(xr1 > xr0)
            {
                right_mean = mean(img_f(Range(y0, y1), Range(xr0, xr1)))[0];
            }

            //if (center_mean - right_mean > bdr_diff_r)
            if ((xr1 > xr0) && (center_mean > 0)
					&& ((right_mean / center_mean) < bdr_diff_r))
            {
                out(Range(y0, y1), Range(x0, x1)) = center_mean;
                out(Range(y0, y1), Range(0, x0)) = 0;
                out(Range(y0, y1), Range(x1, w)) = 0;
                row_set = true;
            }
        }
    }

    // ----------------- 上→下 -----------------
    for (int x0 = 0; x0 < w; x0 += bdr_width) {
        int x1 = min(x0 + bdr_width, w);
        bool col_set = false;
        for (int y0 = 0; y0 < h; y0 += slice_width) {
            if (col_set) break;
            int y1 = min(y0 + slice_width, h);

            Mat center_block = img_f(Range(y0, y1), Range(x0, x1));
            double center_mean = mean(center_block)[0];

            int yb0 = min(h, y1);
            int yb1 = min(h, y1 + slice_width);
            double bottom_mean = 0;
            if(yb1 > yb0) bottom_mean = mean(img_f(Range(yb0, yb1), Range(x0, x1)))[0];

            //if (center_mean - bottom_mean > bdr_diff_r)
            if((yb1 > yb0) && (center_mean > 0)
                && ((bottom_mean / center_mean) < bdr_diff_r))
            {
                out(Range(y0, y1), Range(x0, x1)) = center_mean;
                out(Range(0, y0), Range(x0, x1)) = 0;
                out(Range(y1, h), Range(x0, x1)) = 0;
                col_set = true;
            }
        }
    }

    // ----------------- 下→上 -----------------
    for (int x0 = 0; x0 < w; x0 += bdr_width) {
        int x1 = min(x0 + bdr_width, w);
        bool col_set = false;
        for (int y0 = h - slice_width; y0 >= 0; y0 -= slice_width) {
            if (col_set) break;
            int y1 = min(y0 + slice_width, h);

            Mat center_block = img_f(Range(y0, y1), Range(x0, x1));
            double center_mean = mean(center_block)[0];

            int yt0 = max(0, y0 - slice_width);
            int yt1 = y0;
            double top_mean = 0;
            if(yt1 > yt0) top_mean = mean(img_f(Range(yt0, yt1), Range(x0, x1)))[0];

            if((yt1 > yt0) && (center_mean > 0)
                && ((top_mean / center_mean) < bdr_diff_r))
            {
                out(Range(y0, y1), Range(x0, x1)) = center_mean;
                out(Range(0, y0), Range(x0, x1)) = 0;
                out(Range(y1, h), Range(x0, x1)) = 0;
                col_set = true;
            }
        }
    }

    // 转回原位深
    Mat out_final;
    if (img.depth() == CV_8U) {
        out.convertTo(out_final, CV_8U);
    } else {
        out.convertTo(out_final, CV_16U);
    }

    return out_final;
}

/**
 * @brief 对输入图像执行：阈值处理、二值化、形态学闭运算、边界提取
 * @param img 输入图像（8位或16位灰度/彩色）
 * @param binary 输出二值图像 (8位)
 * @param morph 输出形态学闭运算结果 (8位)
 * @param edges 输出边界图像 (8位)
 */
void draw_wave_front_8bit(const Mat& img, Mat& binary, Mat& morph, Mat& edges,
							int erode_strength = 0)
{
    // 转灰度
    Mat gray;
    if (img.channels() == 3 || img.channels() == 4)
        cvtColor(img, gray, COLOR_BGR2GRAY);
    else
        gray = img;

    // 如果是16位图像，转换为8位
    Mat gray8;
    if (gray.depth() == CV_16U)
    {
        double minVal, maxVal;
        minMaxLoc(gray, &minVal, &maxVal);
        double scale = 255.0 / (maxVal > 0 ? maxVal : 1.0);
        gray.convertTo(gray8, CV_8U, scale);
    }
    else
    {
        gray8 = gray;
    }

    // Otsu 二值化
    threshold(gray8, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);

    // 形态学闭运算（膨胀后腐蚀）
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(binary, morph, MORPH_CLOSE, kernel, Point(-1, -1), 1);

    if (erode_strength > 0)
    {
        // 根据腐蚀强度放大 kernel 大小、增加迭代次数
        int ksize = 3 + 2 * std::min(erode_strength, 3);  // 3→5→7→9
        Mat ekernel = getStructuringElement(MORPH_RECT, Size(ksize, ksize));
        int iters = std::min(erode_strength, 3);
        erode(morph, morph, ekernel, Point(-1, -1), iters);
    }
    // 提取边界：膨胀 - 原图
    Mat dilated;
    dilate(morph, dilated, kernel, Point(-1, -1), 1);
    subtract(dilated, morph, edges);
}

/**
 * @brief 对输入图像执行：阈值处理(保留16位)、二值化、形态学闭运算、边界提取
 * @param img 输入图像（8位或16位灰度/彩色）
 * @param binary 输出二值图像 (如果输入是16位 -> CV_16U 0/65535；否则 CV_8U 0/255)
 * @param morph 输出形态学闭运算结果 (CV_8U)
 * @param edges 输出边界图像 (CV_8U)
 */
void draw_wave_front(const Mat& img, Mat& binary, Mat& morph, Mat& edges,
    int erode_strength = 0)
{
    // 转灰度
    Mat gray;
    if (img.channels() == 3 || img.channels() == 4)
        cvtColor(img, gray, COLOR_BGR2GRAY);
    else
        gray = img;

    // -------------------------
    // 二值化（支持16位）
    // -------------------------
    if (gray.depth() == CV_16U)
    {
        // 在缩放后的8位图上用Otsu找阈值
        double minv, maxv;
        minMaxLoc(gray, &minv, &maxv);
        Mat gray8;
        gray.convertTo(gray8, CV_8U, 255.0 / (maxv > 0 ? maxv : 1.0));

        Mat dummy;
        double otsu_thresh = threshold(gray8, dummy, 0, 255, THRESH_BINARY | THRESH_OTSU);

        // 映射回16位阈值
        double scaled_thresh = otsu_thresh * (maxv / 255.0);

        // 生成16位二值图（0 / 65535）
        Mat mask = (gray > scaled_thresh);
        mask.convertTo(binary, CV_16U, 65535);
    }
    else
    {
        // 8位图直接用Otsu
        threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
    }

    // -------------------------
    // 形态学闭运算
    // -------------------------
    Mat binary8;
    if (binary.depth() == CV_16U)
        binary.convertTo(binary8, CV_8U, 255.0 / 65535.0);
    else
        binary8 = binary;

    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(binary8, morph, MORPH_CLOSE, kernel, Point(-1, -1), 1);

    if (erode_strength > 0)
    {
        // 根据腐蚀强度放大 kernel 大小、增加迭代次数
        int ksize = 3 + 2 * std::min(erode_strength, 3);  // 3→5→7→9
        Mat ekernel = getStructuringElement(MORPH_RECT, Size(ksize, ksize));
        int iters = std::min(erode_strength, 3);
        erode(morph, morph, ekernel, Point(-1, -1), iters);
    }
    // -------------------------
    // 提取边界：膨胀 - 原图
    // -------------------------
    Mat dilated;
    dilate(morph, dilated, kernel, Point(-1, -1), 1);
    subtract(dilated, morph, edges);
}

#define FOUR_DIR '1'
#define DRAW_WAVE_OTSU_8BIT '2'
#define DRAW_WAVE_OTSUL '3'

int main(int argc, char** argv)
{
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " image_file" << endl;
        return -1;
    }

    string img_fn = argv[1];

    Mat img = imread(img_fn, IMREAD_UNCHANGED);
    if (img.empty()) {
        cout << "Cannot read image: " << img_fn << endl;
        return -1;
    }

    if (2 == argc || FOUR_DIR == argv[2][0])
    {
        int bdr_width = 20;      // 默认值
        int slice_width = 20;   // 默认值
        double bdr_diff_r = 0.7; // 默认值
        string name_apx = "-my_edge04";

        if (argc >= 4) bdr_width = std::atoi(argv[3]);
        if (argc >= 5) slice_width = std::atoi(argv[4]);
        if (argc >= 6) bdr_diff_r = std::atof(argv[5]);
        if (argc >= 7) name_apx += argv[6];

        std::ostringstream oss;
        oss << bdr_diff_r;
        name_apx += std::to_string(bdr_width) + "x"
            + std::to_string(slice_width) + std::string("x") + oss.str();


        Mat boundary = detect_boundaries_block_4dir(img, bdr_width,
													slice_width, bdr_diff_r);
		string dst_fn = make_new_filename(img_fn, name_apx);
		Mat edges;
		convertScaleAbs(boundary, edges); // 可视化
		imwrite(dst_fn, edges);

		cout << "Saved result: " << dst_fn << endl;
    }
    else
    {
        string name_apx;
        int erode_strength = 0;
        if (argc > 3)
        {
            erode_strength = std::atoi(argv[3]);
        }

		Mat binary, morph, edges;
        if (DRAW_WAVE_OTSU_8BIT == argv[2][0])
        {
            draw_wave_front_8bit(img, binary, morph, edges, erode_strength);
            name_apx = "-draw_wave-otsu8bit";
        }
        else
        {
            draw_wave_front(img, binary, morph, edges, erode_strength);
            name_apx = "-draw_wave-otsu";
        }

        if (argc > 3)
        {
            name_apx += "-er_" + std::string(argv[3]);
        }

		string dst_bin_fn = make_new_filename(img_fn, name_apx + "-bin");
		string dst_mor_fn = make_new_filename(img_fn, name_apx + "-mor");
		string dst_edg_fn = make_new_filename(img_fn, name_apx + "-edg");

		imwrite(dst_bin_fn, binary);
		imwrite(dst_mor_fn, morph);
		imwrite(dst_edg_fn, edges);
    }

    return 0;
}
