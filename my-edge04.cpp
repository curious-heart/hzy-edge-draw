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

int main(int argc, char** argv)
{
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " image_file" << endl;
        return -1;
    }

    string img_fn = argv[1];

	int bdr_width = 20;      // 默认值
    int slice_width = 20;   // 默认值
    double bdr_diff_r = 0.7; // 默认值
    string name_apx = "-my_edge04";

    if (argc >= 3) bdr_width   = std::atoi(argv[2]);
    if (argc >= 4) slice_width = std::atoi(argv[3]);
    if (argc >= 5) bdr_diff_r  = std::atof(argv[4]);
    if (argc >= 6) name_apx = argv[5];

    Mat img = imread(img_fn, IMREAD_UNCHANGED);
    if (img.empty()) {
        cout << "Cannot read image: " << img_fn << endl;
        return -1;
    }

    Mat boundary = detect_boundaries_block_4dir(img, bdr_width, 
												slice_width, bdr_diff_r);

	string dst_fn = make_new_filename(img_fn, name_apx);

    Mat edges;
    convertScaleAbs(boundary, edges); // 可视化
    imwrite(dst_fn, edges);

    cout << "Saved result: " << dst_fn << endl;

    return 0;
}
