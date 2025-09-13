#include "opencv2/opencv.hpp"
#include <iostream>
#include <string>
#include <algorithm>

using namespace cv;
using namespace std;

Mat detect_boundaries_block_4dir(const Mat& img, int bdr_width = 5, int slice_width = 4, float bdr_diff = 20.0f)
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
            float center_mean = mean(center_block)[0];

            int xl0 = max(0, x0 - bdr_width);
            int xl1 = x0;
            float left_mean = 0;
            if (xl1 > xl0) left_mean = mean(img_f(Range(y0, y1), Range(xl0, xl1)))[0];

            if (center_mean - left_mean > bdr_diff) {
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
            float center_mean = mean(center_block)[0];

            int xr0 = x1;
            int xr1 = min(w, x1 + bdr_width);
            float right_mean = 0;
            if (xr1 > xr0) right_mean = mean(img_f(Range(y0, y1), Range(xr0, xr1)))[0];

            if (center_mean - right_mean > bdr_diff) {
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
            float center_mean = mean(center_block)[0];

            int yb0 = min(h, y1);
            int yb1 = min(h, y1 + slice_width);
            float bottom_mean = 0;
            if (yb1 > yb0) bottom_mean = mean(img_f(Range(yb0, yb1), Range(x0, x1)))[0];

            if (center_mean - bottom_mean > bdr_diff) {
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
            float center_mean = mean(center_block)[0];

            int yt0 = max(0, y0 - slice_width);
            int yt1 = y0;
            float top_mean = 0;
            if (yt1 > yt0) top_mean = mean(img_f(Range(yt0, yt1), Range(x0, x1)))[0];

            if (center_mean - top_mean > bdr_diff) {
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
    Mat img = imread(img_fn, IMREAD_UNCHANGED);
    if (img.empty()) {
        cout << "Cannot read image: " << img_fn << endl;
        return -1;
    }

    Mat boundary = detect_boundaries_block_4dir(img, 5, 10, 20.0f);

    string dst_fn = img_fn.substr(0, img_fn.find_last_of(".")) + "_my_edge04.png";

    Mat edges;
    convertScaleAbs(boundary, edges); // 可视化
    imwrite(dst_fn, edges);

    cout << "Saved result: " << dst_fn << endl;

    return 0;
}
