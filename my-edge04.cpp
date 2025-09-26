#define NOMINMAX
#include <windows.h>

#include "opencv2/opencv.hpp"
#include "opencv2/ximgproc.hpp"
#include <iostream>
#include <string>
#include <algorithm>
#include <filesystem>
#include <cmath>

#include <locale>

using namespace cv;
using namespace std;

typedef struct
{
    int w, h;
}grid_w_h_s_t;

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

int countBranchPointsOfGrid(const cv::Mat& grid) {
    CV_Assert(grid.type() == CV_8U || grid.type() == CV_8UC1);

    int grid_w = grid.cols;
    int grid_h = grid.rows;

    // 统计 branch points
    int branchPoints = 0;
    for (int gy = 1; gy < grid.rows - 1; gy++) {
        for (int gx = 1; gx < grid.cols - 1; gx++) {
            if (grid.at<uchar>(gy, gx) == 0) continue;

            int neighbors = 0;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dy == 0 && dx == 0) continue;
                    if (grid.at<uchar>(gy + dy, gx + dx) > 0) {
                        neighbors++;
                    }
                }
            }
            if (neighbors >= 3) {
                branchPoints++;
            }
        }
    }

    return branchPoints;
}

struct ImageInfo {
    double linearity;
    int branchPoints;
    double elongation;
};

// --- PCA 计算函数
ImageInfo analyzeImage(const cv::Mat& img, const cv::Mat& grid) {
    // --- 二值化（非零即前景）
    cv::Mat binary = (img > 0);
    binary.convertTo(binary, CV_8U);

    // --- 取最大连通域
    cv::Mat labels, stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8, CV_32S);
    if (nLabels <= 1) {  // 只有背景
        return {0.0, 0, 0.0};
    }

    // 找到最大连通域
    int maxArea = 0;
    int maxLabel = 1;
    for (int i = 1; i < nLabels; i++) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area > maxArea) {
            maxArea = area;
            maxLabel = i;
        }
    }
    cv::Mat mask = (labels == maxLabel);

    // --- 提取前景点坐标
    std::vector<cv::Point> points;
    cv::findNonZero(mask, points);
    if (points.size() < 10) {
        return {0.0, 0, 0.0};
    }

    // --- PCA
    cv::Mat dataPts(points.size(), 2, CV_64F);
    for (size_t i = 0; i < points.size(); ++i) {
        dataPts.at<double>(i, 0) = points[i].x;
        dataPts.at<double>(i, 1) = points[i].y;
    }
    cv::PCA pca_analysis(dataPts, cv::Mat(), cv::PCA::DATA_AS_ROW);
    cv::Mat eigenvalues = pca_analysis.eigenvalues;
    double linearity = eigenvalues.at<double>(0) / (eigenvalues.at<double>(1) + 1e-8);

    // --- 骨架化
    /* （简单版：使用腐蚀膨胀方式）
    cv::Mat skel(mask.size(), CV_8U, cv::Scalar(0));
    cv::Mat temp, eroded;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3,3));
    cv::Mat imgCopy = mask.clone();
    bool done;
    do {
        cv::erode(imgCopy, eroded, element);
        cv::dilate(eroded, temp, element);
        cv::subtract(imgCopy, temp, temp);
        cv::bitwise_or(skel, temp, skel);
        imgCopy = eroded.clone();
        done = (cv::countNonZero(imgCopy) == 0);
    } while (!done);
    */

    /*use thinning*/
    /*
	cv::Mat skel;
	cv::ximgproc::thinning(mask, skel, cv::ximgproc::THINNING_ZHANGSUEN);
    */

    // 分叉点统计
    int branchPoints = 0;
    /*
	cv::Mat skel = mask;
    for (int y = 1; y < skel.rows - 1; ++y) {
        for (int x = 1; x < skel.cols - 1; ++x) {
            if (skel.at<uchar>(y, x)) {
                int neighbors = 0;
                for (int dy=-1; dy<=1; ++dy) {
                    for (int dx=-1; dx<=1; ++dx) {
                        if (dy==0 && dx==0) continue;
                        if (skel.at<uchar>(y+dy, x+dx)) neighbors++;
                    }
                }
                if (neighbors >= 3) branchPoints++;
            }
        }
    }
    */

    branchPoints = countBranchPointsOfGrid(grid);
    // --- elongation
    double majorAxis = std::sqrt(eigenvalues.at<double>(0));
    double minorAxis = std::sqrt(eigenvalues.at<double>(1));
    double elongation = majorAxis / (minorAxis + 1e-5);

    return {linearity, branchPoints, elongation};
}

// --- 主选择函数
int selectBestImage(const std::vector<cv::Mat>& images,
					const std::vector<cv::Mat>& grids) {
    std::vector<double> scores;
    std::vector<ImageInfo> infos;

    for (size_t i = 0; i < images.size(); ++i) {
        ImageInfo info = analyzeImage(images[i], grids[i]);
        infos.push_back(info);

        double score = (info.linearity * 8.0 + info.elongation * 2)
						/ (info.branchPoints + 1);
        scores.push_back(score);

        // 打印每张图 info
        std::cout << "图像 " << i << " info: "
                  << "linearity=" << info.linearity
                  << ", branchPoints=" << info.branchPoints
                  << ", elongation=" << info.elongation
                  << ", score=" << score << std::endl;
    }

    // 找到最大 score 的索引
    auto maxIt = std::max_element(scores.begin(), scores.end());
    int bestIdx = std::distance(scores.begin(), maxIt);
    std::cout << "score 最大的图像索引: " << bestIdx << std::endl;
    return bestIdx;
}

typedef enum
{
    LEFT_TO_RIGHT = 0x1,
    RIGHT_TO_LEFT = 0x2,
    TOP_TO_BOTTOM = 0x4,
    BOTTOM_TO_TOP = 0x8,

    DIRC_FULL_BITS = 0x0F
}detect_dound_dir_e_t;
static inline const char* dir_e_to_str(detect_dound_dir_e_t dirc)
{
	static const char* gs_dirc_str[] = {
		"None",
		"L-R",
		"R-L",
		"T-B",
		"B-T"
	};
    if (dirc <= 0) return nullptr;
	int pos = static_cast<int>(log2(static_cast<unsigned int>(dirc))) + 1;
	if (pos >= sizeof(gs_dirc_str) / sizeof(gs_dirc_str[0])
        || pos < 0) return nullptr;

    return gs_dirc_str[pos];
}

Mat detect_boundaries_block_4dir(const Mat& img, int bdr_width = 20, int slice_width = 20,
								double bdr_diff_r = 0.9f, 
								detect_dound_dir_e_t dirc = DIRC_FULL_BITS,
								std::vector<cv::Mat>* all_outs = nullptr,
								std::vector<std::string>* all_outs_names = nullptr,
								int * best_idx = nullptr)
{
    Mat gray;
    if (img.channels() > 1) {
        cvtColor(img, gray, COLOR_BGR2GRAY);
    }
    else {
        gray = img;
    }

    int h = gray.rows;
    int w = gray.cols;
    Mat img_f;
    gray.convertTo(img_f, CV_32F);

	std::vector<cv::Mat> result_images;
	std::vector<cv::Mat> result_grids;

	Mat out = Mat::zeros(h, w, CV_32F);
    // ----------------- 左→右 -----------------
    if (dirc | LEFT_TO_RIGHT)
    {
		int grid_w = (w + bdr_width - 1) / bdr_width;
		int grid_h = (h + slice_width - 1) / slice_width;
		Mat grid = Mat::zeros(grid_h, grid_w, CV_8U);

		out.setTo(0);
        for (int y0 = 0; y0 < h; y0 += slice_width)
        {
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
                if ((xl1 > xl0) && (center_mean > 0)
                    && ((left_mean / center_mean) < bdr_diff_r))
                {
                    out(Range(y0, y1), Range(x0, x1)) = center_mean;
                    out(Range(y0, y1), Range(0, x0)) = 0;
                    out(Range(y0, y1), Range(x1, w)) = 0;
                    row_set = true;

					grid.at<uchar>(y0 / slice_width, x0 / bdr_width) = 1;
                }
            }
        }

		Mat out_final;
		if (img.depth() == CV_8U) {
			out.convertTo(out_final, CV_8U);
		} else {
			out.convertTo(out_final, CV_16U);
		}
		result_images.push_back(out_final);
		if (all_outs_names) all_outs_names->push_back(dir_e_to_str(LEFT_TO_RIGHT));
		result_grids.push_back(grid);
	}

    // ----------------- 右→左 -----------------
    if (dirc | RIGHT_TO_LEFT)
    {
		int grid_w = (w + bdr_width - 1) / bdr_width;
		int grid_h = (h + slice_width - 1) / slice_width;
        int grid_w_extra = (bdr_width - w % bdr_width) % bdr_width;
		Mat grid = Mat::zeros(grid_h, grid_w, CV_8U);

		out.setTo(0);
        for (int y0 = 0; y0 < h; y0 += slice_width)
        {
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
                if (xr1 > xr0)
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

					grid.at<uchar>(y0 / slice_width,
								  (x0 + grid_w_extra) / bdr_width) = 1;
                }
            }
        }

		Mat out_final;
		if (img.depth() == CV_8U) {
			out.convertTo(out_final, CV_8U);
		} else {
			out.convertTo(out_final, CV_16U);
		}
		result_images.push_back(out_final);
		if (all_outs_names) all_outs_names->push_back(dir_e_to_str(RIGHT_TO_LEFT));
		result_grids.push_back(grid);
    }

    // ----------------- 上→下 -----------------
    if (dirc | TOP_TO_BOTTOM)
    {
		int grid_w = (w + slice_width - 1) / slice_width;
		int grid_h = (h + bdr_width- 1) / bdr_width;
		Mat grid = Mat::zeros(grid_h, grid_w, CV_8U);

		out.setTo(0);
        for (int x0 = 0; x0 < w; x0 += slice_width)
        {
            int x1 = min(x0 + slice_width, w);
            bool col_set = false;
            for (int y0 = 0; y0 < h; y0 += bdr_width) {
                if (col_set) break;
                int y1 = min(y0 + bdr_width, h);

                Mat center_block = img_f(Range(y0, y1), Range(x0, x1));
                double center_mean = mean(center_block)[0];

                int yt0 = max(0, y0 - bdr_width);
                int yt1 = y0;
                double top_mean = 0;
                if (yt1 > yt0) top_mean = mean(img_f(Range(yt0, yt1), Range(x0, x1)))[0];

                if ((yt1 > yt0) && (center_mean > 0)
                    && ((top_mean / center_mean) < bdr_diff_r))
                {
                    out(Range(y0, y1), Range(x0, x1)) = center_mean;
                    out(Range(0, y0), Range(x0, x1)) = 0;
                    out(Range(y1, h), Range(x0, x1)) = 0;
                    col_set = true;

					grid.at<uchar>(y0 / bdr_width, x0 / slice_width) = 1;
                }

            }
        }

		Mat out_final;
		if (img.depth() == CV_8U) {
			out.convertTo(out_final, CV_8U);
		} else {
			out.convertTo(out_final, CV_16U);
		}
		result_images.push_back(out_final);
        if (all_outs_names) all_outs_names->push_back(dir_e_to_str(TOP_TO_BOTTOM));
		result_grids.push_back(grid);
    }

    // ----------------- 下→上 -----------------
    if (dirc | BOTTOM_TO_TOP)
    {
		int grid_w = (w + slice_width - 1) / slice_width;
		int grid_h = (h + bdr_width - 1) / bdr_width;
		int grid_h_extra = (bdr_width - h % bdr_width) % bdr_width;
		Mat grid = Mat::zeros(grid_h, grid_w, CV_8U);

		out.setTo(0);
        for (int x0 = 0; x0 < w; x0 += slice_width)
        {
            int x1 = min(x0 + slice_width, w);
            bool col_set = false;
            for (int y0 = h - bdr_width; y0 >= 0; y0 -= bdr_width) {
                if (col_set) break;
                int y1 = min(y0 + bdr_width, h);

                Mat center_block = img_f(Range(y0, y1), Range(x0, x1));
                double center_mean = mean(center_block)[0];

                int yb0 = min(h, y1);
                int yb1 = min(h, y1 + bdr_width);
                double bottom_mean = 0;
                if (yb1 > yb0) bottom_mean = mean(img_f(Range(yb0, yb1), Range(x0, x1)))[0];

                if ((yb1 > yb0) && (center_mean > 0)
                    && ((bottom_mean / center_mean) < bdr_diff_r))
                {
                    out(Range(y0, y1), Range(x0, x1)) = center_mean;
                    out(Range(0, y0), Range(x0, x1)) = 0;
                    out(Range(y1, h), Range(x0, x1)) = 0;
                    col_set = true;

					grid.at<uchar>((y0 + grid_h_extra) / bdr_width,
									x0 / slice_width) = 1;
                }

            }
        }

		Mat out_final;
		if (img.depth() == CV_8U) {
			out.convertTo(out_final, CV_8U);
		} else {
			out.convertTo(out_final, CV_16U);
		}
		result_images.push_back(out_final);
		if (all_outs_names) all_outs_names->push_back(dir_e_to_str(BOTTOM_TO_TOP));
		result_grids.push_back(grid);
    }

    if(result_images.size() == 0)
    {
		cout << "No direction selected!" << endl;
        return Mat();
	}
	int bestIdx = selectBestImage(result_images, result_grids);
    if (all_outs) *all_outs = result_images;
	if (best_idx) *best_idx = bestIdx;
	return result_images[bestIdx];
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

Mat vflip_mat(Mat& img)
{
    Mat dst(img.size(), img.type());
    int rows = img.rows;
    for (int y = 0; y < rows; y++)
    {
        img.row(rows - 1 - y).copyTo(dst.row(y));
    }
    return dst;
}

#define FOUR_DIR '1'
#define FOUR_DIR_FLIP 'F'
#define DRAW_WAVE_OTSU_8BIT '2'
#define DRAW_WAVE_OTSUL '3'

int main(int argc, char** argv)
{
    SetConsoleOutputCP(CP_UTF8);
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

    if (2 == argc || FOUR_DIR == argv[2][0] || FOUR_DIR_FLIP == argv[2][0])
    {
        int bdr_width = 20;      // 默认值
        int slice_width = 20;   // 默认值
        double bdr_diff_r = 0.9; // 默认值
        detect_dound_dir_e_t dirc = DIRC_FULL_BITS;
        string name_apx = "-my_edge04";

        if (argc >= 4) bdr_width = std::atoi(argv[3]);
        if (argc >= 5) slice_width = std::atoi(argv[4]);
        if (argc >= 6) bdr_diff_r = std::atof(argv[5]);
        if (argc >= 7) dirc = (detect_dound_dir_e_t)std::atoi(argv[6]);
        if (argc >= 8) name_apx += argv[7];

		const char* dirc_str = dir_e_to_str(dirc);
        if(!dirc_str)
        {
            cout << "Invalid direction: " << (int)dirc << endl;
            return -1;
		}

        std::ostringstream oss;
        oss << bdr_diff_r;
        name_apx += std::to_string(bdr_width) + "x"
            + std::to_string(slice_width) + std::string("x") + oss.str();

        if (FOUR_DIR_FLIP == argv[2][0])
        {
			img = vflip_mat(img);
            name_apx += "-flip";
        }

        std::vector<cv::Mat> all_outs;
        std::vector<std::string> all_outs_names;
		int best_idx;
        Mat boundary = detect_boundaries_block_4dir(img, bdr_width,
												slice_width, bdr_diff_r, dirc,
            &all_outs, &all_outs_names, &best_idx);
        if (boundary.empty())
        {
			cout << "No boundary detected!" << endl;
            return 0;
        }
        
        for (int i = 0; i < all_outs.size(); i++)
        {
            string fn = make_new_filename(img_fn,
                name_apx + "-" + std::to_string(i) + "_" + all_outs_names[i]);
			Mat edges;
			convertScaleAbs(all_outs[i], edges); // 可视化
            imwrite(fn, edges);
            cout << "Saved result: " << fn << endl;
        }
		string dst_fn = make_new_filename(img_fn, name_apx);

        cout << endl;
		cout << "The selected one : " << best_idx << ", "
			<< all_outs_names[best_idx] << endl;
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
