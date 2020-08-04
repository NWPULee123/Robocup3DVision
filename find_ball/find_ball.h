

#ifndef FIND_BALL_H
#define FIND_BALL_H

#include <opencv2/opencv.hpp>

namespace RedContritio
{

#define BALLINFO(x, y, r) ((BallInfo){cv::Point2d(x, y), r})

struct BallInfo
{
    cv::Point2d center;
    double radius;
};

// 寻找给定图像中球体的球心
class BallFinder
{
private:
    BallInfo GetPosition(cv::Mat bin);
    cv::Mat Pretreat(cv::Mat src);
public:
    static cv::Range canny_thre;
    static cv::Vec<cv::Range, 3> hsv_thre;
    static int BLUR_SIZE;

    cv::Mat hsv_src;
    cv::Mat thre_img;
    cv::Mat gray_src;
    cv::Mat gray_blur;
    cv::Mat edge_img;

    // 进行过 hsv 阈值后，各通道与运算的结果
    cv::Mat th_and_img;

    cv::Mat only_img;

    std::vector<cv::Mat> hsv_channels;
    std::vector<cv::Mat> hsv_thre_channels;
    BallFinder(void);
    void imageProcess(cv::Mat image, BallInfo *pResult);

    int LoadParameter(void);
    int SaveParameter(void);
};

}

#endif