#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include <fstream>
#include <iostream>

#include "find_ball.h"

namespace RedContritio {

double distance(double x1, double y1, double x2, double y2)
{
    return sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
}

cv::Mat bitwise_and(std::vector<cv::Mat> mv);

int BallFinder::BLUR_SIZE = 9;

cv::Range BallFinder::canny_thre(60, 80);

cv::Vec<cv::Range, 3> BallFinder::hsv_thre({
    cv::Range(0, 255),
    cv::Range(0, 255),
    cv::Range(0, 255)});

BallFinder::BallFinder(void)
{

}

void BallFinder::imageProcess(cv::Mat image, BallInfo *pResult)
{
    
    Pretreat(image);

    if (!pResult)
        pResult->radius = -1;
    else
        *pResult = GetPosition(only_img);
}

// 预处理
cv::Mat BallFinder::Pretreat(cv::Mat src) 
{
    cv::cvtColor(src, hsv_src, CV_RGB2HSV);
    cv::split(hsv_src, hsv_channels);

    hsv_thre_channels.clear();

    for(int i=0; i<3; ++i)
    {
        cv::Mat thre_channel;

        
        cv::threshold(thre_channel, thre_channel, hsv_thre[i].end, 255, CV_THRESH_TOZERO_INV);
        cv::threshold(hsv_channels[i], thre_channel, hsv_thre[i].start, 255, CV_THRESH_BINARY);
        hsv_thre_channels.push_back(thre_channel);
    }

    th_and_img = bitwise_and(hsv_thre_channels);
    
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(th_and_img, contours, CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);

    double maxArea = 0;  
    int maxAreaId = 0; 

    for(size_t i = 0; i < contours.size(); i++)  
    {  
        double area = cv::contourArea(contours[i]);  
        if (area > maxArea)  
        {  
            maxArea = area;  
            maxAreaId = i;
        }  
    }

    only_img = cv::Mat::zeros(th_and_img.size(), CV_8UC1);
    cv::drawContours(only_img, contours, maxAreaId, cv::Scalar(1, 1, 1), -1);

    cv::merge(hsv_thre_channels, thre_img);

    cv::cvtColor(thre_img, thre_img, CV_HSV2RGB);
    cv::cvtColor(thre_img, gray_src, CV_RGB2GRAY);
    // GaussianBlur(hsv_src, gray_blur, cv::Size(BLUR_SIZE, BLUR_SIZE), 2, 2);
    GaussianBlur(gray_src, gray_blur, cv::Size(BLUR_SIZE, BLUR_SIZE), 0);
    cv::Canny(gray_blur, edge_img, canny_thre.start, canny_thre.end);
    return edge_img;
}

BallInfo BallFinder::GetPosition(cv::Mat bin)
{
    int sx = 0, sy = 0, cnt = 0;
    for(int i=0; i<bin.rows; ++i)
    {
        for(int j=0; j<bin.cols; ++j)
        {
            if(bin.at<uchar>(i, j))
            {
                sx += j;
                sy += i;
                ++cnt;
            }
        }
    }

    double cx = 1.0 * sx / cnt, cy = 1.0 * sy / cnt;
    
    double sr = 0;
    for(int i=0; i<bin.rows; ++i)
    {
        for(int j=0; j<bin.cols; ++j)
        {
            if(bin.at<uchar>(i, j))
            {
                sr += distance(j, i, cx, cy);
            }
        }
    }

    return BALLINFO(cx, cy, sr/cnt);
}

int BallFinder::LoadParameter(void)
{
    int cnt = 0;
    std::ifstream ifs;
    ifs.open("param.txt", std::_Ios_Openmode::_S_in);
    
    if (!(ifs >> this->canny_thre.start))
        goto END_LOAD;
    ++cnt;
    if (!(ifs >> this->canny_thre.end))
        goto END_LOAD;
    ++cnt;

    for (int i = 0; i < 3; ++i)
    {
        if (!(ifs >> this->hsv_thre[i].start))
            goto END_LOAD;
        ++cnt;
        if (!(ifs >> this->hsv_thre[i].end))
            goto END_LOAD;
        ++cnt;
    }

END_LOAD:
    ifs.close();
    return cnt;
}

int BallFinder::SaveParameter(void)
{
    int cnt = 0;
    std::ofstream ofs;
    ofs.open("param.txt", std::_Ios_Openmode::_S_out);

    ofs << this->canny_thre.start << ' '
        << this->canny_thre.end << std::endl;
    cnt += 2;

    for (int i = 0; i < 3; ++i)
    {
        ofs << this->hsv_thre[i].start << ' '
            << this->hsv_thre[i].end << std::endl;
        cnt += 2;
    }

    ofs.close();
    return cnt;
}

cv::Mat bitwise_and(std::vector<cv::Mat> mv)
{
    std::vector<cv::Mat>::iterator it = mv.begin();
    cv::Mat m = it->clone();
    ++it;

    while(it != mv.end())
    {
        cv::bitwise_and(m, *it, m);
        ++it;
    }
    return m;
}

}