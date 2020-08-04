//**************************************************************************
//--------------------------------------------------------------------------
// Set weightlift Paramenter by HSV
//
// Sharon
// version 1.0
// Date    2017.9.29
//--------------------------------------------------------------------------
//**************************************************************************


#include <opencv2/opencv.hpp>
#include <iostream>

#include "find_ball.h"


using std::cout;
using std::endl;

using cv::Mat;
using cv::waitKey;
using cv::imshow;

using RedContritio::BallInfo;
using RedContritio::BallFinder;

void createHSVPanel(BallFinder& processor);

int main(int argc, char *argv[])
{
    cv::VideoCapture cap(0);
    // cv::VideoCapture cap("my_video-2.mkv");

    Mat image;
    
    BallFinder processor;

    bool auto_play = false;

    double fps = 30;
    char ch;

    int sl_count = 0;

    // auto_play = true;

    if (!cap.isOpened())
    {
        printf("video stream read failed.\n");
        return -1;
    }

    // namedWindow("src_img", cv::WindowFlags::WINDOW_AUTOSIZE);
    // namedWindow("dst_img", cv::WindowFlags::WINDOW_AUTOSIZE);
    // cv::resizeWindow("src_img", 320, 250);
    BallInfo result;

    while (true)
    {
        cap >> image;

        if (image.empty())
        {
            printf("image not load\n");
            waitKey();
            break;
        }

        imshow("src_img", image);

        
        processor.imageProcess(image, &result);

        // cout << image.size << endl;
        // a.imageProcess(image);

        createHSVPanel(processor);

        Mat dst = image.clone();
        cv::circle(dst, result.center, result.radius, cv::Scalar(255, 255, 0), 10);
        imshow("dst_img", dst);
        cout << result.center << ' ' << result.radius << endl;

        // imshow("edge_img", processor.edge_img);
        // cv::createTrackbar("thre_1", "edge_img", &processor.canny_thre.start, 300);
        // cv::createTrackbar("thre_2", "edge_img", &processor.canny_thre.end, 300);

        if (auto_play)
        {
            ch = waitKey(1000/fps);
        }
        else
        {
            ch = waitKey(0);
        }

        switch (ch)
        {
            case 'q':
                cout << "quit" << endl;
                goto END_LOOP;
            case 's':
                sl_count = processor.SaveParameter();
                cout << "saved " << sl_count << endl;
                break;
            case 'l':
                sl_count = processor.LoadParameter();
                cout << "loaded " << sl_count << endl;
                break;
            case 'a':
                auto_play = !auto_play;
                break;
            default:
                break;
        }

    }

END_LOOP:
    return 0;
}


void createHSVPanel(BallFinder& processor)
{
    cv::Mat show_img;

    std::vector<cv::Mat> mv;
    for(int i=0; i<3; ++i)
    {
        cv::Mat tmp;
        cv::pyrDown(processor.hsv_thre_channels[i], tmp);
        mv.push_back(tmp);
    }
    
    cv::hconcat(mv, show_img);
    imshow("thre_img", show_img);

    imshow("th_and_img", processor.th_and_img);

    cv::createTrackbar("h_min", "thre_img", &processor.hsv_thre[0].start, 256);
    cv::createTrackbar("h_max", "thre_img", &processor.hsv_thre[0].end, 256);
    cv::createTrackbar("s_min", "thre_img", &processor.hsv_thre[1].start, 256);
    cv::createTrackbar("s_max", "thre_img", &processor.hsv_thre[1].end, 256);
    cv::createTrackbar("v_min", "thre_img", &processor.hsv_thre[2].start, 256);
    cv::createTrackbar("v_max", "thre_img", &processor.hsv_thre[2].end, 256);
}