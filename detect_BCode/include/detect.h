#ifndef DETECT_H
#define DETECT_H

#include "iostream"
#include "cmath"
#include "opencv2/opencv.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

class Detector
{
	public:
		int mark_width;
		int mark_number;
		int thres_paramter;
		cv::Mat result_image;
		vector<cv::Point2f> outer_result;
		vector<cv::Point2f> inner_result;
	
	public:
		void DetectCorners(cv::Mat image);
		void SetThresParamter(int m);
		void SetThresParamter(double distance);
		vector<cv::Point2f> GetDetectResult();
		void ShowResultImage();
	
	public:
		Detector(int width=4, int number=4, int paramter=23);
		~Detector(){};
		void GetIniContours(cv::Mat image, vector<vector<cv::Point>> &contours, vector<cv::Vec4i> &hierarchy);
		bool TestImageCode(cv::Mat transform_image);
		void ClockwiseSort(vector<Point2f> &src, vector<Point> contour);	
};

#endif
