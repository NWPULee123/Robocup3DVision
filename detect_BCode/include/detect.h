#ifndef DETECT_H
#define DETECT_H

#include "iostream"
#include "cmath"
#include "opencv2/opencv.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include "filter.h"

using namespace std;
using namespace cv;

namespace RectDetect
{
class Detector
{
	public:
		Detector(int width=4, int number=4, int paramter=23);
		~Detector(){};
		int thres_paramter;
		vector<cv::Point2f> outer_result;
		vector<cv::Point2f> inner_result;
	
	public:
		void DetectCorners(cv::Mat image);
		vector<cv::Point2f> GetDetectResult();
		void SetThresParamter(int m);
		void SetParamter(double distance);
		void ShowResultImage();
	
	private:
		int mark_width;
		int mark_number;
		double USM_paramter;
		cv::Mat result_image;

		//for detection
		void GetIniContours(cv::Mat image, vector<vector<cv::Point>> &ori_contours, vector<vector<cv::Point>> &contours, vector<cv::Vec4i> &hierarchy);
		bool TestImageCode(cv::Mat transform_image);
		void ClockwiseSort(vector<Point2f> &src, vector<Point> contour);

		//for liner regression
		void GetLinerRegressionWeights(vector<Point> ori_contours, cv::Mat &Weights);
		void CreateFilterData(vector<cv::Point> ori_contours, vector<vector<double>> &input_x, vector<vector<double>> &input_y);
		void Getcorners(cv::Mat Weights, vector<cv::Point2d> &caculated_corners);
		bool isWeightsValid(cv::Mat Weights);
};

}

#endif
