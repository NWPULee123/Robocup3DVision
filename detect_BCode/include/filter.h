#ifndef FILTER_H
#define FILTER_H

#include "iostream"
#include "cmath"
#include "opencv2/opencv.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

class UpperclassFilter
{
	public:
		int x_number;
		int time_step;
		vector<double> new_data;
		vector<double> effective_data;
	
	public:
		UpperclassFilter(int time_step = 1, int x_number = 5);
		~UpperclassFilter(){};
		void SetTimeStep(int t);
		void SetXNumber(int n);
		void ProcessNewData(double d);
		//void 
	
	public:
		void PushNewData(double d);
		void PushEffectiveData(double d);
		void CreateRegressionData(vector<vector<double>> &x_data, vector<double> &y_data);
		bool isValid(double distance);
		bool isEffective(double y_predicted, double input_data);
		cv::Mat LinerRegression(vector<vector<double>> x_data, vector<double> y_data);
};

#endif