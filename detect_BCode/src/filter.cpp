#include "filter.h"

UpperclassFilter::UpperclassFilter(int time_step, int x_number):
time_step(time_step), x_number(x_number)
{
	new_data.resize(time_step);
	effective_data.resize(time_step + x_number);
}

void UpperclassFilter::ProcessNewData(double d)
{
	if(!isValid(d))
		return;
	PushNewData(d);
	if(this->effective_data.size()<this->x_number+this->time_step)
		PushEffectiveData(d);

	cv::Mat W(this->time_step, 1, CV_32F);
	vector<vector<double>> x_data;
	vector<double> y_data;
	W = LinerRegression(x_data, y_data);

	cv::Mat x_in(1, this->time_step, CV_32F);
	x_in.at<float>(0) = d;
	for(int i=1; i<this->time_step; i++)
		x_in.at<float>(i) = x_data[0][i-1];
	cv::Mat y_predicted = x_in * W;
	if(isEffective(y_predicted.at<double>(0),d))
		PushEffectiveData(y_predicted.at<double>(0));
}

void UpperclassFilter::PushNewData(double d)
{
	if(this->new_data.size()<this->time_step)
		this->new_data.push_back(d);
	else
	{
		this->new_data.erase(this->new_data.begin());
		this->new_data.push_back(d);
	}
}

void UpperclassFilter::PushEffectiveData(double d)
{
	if(this->effective_data.size() < this->x_number+this->time_step)
		this->effective_data.push_back(d);
	else
	{
		this->effective_data.erase(this->effective_data.begin());
		this->effective_data.push_back(d);
	}
}

bool UpperclassFilter::isValid(double distance)
{
	return (distance>0.15 && distance<4);
}

bool UpperclassFilter::isEffective(double y_predict, double input_data)
{
	double diff = y_predict - input_data;
	double y_previous = this->effective_data.front();
	// if(diff > 0.01)
}

void UpperclassFilter::CreateRegressionData(vector<vector<double>> &x_data, vector<double> &y_data)
{
	for(int i=0; i<this->x_number; i++)
	{
		for(int j=1; j<=this->time_step; j++)
		{
			x_data[i].push_back(this->effective_data[this->effective_data.size()-i-j-1]);
		}
		y_data.push_back(this->effective_data[this->effective_data.size()-i-1]);
	}
}

cv::Mat UpperclassFilter::LinerRegression(vector<vector<double>> x_data, vector<double> y_data)
{
	cv::Mat X(this->x_number, this->time_step+1, CV_32F);
	cv::Mat Y(this->x_number,1, CV_32F);
	cv::Mat W(this->time_step+1, 1, CV_32F);
	for(int i=0; i<this->x_number; i++)
	{
		for(int j=0; j<this->time_step; j++)
			X.at<float>(i,j) = x_data[i][j]; 
		X.at<float>(i,this->time_step) = 1.f;
		Y.at<float>(i) = y_data[i];
	}
	cv::Mat xTx = X.t()*X;
	cv::invert(xTx, xTx);
	W = xTx*X.t()*Y;
	return W;
}

void UpperclassFilter::SetTimeStep(int t)
{
	this->time_step = t;
}

void UpperclassFilter::SetXNumber(int n)
{
	this->x_number = n;
}

