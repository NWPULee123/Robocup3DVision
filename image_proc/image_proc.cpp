#include <iostream>
#include <cstring>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main()
{
	string path = "/home/lcl/Robocup3DVision/calibration/src0/cap_image/1.jpg";
	cv::Mat image = cv::imread(path, -1);
	cv::imshow("ori_image", image);
	cv::Mat blur_image, usm_image;

	// cv::Mat image_threshold;
	// cv::cvtColor(image, image_threshold, CV_BGR2GRAY);
	// cv::adaptiveThreshold(image_threshold, image_threshold, 255,CV_ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY_INV, this->thres_paramter,7 );
	// cv::imshow("threshold", image_threshold);
	// cv::imshow("image", image);

	//边缘增强
	cv::Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, -1, 5.5, -1, 0, -1, 0);
	cv::filter2D(image, image, image.depth(), kernel);
		cv::imshow("image", image);
	cv::GaussianBlur(image, blur_image, cv::Size(0,0), 15);
	cv::addWeighted(image, 2, blur_image, -1, 0, usm_image);
	cv::addWeighted(usm_image, 1.5, image, -0.5, 0, usm_image);
	cout<<"xxx"<<endl;
	cv::imshow("usm_image", usm_image);
	cv::waitKey(0);
	return 0;
}
