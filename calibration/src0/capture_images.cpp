#include "iostream"
#include "cstring"
#include "opencv2/opencv.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

int main()
{
	cv::VideoCapture capture(0);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
	capture.set(CV_CAP_PROP_FRAME_WIDTH, 1280);

	cv::Mat frame;
	string s1 = "/home/lcl/Robocup3DVision/calibration/src0/cap_image/";
	string s2 = ".jpg";
	int camera_ids = 1;
	while(capture.grab())
	{
		capture.retrieve(frame);
		cv::imshow("frame", frame);
		char s = cv::waitKey(1);
		if(s == 'c')
		{
			char s3[2];
			sprintf(s3, "%d", camera_ids);
			string path = s1 + s3 + s2;
			cout<<"catch image"<<camera_ids<<" to "<<path<<endl;
			camera_ids ++ ;
			cv::imwrite(path, frame);
		}

		else if( s == 'q')
			break;
	}
	return 0;
}
