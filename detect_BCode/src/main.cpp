#include "detect.h"
#include "pose.h"


using namespace cv;
using namespace std;

void ContinueslyTest(cv::Mat frame , ofstream &fout, double &distance, int &count, cv::Point2f &avg_pos)
{
	Detector *mark_detector = new Detector();
	mark_detector->SetThresParamter(distance);
	mark_detector->DetectCorners(frame);
	vector<cv::Point2f> detect_result = mark_detector->GetDetectResult();

	Pose *pose_estimate = new Pose();
	pose_estimate->InitializeParam();
	cv::Vec2f direction_vec;
	cv::Point2f pos;
	cv::Mat R_rw, t_rw;
	pose_estimate->TransformRobotToWorld(detect_result, R_rw, t_rw, false);
	pose_estimate->GetRobotPoseInWorld(direction_vec, pos, R_rw, t_rw);
	distance = pose_estimate->GetDistanceToMark();

	cout<<"pos : "<<pos<<'\t'<<direction_vec<<'      \t'<<"param : "<<mark_detector->thres_paramter<<endl;
	fout<<"pos : "<<pos<<'\t'<<direction_vec<<'      \t'<<"param : "<<mark_detector->thres_paramter<<endl;
	// count ++;
	// avg_pos += pos;
	// if(count%4 == 0)
	// {
	// 	avg_pos /= 4;
	// 	cout<<"pos : "<<avg_pos<<'\t'<<"param : "<<mark_detector->thres_paramter<<endl;
	// 	fout<<"pos : "<<avg_pos<<endl;
	// 	avg_pos = cv::Point2f(0,0);
	// }
}


int main(int argc, char *argv[])
{
	int adaptiveThreshWinSizeMax=23;

	cv::VideoCapture capture(2);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
	capture.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
	cv::Mat frame;
	ofstream fout;
	double distance = 2.0;
	int test_count = 0;
	cv::Point2f avg_pos(0,0);
	fout.open("../bin/distance.txt");
	if ( ! fout)
		cout << "Failed to open files" <<endl;
	while(capture.grab())
	{
		capture.retrieve(frame);
		cv::imshow("frame",frame);
		if(argc > 1)
			ContinueslyTest(frame, fout, distance, test_count, avg_pos);
		else
		{
			char s = cv::waitKey(1);
			if(s == 'c')
			{
				Detector *mark_detector = new Detector();
				mark_detector->SetThresParamter(adaptiveThreshWinSizeMax);
				mark_detector->DetectCorners(frame);
				mark_detector->ShowResultImage();
				vector<cv::Point2f> detect_result = mark_detector->GetDetectResult();

				Pose *pose_estimate = new Pose();
				pose_estimate->InitializeParam();
				cv::Vec2f direction_vec;
				cv::Point2f pos;
				cv::Mat R_rw, t_rw;
				pose_estimate->TransformRobotToWorld(detect_result, R_rw, t_rw);
				pose_estimate->GetRobotPoseInWorld(direction_vec, pos, R_rw, t_rw);

				cout<<"direction_vec : "<<direction_vec<<endl<<"pos : "<<pos<<endl;
			}
			else if(s == 'q')
			break;
		}
	}
	fout.close();
	return 0;
}
