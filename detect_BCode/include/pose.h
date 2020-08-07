#ifndef POSE_H
#define POSE_H

#include <iostream>
#include <cmath>
#include <cstring>
#include "opencv2/opencv.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

class Pose
{
	public:
		Pose();
		~Pose(){};


	public:
		cv::Mat camera_R_vec;
		cv::Mat camera_t_vec;
		void InitializeParam();
		void GetRobotPoseInWorld(cv::Vec2f &direction_vec, cv::Point2f &pos, cv::Mat R_rw, cv::Mat t_rw, bool debug = true);
		void GetBallPositionInWorld(cv::Point2f detect_result, cv::Point2f &ball_position, bool debut = true);
		void TransformRobotToWorld(vector<Point2f> detect_result, cv::Mat &R_rw, cv::Mat &t_rw, bool debug = true);
		cv::Vec2f RotationMatrixToAngles(cv::Mat R);
		double GetDistanceToMark();


	protected:
		double mark_length;
		double ball_diameter;
		double distance_to_mark;
		std::string camera_param;
		std::string external_param;
		cv::Mat camera_matrix;
		cv::Mat dist_coef;
		cv::Mat camera_T;
		cv::Mat camera_T_34;
		void Create3dPoints(vector<cv::Point3d> &pos_3d, double d);
		void VectorConvert(vector<cv::Point2f> src, vector<cv::Point2d> &dst);
		void reprojection(cv::Mat camera_matrix, cv::Mat R_wc, cv::Mat t_wc, cv::Point2f p_2d, double z_3d, cv::Point3f & p_3d);
};

#endif