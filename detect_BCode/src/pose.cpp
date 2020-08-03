#include "pose.h"

using namespace std;
using namespace cv;

//rw is for Robot to World
//wc is for World to Camera
//rc is for Robot to Camera
//用矩阵之前随手convertTo成float型

Pose::Pose()
{
	mark_length = 0.0865; //meter
	ball_diameter = 0;
	distance_to_mark = 2.0;
	camera_param = "../bin/camera_param.yaml";
	external_param = "../bin/ex_param.yaml";
	camera_matrix = cv::Mat::eye(3,3,CV_32F);
	dist_coef= cv::Mat::zeros(3,1,CV_32F);
	camera_R_vec = cv::Mat::zeros(3,1,CV_32F);
	camera_t_vec = cv::Mat::zeros(3,1,CV_32F);
	camera_T = cv::Mat::eye(3,3,CV_32F);
	camera_T_34 = cv::Mat::eye(3,4,CV_32F);
	optimizer.resize(4);
}

void Pose::InitializeParam()
{
	cv::FileStorage fs_camera(this->camera_param, FileStorage::READ);
	cv::FileStorage fs_external(this->external_param, FileStorage::READ);
    if(!fs_camera.isOpened())
	{
		cout<<"Failed to open file : camera_param"<<endl;
		return ;
	}
	if(!fs_external.isOpened())
	{
		cout<<"Failed to open file : external_param"<<endl;
		return ;
	}
	fs_camera["camera_matrix"] >> this->camera_matrix;
	fs_camera["distortion_coefficients"] >> this->dist_coef;
	fs_external["vector_R_w2c"] >> this->camera_R_vec;
	fs_external["vector_t_w2c"] >> this->camera_t_vec;
	fs_external["matrix_T_w2c"] >> this->camera_T;
	fs_external["matrix_T_w2c_34"] >> this->camera_T_34;
}

void Pose::GetRobotPoseInWorld(cv::Vec2f &direction_vec, cv::Point2f &pos, cv::Mat R_rw, cv::Mat t_rw, bool debug)
{
	//旋转矩阵 -> 比赛世界坐标系中mark与x轴夹角
	direction_vec = RotationMatrixToAngles(R_rw);
	//世界坐标系XOY平面中机器人坐标(debug用的世界坐标系)
	//世界坐标系位于相机正下方，XOY与地面重合
	pos = cv::Point2f(t_rw.at<float>(0), t_rw.at<float>(1));
	//比赛时的世界坐标系，原点为机器人起点
	if(!debug)
	{
		pos.x = -pos.x;
		pos.y = 4.50 - pos.y;
	}
}

void Pose::GetBallPositionInWorld(vector<Point2f> detect_result, cv::Point2f &ball_position, bool debug)
{
	cv::Mat R_vec_wc = this->camera_R_vec.clone(), t_wc = this->camera_t_vec.clone(), R_wc;
	cv::Rodrigues(R_vec_wc, R_wc);
	R_wc.convertTo(R_wc, CV_32F);
	t_wc.convertTo(t_wc, CV_32F);
	for(int i=0; i<detect_result.size(); i++)
	{
		cv::Point2f p_2d = detect_result[i];
		cv::Point3f p_3d;
		if(i < 2)
			reprojection(camera_matrix, R_wc, t_wc, p_2d, this->ball_diameter, p_3d);
		else
			reprojection(camera_matrix, R_wc, t_wc, p_2d, 0, p_3d);
		ball_position.x += p_3d.x;
		ball_position.y += p_3d.y;
	}
	ball_position.x /= 4;
	ball_position.y /= 4;
	if(!debug)
	{
		ball_position.x = -ball_position.x;
		ball_position.y = 4.50 - ball_position.y;
	}
}

void Pose::TransformRobotToWorld(vector<Point2f> detect_result, cv::Mat &R_rw, cv::Mat &t_rw, bool debug)
{
	vector<cv::Point3d> pos_3d(4);
	Create3dPoints(pos_3d, this->mark_length);
	vector<cv::Point2d> pos_2d(4);
	VectorConvert(detect_result, pos_2d);

	//解robot to camera
	cv::Mat camera_matrix = this->camera_matrix.clone();
	camera_matrix.convertTo(camera_matrix, CV_32F);
	cv::Mat dist_coef = this->dist_coef.clone();
	cv::Mat R_vec_rc, t_rc, R_rc;
	cv::solvePnP(pos_3d, pos_2d, camera_matrix, dist_coef, R_vec_rc, t_rc, false);
	cv::Rodrigues(R_vec_rc, R_rc);	//PnP解出来的是旋转向量，需要转换
	R_rc.convertTo(R_rc, CV_32F);
	t_rc.convertTo(t_rc, CV_32F);

	//world to camera，即标定的外参
	cv::Mat R_vec_wc = this->camera_R_vec.clone(), t_wc = this->camera_t_vec.clone(), R_wc;
	cv::Rodrigues(R_vec_wc, R_wc);
	R_wc.convertTo(R_wc, CV_32F);
	t_wc.convertTo(t_wc, CV_32F);

	//联立w to c 和 r to c即可得到r to w
	R_rw = R_wc.t()*R_rc;
	t_rw = R_wc.t()*(t_rc - t_wc);
	R_rw.convertTo(R_rw, CV_32F);
	t_rw.convertTo(t_rw, CV_32F);

	//保存距离，以供detect模块实时调整阈值化参数
	double distance = 0;
	for(int i=0; i<3; i++)
		distance += t_rc.at<float>(i)*t_rc.at<float>(i);
	distance = sqrt(distance);
	this->distance_to_mark = distance;

	if(debug)
		cout<<"distance : "<<distance<<endl;
}

//mark的三维坐标
void Pose::Create3dPoints(vector<cv::Point3d> &pos_3d, double d)
{
	pos_3d[0] = cv::Point3d(d/2.d, -d/2.d, 0);
	pos_3d[1] = cv::Point3d(-d/2.d, -d/2.d, 0);
	pos_3d[2] = cv::Point3d(-d/2.d, d/2.d, 0);
	pos_3d[3] = cv::Point3d(d/2.d, d/2.d, 0);
}

void Pose::VectorConvert(vector<cv::Point2f> src, vector<cv::Point2d> &dst)
{
	for(int i=0; i<src.size(); i++)
		dst[i] = cv::Point2d(src[i]);
}


//每天一个生活小妙招
cv::Vec2f Pose::RotationMatrixToAngles(cv::Mat &R)
{
	cv::Mat rotated_vec(3,1,CV_32F);
	cv::Mat unit_vec (3,1,CV_32F);
	unit_vec.at<float>(0) = 1;
	unit_vec.at<float>(1) = 0;
	unit_vec.at<float>(2) = 0;
	rotated_vec = R*unit_vec;

	cv::Vec2f direction_vector;
	direction_vector[0] = rotated_vec.at<float>(1);
	direction_vector[1] = rotated_vec.at<float>(0)<0 ? -rotated_vec.at<float>(0) : rotated_vec.at<float>(0);
	//转为单位向量
	double length = sqrt(direction_vector[0]*direction_vector[0] + direction_vector[1]*direction_vector[1]);
	direction_vector[0] /= length;
	direction_vector[1] /= length;
	//double angles = -atan2(rotated_vec.at<float>(1), -rotated_vec.at<float>(0)) * 180.0f/3.141592653589793f;
    return direction_vector;
}

/**
 * 像素平面的点投影到世界坐标系，用于计算球的位置
 * (不用这个方法算机器人位姿，因为无法得到R)
 * 		 |  u  |				    |	X	|	
 * Zc |	 v  |	=	 K( R	|	Y	|	+ 	t)
 * 		 |  1  |				   |	Z	|
 * u, v, Z, K, R, t已知， 移项解Zc, X, Y
 * */

void Pose::reprojection(cv::Mat camera_matrix, cv::Mat R_wc, cv::Mat t_wc, cv::Point2f p_2d, double z_3d, cv::Point3f & p_3d)
{
	cv::Mat K_inv;
	cv::invert(camera_matrix,K_inv);
	cv::Mat a(3,1,CV_32F);
	cv::Mat b(3,1,CV_32F);
	cv::Mat c(3,1,CV_32F);
	cv::Mat d(3,1,CV_32F);
	for(int i=0; i<3; i++)
	{
		a.at<float>(i) = R_wc.at<float>(i,0);
		b.at<float>(i) = R_wc.at<float>(i,1);
		c.at<float>(i) = -(K_inv.at<float>(i,0)*p_2d.x + K_inv.at<float>(i,1)*p_2d.y + K_inv.at<float>(i,2));
		d.at<float>(i) = -(R_wc.at<float>(i,2)*z_3d + t_wc.at<float>(i));
	}
	cv::Mat A(3,3,CV_32F);
	cv::Mat B(3,1,CV_32F);
	a.copyTo(A.rowRange(0,3).col(0));
	b.copyTo(A.rowRange(0,3).col(1));
	c.copyTo(A.rowRange(0,3).col(2));
	d.copyTo(B.rowRange(0,3).col(0));
	cv::Mat x(3,1,CV_32F, CV_SVD);
	cv::solve(A,B,x);
	float X=x.at<float>(0),Y=x.at<float>(1),z=x.at<float>(2);
	p_3d.x = X;
	p_3d.y = Y;
	p_3d.z = z_3d;
}

double Pose::GetDistanceToMark()
{
	return this->distance_to_mark;
}
