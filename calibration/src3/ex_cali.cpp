#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include <iostream>
#include <ctime>

using namespace std;
using namespace cv;

int  X_direction = 7;
int  Y_direction = 5;
int aruco_board_type =  1;
double square_length = 0.0365;
double mark_length = 0.0279;
string camera_params_file = "../bin/camera_param.yaml";
string detector_params_file = "../bin/detector_params.yml";
string output_file = "./ex_param.yaml";
int CAMERA_ID = 2;

bool readCameraParams(cv::Mat &camera_matrix, cv::Mat &dist_coef)
{	
	cv::FileStorage fs(camera_params_file, FileStorage::READ);
	if(!fs.isOpened())
	{
		cout<<"Failed to load camera_params file";
		return false;
	}
	fs["camera_matrix"] >>camera_matrix;
	fs["distortion_coefficients"] >> dist_coef;
	return true;
}

 void readDetectorParameters(Ptr<aruco::DetectorParameters> &params) 
{
    cv::FileStorage fs(detector_params_file, FileStorage::READ);
    if(!fs.isOpened())
	{
		cout<<"Failed to load detector_params file."<<endl;
		return ;
	}
    fs["adaptiveThreshWinSizeMin"] >> params->adaptiveThreshWinSizeMin;
    fs["adaptiveThreshWinSizeMax"] >> params->adaptiveThreshWinSizeMax;
    fs["adaptiveThreshWinSizeStep"] >> params->adaptiveThreshWinSizeStep;
    fs["adaptiveThreshConstant"] >> params->adaptiveThreshConstant;
    fs["minMarkerPerimeterRate"] >> params->minMarkerPerimeterRate;
    fs["maxMarkerPerimeterRate"] >> params->maxMarkerPerimeterRate;
    fs["polygonalApproxAccuracyRate"] >> params->polygonalApproxAccuracyRate;
    fs["minCornerDistanceRate"] >> params->minCornerDistanceRate;
    fs["minDistanceToBorder"] >> params->minDistanceToBorder;
    fs["minMarkerDistanceRate"] >> params->minMarkerDistanceRate;
    fs["cornerRefinementMethod"] >> params->cornerRefinementMethod;
    fs["cornerRefinementWinSize"] >> params->cornerRefinementWinSize;
    fs["cornerRefinementMaxIterations"] >> params->cornerRefinementMaxIterations;
    fs["cornerRefinementMinAccuracy"] >> params->cornerRefinementMinAccuracy;
    fs["markerBorderBits"] >> params->markerBorderBits;
    fs["perspectiveRemovePixelPerCell"] >> params->perspectiveRemovePixelPerCell;
    fs["perspectiveRemoveIgnoredMarginPerCell"] >> params->perspectiveRemoveIgnoredMarginPerCell;
    fs["maxErroneousBitsInBorderRate"] >> params->maxErroneousBitsInBorderRate;
    fs["minOtsuStdDev"] >> params->minOtsuStdDev;
    fs["errorCorrectionRate"] >> params->errorCorrectionRate;
    return;
}

void saveCameraParams(cv::Mat R_vector, cv::Mat t, cv::Mat T, cv::Mat T34) 
{
    FileStorage fs(output_file, FileStorage::WRITE);
    if(!fs.isOpened())
	{
		cout<<"Failed to load file . "<<endl;
		return ;
	}
	fs << "vector_R_w2c" << R_vector;
	fs << "vector_t_w2c" << t;
    fs << "matrix_T_w2c" << T;
	fs << "matrix_T_w2c_34" << T34;
}

void detectcorners(cv::Mat frame, int dictionary_id,  vector<vector<cv::Point2f>> &corners, vector<int> &ids, cv::Mat &corner_pos, cv::Mat &ids_pos)
{
	cv::Ptr<cv::aruco::Dictionary>  dict = cv::aruco::getPredefinedDictionary(dictionary_id);
	cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();;
	readDetectorParameters(parameters);
	cv::aruco::detectMarkers(frame, dict, corners, ids, parameters);
	Ptr<aruco::CharucoBoard> board = aruco::CharucoBoard::create(X_direction, Y_direction, (float)square_length,
                                                            (float)mark_length, dict);
	cv::aruco::interpolateCornersCharuco(corners, ids, frame, board, corner_pos, ids_pos);
}

void getCorrespondingPoints(cv::Mat corner_pos, cv::Mat ids_pos, vector<cv::Point2f> &pos_2d,  vector<cv::Point3f> &pos_3d)
{
	for(int i=0; i<corner_pos.rows; i++)
	{
		cv::Point2f point2d = cv::Point2f(corner_pos.at<float>(i,0), corner_pos.at<float>(i,1));
		int row = ids_pos.at<int>(i)/(X_direction-1);
		int col = ids_pos.at<int>(i)%(X_direction-1);
		cv::Point3f point3d = cv::Point3f(square_length*col, 0, square_length*row+0.0765 );
		//cv::Point3f point3d = cv::Point3f(1*col, 1*row, 0);
		pos_2d.push_back(point2d);
		pos_3d.push_back(point3d);
	}
}

void testCorrespondingPoints(cv::Mat corner_pos, cv::Mat ids_pos, vector<cv::Point2f> &pos_2d)
{
	pos_2d.clear();
	for(int i=0; i<corner_pos.rows; i++)
	{
		cv::Point2f point2d = cv::Point2f(corner_pos.at<float>(i,0), corner_pos.at<float>(i,1));
		pos_2d.push_back(point2d);
	}
}

void reprojection(cv::Mat camera_matrix, cv::Mat R_wc, cv::Mat t_wc, cv::Point2f p_2d, double z_3d, cv::Point3f & p_3d)
{
	R_wc.convertTo(R_wc, CV_32F);
	t_wc.convertTo(t_wc, CV_32F);
	cout<<t_wc<<endl;
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


int main(int argc, char *argv[])
{
	cv::Mat  camera_matrix;
	cv::Mat dist_coef;
	string filename = "../bin/camera_param.yaml";
	readCameraParams(camera_matrix, dist_coef);
	cv::Mat frame;
	vector<vector<cv::Point2f>> corners;
	vector<int> ids;
	cv::Mat ids_pos;
	cv::Mat corner_pos;
	vector<cv::Point2f> pos_2d;
	vector<cv::Point3f> pos_3d;
	cv::Mat R_wc_vec,t_wc,R_wc;
	cv::Mat T_wc(4,4,CV_32F);
	cv::Mat T_wc_34(3,4,CV_32F);

	string s1 = "/home/lcl/Robocup3DVision/calibration/src0/cap_image/";
	string s2 = ".jpg";
    for(int i=1; i<=5; i++)
    {
		char s[2];
        sprintf(s, "%d", i);
        string path = s1 + s + s2;
        frame = cv::imread(path, 1);
		cv::imshow("image", frame);
		char key = cv::waitKey(10000);
		if(key=='c')
		{
			cv::Mat imageCopy = frame;
			detectcorners(frame, aruco_board_type, corners, ids, corner_pos, ids_pos);
			cv::aruco::drawDetectedCornersCharuco(imageCopy, corner_pos, ids_pos);
			if(ids_pos.total()<4)
			{
				cout<<"Too less corners found, please try again"<<endl;
			}
			else
			{
				cout<<"Corners found"<<endl;
				getCorrespondingPoints(corner_pos, ids_pos, pos_2d, pos_3d);
				cv::solvePnP(pos_3d, pos_2d, camera_matrix, dist_coef, R_wc_vec, t_wc);


				
				cv::aruco::drawAxis(imageCopy, camera_matrix, dist_coef, R_wc_vec, t_wc, 1.0);

				R_wc_vec.convertTo(R_wc_vec, CV_32F);
				t_wc.convertTo(t_wc, CV_32F);
				// cout<<t_wc<<endl;
				cv::Mat t_wc_moved = t_wc.clone();
				t_wc_moved.at<float>(0) = 0;
				//t_wc_moved.at<float>(1) = 0;
				t_wc_moved.at<float>(2) = 0;
				t_wc_moved.convertTo(t_wc_moved, CV_32F);
				// cout<<t_wc_moved<<endl;
				cv::imshow("axis", imageCopy);
				cv::waitKey(0);	
				cv::Rodrigues(R_wc_vec, R_wc);
				for(int i=0; i<3; i++)
				{
					T_wc.at<float>(i,0) = R_wc.at<float>(i,0);
					T_wc.at<float>(i,1) = R_wc.at<float>(i,1);
					T_wc.at<float>(i,2) = R_wc.at<float>(i,2);
					T_wc.at<float>(i,3) = t_wc.at<float>(i,0);
					
					T_wc_34.at<float>(i,0) = R_wc.at<float>(i,0);
					T_wc_34.at<float>(i,1) = R_wc.at<float>(i,1);
					T_wc_34.at<float>(i,2) = R_wc.at<float>(i,2);
					T_wc_34.at<float>(i,3) = t_wc.at<float>(i,0);
				}
				cv::Mat T_wc_moved = T_wc.clone();
				cv::Mat T_wc_moved_34 = T_wc_34.clone();

				for(int i=0; i<3; i++)
				{
					T_wc_moved.at<float>(i,0) = R_wc.at<float>(i,0);
					T_wc_moved.at<float>(i,1) = R_wc.at<float>(i,1);
					T_wc_moved.at<float>(i,2) = R_wc.at<float>(i,2);
					T_wc_moved.at<float>(i,3) = t_wc_moved.at<float>(i,0);
					
					T_wc_moved_34.at<float>(i,0) = R_wc.at<float>(i,0);
					T_wc_moved_34.at<float>(i,1) = R_wc.at<float>(i,1);
					T_wc_moved_34.at<float>(i,2) = R_wc.at<float>(i,2);
					T_wc_moved_34.at<float>(i,3) = t_wc_moved.at<float>(i,0);
				}


				for(int j=0; j<3; j++)
				{
					int i=j*3;
					cv::Mat P(4,1,CV_32F);
					P.at<float>(0) = pos_3d[i].x;
					P.at<float>(1) = pos_3d[i].y;
					P.at<float>(2) = pos_3d[i].z;
					P.at<float>(3) = 1;
					cv::Mat  Pc(3,1,CV_32F) ;
					cv::Mat  Pu(3,1,CV_32F);					// cout<<"pw_moved "<<i<<" : "<<pw_moved<<endl;
					camera_matrix.convertTo(camera_matrix,CV_32F);
					Pc = T_wc_34*P;
					Pu = camera_matrix*Pc;
					cout<<"Reprojection test "<<j<<" : "<<endl;
					cout<<"Pc : "<<Pu.at<float>(0)/Pc.at<float>(2)<<" "<<Pu.at<float>(1)/Pc.at<float>(2)<<endl;
					cout<<"pc : "<<pos_2d[i].x<<" "<<pos_2d[i].y<<endl;
					cout<<Pu.at<float>(0)/Pc.at<float>(2)/pos_2d[i].x<<" "<<Pu.at<float>(1)/Pc.at<float>(2)/pos_2d[i].y<<endl;
					cout<<endl;
					cv::Point3f pw;
					cv::Point3f pw_moved;
					reprojection(camera_matrix, R_wc, t_wc, pos_2d[i], pos_3d[i].z, pw);
					reprojection(camera_matrix, R_wc, t_wc_moved, pos_2d[i],pos_3d[i].z, pw_moved);
					cout<<"pw "<<i<<" : "<<pw<<endl;
					cout<<"pw_moved "<<i<<" : "<<pw_moved<<endl;
					cout<<"Pw "<<i<<" : "<<pos_3d[i]<<endl;
				}
				char save = cin.get();
				if(save == 's')
					// saveCameraParams(R_wc_vec , t_wc_moved, T_wc_moved, T_wc_moved_34); 
					saveCameraParams(R_wc_vec , t_wc, T_wc, T_wc_34); 
			}
		}
		else if(key == 'q')
			break;
	}

	return 0;
}
