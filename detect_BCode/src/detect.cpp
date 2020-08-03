#include "detect.h"

using namespace std;
using namespace cv;

Detector::Detector(int width, int number, int paramter): 
mark_width(width),mark_number(number),thres_paramter(paramter)
{
	outer_result.resize(4);
	inner_result.resize(4);
}

/**
 * step 1 : 提取轮廓，多边形拟合
 * step 2.a : 选取轮廓中的四边形凸包
 * step 2.b: 满足a条件的轮廓中，检测是否含有多边形凸包的子轮廓
 * step 2.c: 若轮廓大致满足对边相等，加入candidates
 * step 3 :  对轮廓做透视变换，大致判断其内部结构
 * candidates为mark外圈轮廓，candidates2为内圈，内圈并没有用到
 * */

void Detector::DetectCorners(cv::Mat image)
{
	//step 1
	vector<vector<cv::Point>> contours;
	vector<cv::Vec4i> hierarchy;
	GetIniContours(image, contours, hierarchy);

	vector<vector<cv::Point>> candidates;
	vector<vector<cv::Point>> candidates2;
	

	for(int i=0; i<contours.size(); i++)
	{
		//step 2.a
		if(contours[i].size() == 4 && cv::isContourConvex(contours[i]))
		{
			cv::Mat imagex  = image.clone();
			cv::drawContours(imagex, contours, i,cv::Scalar(255,0,0),2);
			
			int child_contours = hierarchy[i][2];
			int grandson = hierarchy[child_contours][2];
			if(child_contours!=-1)
			{
				//step 2.b
				if(contours[child_contours].size() >3 && grandson == -1 && isContourConvex(contours[child_contours]) && cv::arcLength(contours[child_contours],1)>15)
				{
					cv::drawContours(imagex, contours, child_contours,cv::Scalar(0,255,0),1.5);
					vector<cv::Point2f> sorted_candidates(4);
		 			ClockwiseSort(sorted_candidates, contours[i]);
					double length[4];
					double k[4];
					double k1=0, k2=0;
					for(int j=0; j<4; j++)
					{
						cv::Point2f p1 = sorted_candidates[(j+1)%4];
						cv::Point2f p2 = sorted_candidates[j];
						k[j] = fabs((p1.y-p2.y)/(p1.x-p2.x));
						length[j] = sqrt((p1.y-p2.y)*(p1.y-p2.y)+(p1.x-p2.x)*(p1.x-p2.x));
					}
					//step 2.c
					if((length[0]/length[2]<1.2&& length[0]/length[2]>0.8) && (length[1]/length[3]<1.2 && length[1]/length[3]>0.8) )
					{
						candidates.push_back(contours[i]);
						candidates2.push_back(contours[child_contours]);
					}
				}
			}
		}
	}
	//step 3
	for(int i=0; i<candidates.size(); i++)
	{
		cv::Mat imagex = image.clone();
		cv::drawContours(imagex, candidates, i,cv::Scalar(255,0,0),2);
		cv::Mat transform_image(300,300,CV_8UC1);
		vector<cv::Point2f> src(4);
		//调整对应点顺序
		ClockwiseSort(src, candidates[i]);
		vector<cv::Point2f> dst(4);
		dst[0]=Point(0,0);
    	dst[3]=Point(0,transform_image.rows-1);
    	dst[1]=Point(transform_image.cols-1,0);
    	dst[2]=Point(transform_image.cols-1,transform_image.rows-1);
		//透视变换
		cv::Mat matrix = cv::getPerspectiveTransform(src,dst);
		cv::warpPerspective(image,transform_image,matrix,transform_image.size());	
		if(TestImageCode(transform_image)){
			ClockwiseSort(this->outer_result, candidates[i]);
			ClockwiseSort(this->inner_result, candidates2[i]);
			cv::drawContours(image, candidates, i,cv::Scalar(255,0,0),1.5);
			cv::drawContours(image, candidates2, i,cv::Scalar(0,255,0),1.5);
			break;
		}
	}
	this->result_image = image;
}

void Detector::GetIniContours(cv::Mat image, vector<vector<cv::Point>> &contours, vector<cv::Vec4i> &hierarchy)
{
	if(this->thres_paramter%2==0)
		this->thres_paramter++;
	cv::Mat blur_image, usm_image;

	cv::Mat image_threshold;
	// cv::cvtColor(image, image_threshold, CV_BGR2GRAY);
	// cv::adaptiveThreshold(image_threshold, image_threshold, 255,CV_ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY_INV, this->thres_paramter,7 );
	// cv::imshow("threshold", image_threshold);
	// cv::imshow("image", image);

	//边缘增强
	cv::GaussianBlur(image, blur_image, cv::Size(0,0), 15);
	cv::addWeighted(image, 4.5, blur_image, -3.5, 0, usm_image);
	// cv::imshow("sharp_image", usm_image);
	cv::cvtColor(usm_image, image_threshold, CV_BGR2GRAY);
	cv::adaptiveThreshold(image_threshold, image_threshold, 255,CV_ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY_INV, this->thres_paramter,7 );
	cv::imshow("sharp_threshold", image_threshold);
	
	// cv::waitKey(0);
	vector<vector<cv::Point>> ori_contours; 
	cv::findContours(image_threshold, ori_contours, hierarchy, CV_RETR_TREE,CV_CHAIN_APPROX_NONE);
	contours.resize(ori_contours.size());

	for(int i=0; i<ori_contours.size(); i++)
		cv::approxPolyDP(ori_contours[i], contours[i], cv::arcLength(ori_contours[i],1) * 0.1, true);
}


void Detector::ClockwiseSort(vector<Point2f> &src, vector<Point>contour)
{
	int sum_x=0,sum_y=0;
	for(int i=0; i<4; i++)
	{
		sum_x += contour[i].x;
		sum_y += contour[i].y;
	}
	float avg_x = sum_x/4;
	float avg_y = sum_y/=4;
	for(int i=0; i<4; i++)
	{
		if(contour[i].x<avg_x && contour[i].y<avg_y)
			src[0] = contour[i];
		else if(contour[i].x>avg_x && contour[i].y<avg_y)
			src[1] = contour[i];
		else if(contour[i].x<avg_x && contour[i].y>avg_y)
			src[3] = contour[i];
		else if(contour[i].x>avg_x && contour[i].y>avg_y)
			src[2] = contour[i];
	}
}

bool Detector::TestImageCode(cv::Mat transform_image)
{
	cv::Mat image = transform_image.clone();
	cv::cvtColor(image, image, CV_BGR2GRAY);
	cv::threshold(image, image, 0, 255, CV_THRESH_OTSU);
	 //cv::imshow("OTSU", image);
	 //waitKey(0);
	int row = image.rows/this->mark_width;
	int col = image.cols/this->mark_width;
	int white_block = 0;
	int black_block = 0;
	for(int i=0 ; i<this->mark_number; i++)
	{
		for(int j=0; j<this->mark_number; j++)
		{
			double sum_white = 1;
			double sum_black = 1;
			int x1 = (i)*row+row/6;
			int x2 = (i+1)*row-row/6;
			int y1 = (j)*col+col/6;
			int y2 = (j+1)*col-col/6;
			cv::Point p1(x1,y1), p2(x1,y2), p3(x2, y1), p4(x2,y2);
			//For Debugging
			// cv::Mat image_t  =  image.clone();
			// cv::line(image_t,p1,p2,Scalar(200,200,0),2);
			// cv::line(image_t,p2,p4,Scalar(200,200,0),2);
			// cv::line(image_t,p4,p3,Scalar(200,200,0),2);
			// cv::line(image_t,p3,p1,Scalar(200,200,0),2);
			// cv::imshow("xx",image_t);
			//  cv::waitKey(0);
			for(int x=x1; x<x2; x++)
				for(int y=y1; y<y2; y++)
				{
					if(image.at<uchar>(x,y)==0)
						sum_black ++;
					else if(image.at<uchar>(x,y) ==255)
						sum_white ++;
				}
			if(sum_white/sum_black>2)
				white_block++;
			else if(sum_black/sum_white>2)
				black_block ++;
		}
	}
	if(white_block>2 && black_block >=10)
		return true;
	else return false;
}

void Detector::SetThresParamter(int m)
{
	this->thres_paramter = m;
}


void Detector::SetThresParamter(double distance)
{
	if(distance<0.3 || distance>100)
		this->thres_paramter = 83;
	else if(distance>=0.3 && distance <0.45)
		this->thres_paramter = 63;
	else if(distance>=0.45 && distance <1.3)
		this->thres_paramter = 43;
	else  this->thres_paramter = 23;
}

vector<cv::Point2f> Detector::GetDetectResult()
{
	return this->outer_result;
}

void Detector::ShowResultImage()
{
	cv::imshow("result", this->result_image);
	cv::waitKey(0);
}


