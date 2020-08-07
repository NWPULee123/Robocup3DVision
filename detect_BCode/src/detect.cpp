#include "detect.h"

using namespace std;
using namespace cv;

namespace RectDetect
{

bool compareX(cv::Point a, cv::Point b)
{
	return a.x < b.x;
}

bool compareY(cv::Point a, cv::Point b)
{
	return a.y < b.y;
}

Detector::Detector(int width, int number, int paramter): 
mark_width(width),mark_number(number),thres_paramter(paramter)
{
	this->USM_paramter = 3.0;
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
	vector<vector<cv::Point>> ori_contours, contours;
	vector<cv::Vec4i> hierarchy;
	GetIniContours(image, ori_contours, contours, hierarchy);

	vector<vector<cv::Point>> candidates;
	vector<vector<cv::Point>> candidates2;
	vector<int> candidate_ids;
	cv::Mat ori_contours_image = image.clone();
	cv::Mat caculated_image = image.clone();
	cv::Mat final_image = image.clone();

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
						candidate_ids.push_back(i);
						candidates.push_back(contours[i]);
						candidates2.push_back(contours[child_contours]);
					}
				}
			}
		}
	}
	int target_id = -1;
	
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
			target_id = candidate_ids[i];
			cv::drawContours(ori_contours_image, ori_contours, target_id,cv::Scalar(255,0,0),1.5);
			ClockwiseSort(this->outer_result, candidates[i]);
			ClockwiseSort(this->inner_result, candidates2[i]);
			cv::drawContours(image, candidates, i,cv::Scalar(255,0,0),1.5);
			cv::drawContours(image, candidates2, i,cv::Scalar(0,255,0),1.5);
			break;
		}
	}
	cv::Mat Weights(2,4,CV_64F);
	GetLinerRegressionWeights(ori_contours[target_id], Weights);
	vector<cv::Point2d> caculated_corners(4);
	Getcorners(Weights, caculated_corners);
	cout<<"ori_corners :"<<endl<<this->outer_result<<endl;
	cout<<"caculated_corners : "<<endl<<caculated_corners<<endl;

	if(isWeightsValid(Weights))
	{
		for(int i=0; i<4; i++)
		{
			this->outer_result[i] = 0.5*(this->outer_result[i] + cv::Point2f(caculated_corners[i]));
		}
	}
	cout<<"final_corners :"<<endl<<this->outer_result<<endl;
	vector<vector<cv::Point>> cal_corners(1);
	cal_corners[0].resize(4);
	for(int i=0; i<4; i++)
		cal_corners[0][i] = caculated_corners[i];
	vector<vector<cv::Point>> result_corners(1);
	result_corners[0].resize(4);
	for(int i=0; i<4; i++)
		result_corners[0][i] = this->outer_result[i];
	cv::drawContours(caculated_image, cal_corners, 0, cv::Scalar(255,0,0),1.5);
	cv::drawContours(final_image, result_corners, 0, cv::Scalar(255,0,0),1.5);

	cv::imshow("caculated_contours", caculated_image);
	cv::imshow("final_contours", final_image);
	cv::waitKey(0);
	this->result_image = image;
}



//初步筛选边缘
void Detector::GetIniContours(cv::Mat image, vector<vector<cv::Point>> &ori_contours, vector<vector<cv::Point>> &contours, vector<cv::Vec4i> &hierarchy)
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
	cv::addWeighted(image, this->USM_paramter, blur_image, -(this->USM_paramter-1), 0, usm_image);
	// cv::imshow("sharp_image", usm_image);
	cv::cvtColor(usm_image, image_threshold, CV_BGR2GRAY);
	cv::adaptiveThreshold(image_threshold, image_threshold, 255,CV_ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY_INV, this->thres_paramter,7 );
	cv::imshow("sharp_threshold", image_threshold);
	
	// cv::waitKey(0);
	cv::findContours(image_threshold, ori_contours, hierarchy, CV_RETR_TREE,CV_CHAIN_APPROX_NONE);
	contours.resize(ori_contours.size());

	for(int i=0; i<ori_contours.size(); i++)
		cv::approxPolyDP(ori_contours[i], contours[i], cv::arcLength(ori_contours[i],1) * 0.1, true);
}



//检测内部结构
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



//求直线交点
void Detector::Getcorners(cv::Mat Weights, vector<cv::Point2d> &caculated_corners)
{
	for(int i=0; i<4; i++)
	{
		double k1 = Weights.at<double>(1,i);
		double b1 = Weights.at<double>(0,i);
		double k2 = Weights.at<double>(1,(i+1)%4);
		double b2 = Weights.at<double>(0,(i+1)%4);
		caculated_corners[(i+1)%4].x = (b2-b1)/(k1-k2);
		caculated_corners[(i+1)%4].y = 0.5*(k1*caculated_corners[(i+1)%4].x+b1+k2*caculated_corners[(i+1)%4].x+b2);
	}
}



//线性回归
void Detector::GetLinerRegressionWeights(vector<cv::Point> ori_contours, cv::Mat &Weights)
{
	vector<vector<double>> input_x(4);
	vector<vector<double>> input_y(4);
	CreateFilterData(ori_contours, input_x, input_y);
	UpperclassFilter *filter = new UpperclassFilter();
	for(int i=0; i<4; i++)
	{
		cv::Mat W(2,1,CV_64FC1);
		W = filter->LinerRegression(input_x[i], input_y[i]);
		W.copyTo(Weights.rowRange(0,2).col(i));
		cout<<"Edge "<<i<<" :"<<endl;
		cout<<"caculated W : "<<W<<endl;
		for(int j=0; j<input_x[i].size(); j++)
		{
			double predict_y = W.at<double>(0) + W.at<double>(1)*input_x[i][j];
			// cout<<"x : "<<input_x[i][j]<<"\t predict_y : "<<predict_y<<"\t y :"<<input_y[i][j]<<endl;
		}
	}
}



//获取线性回归的输入数据
void Detector::CreateFilterData(vector<cv::Point> ori_contours, vector<vector<double>> &input_x, vector<vector<double>> &input_y)
{
	int max_diff = 5;
	vector<vector<cv::Point>> edges(4);
	cv::Point top_left_corner = this->outer_result[0];
	cv::Point top_right_corner = this->outer_result[1];
	cv::Point lower_right_corner = this->outer_result[2];
	cv::Point lower_left_corner = this->outer_result[3];
	cv::Point center;
	center.x = (top_left_corner.x + top_right_corner.x + lower_right_corner.x + lower_left_corner.x)/4;
	center.y = (top_left_corner.y + top_right_corner.y + lower_right_corner.y + lower_left_corner.y)/4;
	for(int i=0; i<ori_contours.size(); i++)
	{
		int x = ori_contours[i].x, y = ori_contours[i].y;
		if(x>=top_left_corner.x && x<=top_right_corner.x && (fabs(y-top_left_corner.y)<max_diff || fabs(y-top_right_corner.y)<max_diff))
			edges[0].push_back(ori_contours[i]);
		else if(y>=top_right_corner.y && y<=lower_right_corner.y && (fabs(x-top_right_corner.x)<max_diff || fabs(x-lower_right_corner.x)<max_diff))
			edges[1].push_back(ori_contours[i]);
		else if(x>=lower_left_corner.x && x<=lower_right_corner.x && (fabs(y-lower_left_corner.y)<max_diff || fabs(y-lower_right_corner.y)<max_diff))
			edges[2].push_back(ori_contours[i]);
		else if(y>=top_left_corner.y && y<=lower_left_corner.y && (fabs(x-top_left_corner.x)<max_diff || fabs(x-lower_left_corner.x)<max_diff))
			edges[3].push_back(ori_contours[i]);
	}
	for(int i=0; i<4; i++)
	{
		if(i%2 == 0)
			sort(edges[i].begin(), edges[i].end(), compareX);
		else sort(edges[i].begin(), edges[i].end(), compareY);
		int deprecated_num = edges[i].size()/5;
		// cout<<"num: "<<deprecated_num<<endl;
		for(int j=deprecated_num; j<edges[i].size()-deprecated_num; j++)
		{
			input_x[i].push_back(edges[i][j].x);
			input_y[i].push_back(edges[i][j].y);
			// cout<<edges[i][j]<<endl;
		}
	}
}




void Detector::SetThresParamter(int m)
{
	this->thres_paramter = m;
}




void Detector::SetParamter(double distance)
{
	if(distance<0.3 || distance>100)
	{
		this->thres_paramter = 83;
		this->USM_paramter = 1.2;
	}
	else if(distance>=0.3 && distance <0.45)
	{
		this->thres_paramter = 63;
		this->USM_paramter = 1.5;
	}
	else if(distance>=0.45 && distance <1.3)
	{
		this->thres_paramter = 43;
		this->USM_paramter = 2.0;
	}

	else  
	{
		this->thres_paramter = 23;
		this->USM_paramter = 3.0;
	}
}



bool Detector::isWeightsValid(cv::Mat Weights)
{
	for(int i=0; i<4; i++)
	{
		if(Weights.at<float>(0,i)==0 || Weights.at<float>(1,i)==0)
			return false;
	}
	return true;
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

}