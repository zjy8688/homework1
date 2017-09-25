//PB14210079周天贶
#include "stdafx.h"
#include <opencv2/opencv.hpp>
using namespace cv;
#include <iostream>
using namespace std;
#include <time.h>
#include "SubImageMatch.h"
#include <math.h>
#define PI 3.1415

int ustc_ConvertBgr2Gray(Mat bgrImg, Mat& grayImg) {
	//彩色图转化成灰度图
	if (NULL == bgrImg.data)
	{
		cout << "invalid image" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int x = bgrImg.cols;
	int y = bgrImg.rows;
	Mat grayIm(y, x, CV_8UC1);
	for (int i = 0; i<y; i++)
		for (int j = 0; j < x; j++)
		{
			int b = bgrImg.data[3 * (i * x + j) + 0];
			int g = bgrImg.data[3 * (i * x + j) + 1];
			int r = bgrImg.data[3 * (i * x + j) + 2];
			int graypixel = (int)b * 0.114f + g * 0.587f + r * 0.229f;
			grayIm.data[i * x + j] = graypixel;
		}
	grayImg = grayIm;

	//cvtColor(bgrImg, grayImg, CV_BGR2GRAY);
	/*
	#ifdef IMG_SHOW
	namedWindow("BGR2GRAY", 0);
	imshow("BGR2GRAY", grayImg);
	waitKey();
    #endif
	*/
    
	return SUB_IMAGE_MATCH_OK;

}

int ustc_CalcGrad(Mat grayImg, Mat& gradImg_x, Mat& gradImg_y) {
	//计算梯度,然后把梯度大小normalize到0，255之间然后显示
	if (NULL == grayImg.data)
	{
		cout << "invalid image" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	//to do: check gray level image

	int x = grayImg.cols;
	int y = grayImg.rows;
	Mat gradImgx(y, x, CV_32FC1);
	Mat gradImgy(y, x, CV_32FC1);
	for (int i = 0; i < y; i++)
	{
		for (int j = 0; j < x; j++)
		{
			int tempx = 0;
			int tempy = 0;
			if ((i - 1) >= 0 && (j - 1) >= 0) {
				tempx += -grayImg.data[(i - 1)*x + (j - 1)];
				tempy += -grayImg.data[(i - 1)*x + (j - 1)];
			}
			if (i >= 0 && (j - 1) >= 0) {
				tempx += -2 * grayImg.data[(i)*x + (j - 1)];
			}
			if ((i + 1) < y && (j - 1) >= 0) {
				tempx += -grayImg.data[(i + 1)*x + (j - 1)];
				tempy += grayImg.data[(i + 1)*x + (j - 1)];
			}
			if ((i + 1) < y) {
				tempy += 2 * grayImg.data[(i + 1)*x + (j)];
			}
			if ((i - 1) >= 0) {
				tempy += -2 * grayImg.data[(i - 1)*x + (j)];
			}
			if ((i - 1) >= 0 && (j + 1) < x) {
				tempx += grayImg.data[(i - 1)*x + (j + 1)];
				tempy += -grayImg.data[(i - 1)*x + (j + 1)];
			}
			if ((i) >= 0 && (j + 1) < x) {
				tempx += 2 * grayImg.data[(i)*x + (j + 1)];
			}
			if ((i + 1) < y && (j + 1) < x) {
				tempx += grayImg.data[(i + 1)*x + (j + 1)];
				tempy += grayImg.data[(i + 1)*x + (j + 1)];
			}
			/*
			tempx = abs(tempx);
			tempy = abs(tempy);
			if (tempx < 0)tempx = 0;
			if (tempx > 255)tempx = 255;
			if (tempy < 0)tempy = 0;
			if (tempy > 255)tempy = 255;
			*/
			
			
			gradImgx.at<float>(i,j) = (float)tempx;
			gradImgy.at<float>(i,j) = (float)tempy;
		}
	}
	gradImg_x = gradImgx;
	gradImg_y = gradImgy;
	Mat show_x ;
	Mat show_y ;
	cv::normalize(gradImg_x,show_x, 0,1, cv::NORM_MINMAX);
	cv::normalize(gradImg_y,show_y, 0,1, cv::NORM_MINMAX);

	/*
	#ifdef IMG_SHOW
	namedWindow("grad_x", 0);
	imshow("grad_x",show_x);
	waitKey();
	namedWindow("grad_y", 0);
	imshow("grad_y", show_y);
	waitKey();
    #endif
	*/
    
	return SUB_IMAGE_MATCH_OK;

}

int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y, Mat& angleImg, Mat& magImg) {
	//计算辐角和模值
	if (NULL == gradImg_x.data || NULL == gradImg_y.data)
	{
		cout << "invalid image" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if ((gradImg_x.cols != gradImg_y.cols) || (gradImg_x.rows != gradImg_y.rows)) {
		cout << "grad_x and grad_y mismatch" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int x = gradImg_x.cols;
	int y = gradImg_x.rows;
	Mat angleImage(y, x, CV_32FC1);//atan角度
	Mat magImage(y, x, CV_32FC1);
	for (int i=0; i<y; i++)
		for (int j = 0; j < x; j++) {
			angleImage.at<float>(i,j) = atan2(gradImg_y.at<float>(i, j), gradImg_x.at<float>(i, j))*180.0/PI;
			if (angleImage.at<float>(i,j) < 0)
				angleImage.at<float>(i,j) += 180;
			magImage.at<float>(i,j) = sqrt(pow(gradImg_y.at<float>(i, j), 2) + pow(gradImg_x.at<float>(i, j), 2));

		}
	angleImg = angleImage;
	magImg = magImage;
	Mat angleshow;
	Mat magshow;
	cv::normalize(angleImg, angleshow, 0, 1, cv::NORM_MINMAX);
	cv::normalize(magImg, magshow, 0, 1, cv::NORM_MINMAX);
	/*
	namedWindow("angleImg", 0);
	imshow("angleImg", angleshow);
	waitKey();
	namedWindow("magImg", 0);
	imshow("magImg", magshow);
	waitKey();
	*/
	

	return SUB_IMAGE_MATCH_OK;
}
int ustc_Threshold(Mat grayImg, Mat& binaryImg, int th) {
	if (NULL == grayImg.data )
	{
		cout << "invalid image_Threshold" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int x = grayImg.cols;
	int y = grayImg.rows;
	Mat binaryImage(y, x, CV_8UC1);
	for(int i =0; i<y; i++)
		for (int j = 0; j < x; j++)
				binaryImage.data[i*x + j] = grayImg.data[i*x + j] >= th ? 255 : 0;

	binaryImg = binaryImage;
	namedWindow("binaryImage", 0);
	imshow("binaryImage", binaryImg);
	waitKey();
	return SUB_IMAGE_MATCH_OK;
}

int ustc_CalcHist(Mat grayImg, int* hist, int hist_len) {
	//将0到255分成hist_len个灰度等级
	if (NULL == grayImg.data)
	{
		cout << "invalid image_Threshold" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int x = grayImg.cols;
	int y = grayImg.rows;
	int temphist[256];
	for (int i = 0; i < hist_len; i++)
		hist[i] = 0;
	for (int i = 0; i < 256; i++)
		temphist[i] = 0;
	for(int i=0; i<y; i ++)
		for (int j = 0; j < x; j++)
				hist[grayImg.data[i*x + j]]++;
	for (int k = 0; k < 256; k++)
	{
		hist[(int)floor((k) / 255 * (hist_len - 1) + 0.5)]++;
	}
	return SUB_IMAGE_MATCH_OK;

}


int ustc_SubImgMatch_gray(Mat grayImg, Mat subImg, int* x, int* y) {
	if (NULL == grayImg.data)
	{
		cout << "invalid grayImg" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (NULL == subImg.data)
	{
		cout << "invalid subImg" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int cols = grayImg.cols;
	int rows = grayImg.rows;
	int sub_cols = subImg.cols;
	int sub_rows = subImg.rows;
	if (sub_cols > cols || sub_rows > rows)
	{
		cout << "subImg is larger than the image to be matched" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int temp_diff = 0;
	for(int i = 0; i<(cols-sub_cols+1);i++)//子图匹配
		for (int j = 0; j < (rows - sub_rows + 1); j++)
		{
			int temp = 0;
			for (int m = 0; m < sub_cols; m++)
				for (int n = 0; n < sub_rows; n++)
				{
					temp += grayImg.data[(j + n)*cols + (i + m)]-subImg.data[(n)*sub_cols+(m)];
				}
			if (i == 0 && j == 0)
			{
				temp_diff = abs(temp);
				*x = 0;
				*y = 0;
			}
			else
				if (temp_diff > abs(temp))
				{
					temp_diff = abs(temp);
					*x = i;
					*y = j;
				}
				
		}

	cout << "x=" << *x << endl;
	cout << "y=" << *y << endl;
	Mat matchImg(sub_rows, sub_cols, CV_8UC1);
	for (int m = 0; m < sub_cols; m++)
		for (int n = 0; n < sub_rows; n++)
			matchImg.data[n*sub_cols + m] = grayImg.data[(*y + n)*cols + (*x + m)];
	namedWindow("subImage", 0);
	imshow("subImage", subImg);
	waitKey();
	namedWindow("matchImage", 0);
	imshow("matchImage", matchImg);
	waitKey();
	return SUB_IMAGE_MATCH_OK;

}

int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int* x, int* y) {
	if (NULL == colorImg.data)
	{
		cout << "invalid grayImg" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (NULL == subImg.data)
	{
		cout << "invalid subImg" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int cols = colorImg.cols;
	int rows = colorImg.rows;
	int sub_cols = subImg.cols;
	int sub_rows = subImg.rows;
	if (sub_cols > cols || sub_rows > rows)
	{
		cout << "subImg is larger than the image to be matched" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int temp_diff = 0;
	for (int i = 0; i<(cols - sub_cols + 1); i++)//子图匹配
		for (int j = 0; j < (rows - sub_rows + 1); j++)
		{
			int temp = 0;
			for (int m = 0; m < sub_cols; m++)
				for (int n = 0; n < sub_rows; n++)
				{
					temp += colorImg.data[3*((j + n)*cols + (i + m))] - subImg.data[3*((n)*sub_cols + (m))];
					temp += colorImg.data[3 * ((j + n)*cols + (i + m))+1] - subImg.data[3 * ((n)*sub_cols + (m))+1];
					temp += colorImg.data[3 * ((j + n)*cols + (i + m))+2] - subImg.data[3 * ((n)*sub_cols + (m))+2];
				}
			if (i == 0 && j == 0)
			{
				temp_diff = abs(temp);
				*x = 0;
				*y = 0;
			}
			else
				if (temp_diff > abs(temp))
				{
					temp_diff = abs(temp);
					*x = i;
					*y = j;
				}

		}

	cout << "x=" << *x << endl;
	cout << "y=" << *y << endl;
	Mat matchImg(sub_rows, sub_cols, CV_8UC3);
	for (int m = 0; m < sub_cols; m++)
		for (int n = 0; n < sub_rows; n++)
		{
			matchImg.data[3 * (n*sub_cols + m) + 0] = colorImg.data[3 * ((*y + n)*cols + (*x + m)) + 0];
			matchImg.data[3 * (n*sub_cols + m) + 1] = colorImg.data[3 * ((*y + n)*cols + (*x + m)) + 1];
			matchImg.data[3 * (n*sub_cols + m) + 2] = colorImg.data[3 * ((*y + n)*cols + (*x + m)) + 2];
		}
			
	namedWindow("subImage", 0);
	imshow("subImage", subImg);
	waitKey();
	namedWindow("matchImage", 0);
	imshow("matchImage", matchImg);
	waitKey();
	return SUB_IMAGE_MATCH_OK;

}

int ustc_SubImgMatch_corr(Mat grayImg, Mat subImg, int* x, int* y)
{
	//利用相关来匹配
	if (NULL == grayImg.data)
	{
		cout << "invalid grayImg" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (NULL == subImg.data)
	{
		cout << "invalid subImg" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int cols = grayImg.cols;
	int rows = grayImg.rows;
	int sub_cols = subImg.cols;
	int sub_rows = subImg.rows;
	if (sub_cols > cols || sub_rows > rows)
	{
		cout << "subImg is larger than the image to be matched" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	/*
	Mat grayImg_sq(rows, cols, CV_32FC1);
	Mat subImg_sq(sub_rows, sub_cols, CV_32FC1);
	
	for(int i=0; i<rows; i++)
		for (int j = 0; j < cols; j++)
		{
			grayImg_sq.at<float>(i,j) = pow(grayImg.data[i*cols + j], 2);

		}
	for (int i = 0; i<sub_rows; i++)
		for (int j = 0; j < sub_cols; j++)
		{
			subImg_sq.at<float>(i, j) = pow(subImg.data[i*sub_cols + j], 2);

		}
		*/
	double cov = 0;
	for (int i = 0;i<(rows-sub_rows+1);i++)
		for (int j = 0; j < (cols - sub_cols + 1); j++)
		{
			int numer = 0;
			int denom1 = 0;
			int denom2 = 0;
			for(int m = 0; m<sub_rows;m++)
				for (int n = 0; n < sub_cols; n++)
				{
					numer += grayImg.data[(i + m)*cols + (j + n)] * subImg.data[m*sub_cols+n];
					denom1 += grayImg.data[(i + m)*cols + (j + n)]* grayImg.data[(i + m)*cols + (j + n)];
					denom2 += subImg.data[m*sub_cols + n]* subImg.data[m*sub_cols + n];
				}
			if (i == 0 && j == 0) {
				cov = (float)numer / sqrt(denom1) / sqrt(denom2);
				*x = j;
				*y = i;
			}
			else
				if (cov < (float)numer / sqrt(denom1) / sqrt(denom2)) {
					cov = (float)numer / sqrt(denom1) / sqrt(denom2);
					//double sqr_denom = sqrt(denom1*denom2);
					//cov = numer / sqr_denom;
					*x = j;
					*y = i;
				}


		}
	cout << "x=" << *x << endl;
	cout << "y=" << *y << endl;
	Mat matchImg(sub_rows, sub_cols, CV_8UC1);
	for (int m = 0; m < sub_cols; m++)
		for (int n = 0; n < sub_rows; n++)
			matchImg.data[n*sub_cols + m] = grayImg.data[(*y + n)*cols + (*x + m)];
	namedWindow("subImage", 0);
	imshow("subImage", subImg);
	waitKey();
	namedWindow("matchImage", 0);
	imshow("matchImage", matchImg);
	waitKey();
	return SUB_IMAGE_MATCH_OK;


}


int ustc_SubImgMatch_angle(Mat grayImg, Mat subImg, int* x, int* y) {
	if (NULL == grayImg.data)
	{
		cout << "invalid grayImg" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (NULL == subImg.data)
	{
		cout << "invalid subImg" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int cols = grayImg.cols;
	int rows = grayImg.rows;
	int sub_cols = subImg.cols;
	int sub_rows = subImg.rows;
	if (sub_cols > cols || sub_rows > rows)
	{
		cout << "subImg is larger than the image to be matched" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	Mat grad_x, grad_y;
	ustc_CalcGrad(grayImg, grad_x, grad_y);
	Mat angleImg, magImg;
	ustc_CalcAngleMag(grad_x, grad_y, angleImg, magImg);
	

	Mat grad_x_sub, grad_y_sub;
	ustc_CalcGrad(subImg, grad_x_sub, grad_y_sub);
	Mat angleImg_sub, magImg_sub;
	ustc_CalcAngleMag(grad_x_sub, grad_y_sub, angleImg_sub, magImg_sub);

	//cv::normalize(angleImg, angleImg, 0, 1, cv::NORM_MINMAX);
	//cv::normalize(angleImg_sub, angleImg_sub, 0, 1, cv::NORM_MINMAX);
	angleImg.convertTo(angleImg, CV_8UC1, 0.5, 0);
	angleImg_sub.convertTo(angleImg_sub, CV_8UC1, 0.5, 0);


	int temp_diff = 0;
	for (int i = 0; i<(cols - sub_cols + 1); i++)//子图匹配
		for (int j = 0; j < (rows - sub_rows + 1); j++)
		{
			int temp = 0;
			for (int m = 0; m < sub_cols; m++)
				for (int n = 0; n < sub_rows; n++)
				{
					temp += abs(angleImg.data[(j+n)*cols+(i+m)] - angleImg_sub.data[n*sub_cols+m]);
				}
			if (i == 0 && j == 0)
			{
				temp_diff = abs(temp);
				*x = 0;
				*y = 0;
			}
			else
				if (temp_diff > abs(temp))
				{
					temp_diff = abs(temp);
					*x = i;
					*y = j;
				}

		}

	cout << "x=" << *x << endl;
	cout << "y=" << *y << endl;
	Mat matchImg(sub_rows, sub_cols, CV_8UC1);
	for (int m = 0; m < sub_cols; m++)
		for (int n = 0; n < sub_rows; n++)
			matchImg.data[n*sub_cols + m] = grayImg.data[(*y + n)*cols + (*x + m)];
	namedWindow("subImage", 0);
	imshow("subImage", subImg);
	waitKey();
	namedWindow("matchImage", 0);
	imshow("matchImage", matchImg);
	waitKey();
	return SUB_IMAGE_MATCH_OK;

}

int ustc_SubImgMatch_mag(Mat grayImg, Mat subImg, int* x, int* y) {
	if (NULL == grayImg.data)
	{
		cout << "invalid grayImg" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (NULL == subImg.data)
	{
		cout << "invalid subImg" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int cols = grayImg.cols;
	int rows = grayImg.rows;
	int sub_cols = subImg.cols;
	int sub_rows = subImg.rows;
	if (sub_cols > cols || sub_rows > rows)
	{
		cout << "subImg is larger than the image to be matched" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	Mat grad_x, grad_y;
	ustc_CalcGrad(grayImg, grad_x, grad_y);
	Mat angleImg, magImg;
	ustc_CalcAngleMag(grad_x, grad_y, angleImg, magImg);


	Mat grad_x_sub, grad_y_sub;
	ustc_CalcGrad(subImg, grad_x_sub, grad_y_sub);
	Mat angleImg_sub, magImg_sub;
	ustc_CalcAngleMag(grad_x_sub, grad_y_sub, angleImg_sub, magImg_sub);

	//cv::normalize(angleImg, angleImg, 0, 1, cv::NORM_MINMAX);
	//cv::normalize(angleImg_sub, angleImg_sub, 0, 1, cv::NORM_MINMAX);
	magImg.convertTo(magImg, CV_8UC1, 0.5, 0);
	magImg_sub.convertTo(magImg_sub, CV_8UC1, 0.5, 0);
	int temp_diff = 0;
	for (int i = 0; i<(cols - sub_cols + 1); i++)//子图匹配
		for (int j = 0; j < (rows - sub_rows + 1); j++)
		{
			int temp = 0;
			for (int m = 0; m < sub_cols; m++)
				for (int n = 0; n < sub_rows; n++)
				{
					temp += abs(magImg.data[(j + n)*cols+ (i + m)] - magImg_sub.data[n*sub_cols+ m]);
				}
			if (i == 0 && j == 0)
			{
				temp_diff = abs(temp);
				*x = 0;
				*y = 0;
			}
			else
				if (temp_diff > abs(temp))
				{
					temp_diff = abs(temp);
					*x = i;
					*y = j;
				}

		}

	cout << "x=" << *x << endl;
	cout << "y=" << *y << endl;
	Mat matchImg(sub_rows, sub_cols, CV_8UC1);
	for (int m = 0; m < sub_cols; m++)
		for (int n = 0; n < sub_rows; n++)
			matchImg.data[n*sub_cols + m] = grayImg.data[(*y + n)*cols + (*x + m)];
	namedWindow("subImage", 0);
	imshow("subImage", subImg);
	waitKey();
	namedWindow("matchImage", 0);
	imshow("matchImage", matchImg);
	waitKey();
	return SUB_IMAGE_MATCH_OK;

}

int ustc_SubImgMatch_hist(Mat grayImg, Mat subImg, int* x, int* y) {
	//以256个灰度等级做匹配
	if (NULL == grayImg.data)
	{
		cout << "invalid grayImg" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (NULL == subImg.data)
	{
		cout << "invalid subImg" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int cols = grayImg.cols;
	int rows = grayImg.rows;
	int sub_cols = subImg.cols;
	int sub_rows = subImg.rows;
	if (sub_cols > cols || sub_rows > rows)
	{
		cout << "subImg is larger than the image to be matched" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	Mat tempImg(rows, cols, CV_8UC1);
	int diff = 0;
	int hist[256];
	int hist_sub[256];
	int hist_len = 256;

	ustc_CalcHist(subImg, hist_sub, hist_len);
	for(int i =0; i<(rows-sub_rows+1);i++)
		for (int j = 0; j < (cols - sub_cols + 1); j++)
		{
			int temp = 0;
			if(j==0)//每行计算第一个
			{
				for (int m = 0; m<sub_rows; m++)
					for (int n = 0; n < sub_cols; n++)
					{
						tempImg.data[m*sub_rows + n] = grayImg.data[(i+m)*rows + j+n];

					}
				ustc_CalcHist(tempImg, hist, hist_len);
			}
			else//后面在第一个基础上增减
			{
				for (int m = 0; m < sub_rows; m++)
				{
					hist[(int)grayImg.data[(i + m)*rows + j - 1]]--;
					hist[(int)grayImg.data[(i + m)*rows + j + sub_cols - 1]]++;
				}
			}
			
			
			for (int k = 0; k < hist_len; k++)
				temp += abs(hist[k] - hist_sub[k]);
			if (i == 0 && j == 0)
			{
				diff = temp;
				*x = j;
				*y = i;
			}
			else
				if (diff > temp)
				{
					diff = temp;
					*x = j;
					*y = i;
				}
		}

	cout << "x=" << *x << endl;
	cout << "y=" << *y << endl;
	Mat matchImg(sub_rows, sub_cols, CV_8UC1);
	for (int m = 0; m < sub_cols; m++)
		for (int n = 0; n < sub_rows; n++)
			matchImg.data[n*sub_cols + m] = grayImg.data[(*y + n)*cols + (*x + m)];
	namedWindow("subImage", 0);
	imshow("subImage", subImg);
	waitKey();
	namedWindow("matchImage", 0);
	imshow("matchImage", matchImg);
	waitKey();
	return SUB_IMAGE_MATCH_OK;
}
