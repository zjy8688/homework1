//#include "stdafx.h"
#include "SubImageMatch.h"
#include "opencv2/opencv.hpp"
using namespace cv;
#include <iostream>
using namespace std;
#include <time.h>
#include <math.h>
#include<stdio.h>

//#define IMG_SHOW 0
#define SUB_IMAGE_MATCH_OK 1
#define SUB_IMAGE_MATCH_FAIL -1

int ustc_ConvertBgr2Gray(Mat bgrImg, Mat& grayImg)
{
	if (NULL == bgrImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = bgrImg.cols;
	int height = bgrImg.rows;
	for (int row_i = 0; row_i<height; row_i += 1)
	{
		for (int col_j = 0; col_j<width; col_j += 2)
		{
			int location = 3 * (row_i * width + col_j);
			int b1 = bgrImg.data[location + 0];
			int g1 = bgrImg.data[location + 1];
			int r1 = bgrImg.data[location + 2];
			int b2 = bgrImg.data[location + 3];
			int g2 = bgrImg.data[location + 4];
			int r2 = bgrImg.data[location + 5];

			int grayVal1 = (b1 * 19595 + g1 * 38469 + r1 * 7472) >> 16;
			int grayVal2 = (b2 * 19595 + g2 * 38469 + r2 * 7472) >> 16;
			grayImg.data[location / 3] = grayVal1;
			grayImg.data[location / 3 + 1] = grayVal2;
		}
	}

#ifdef IMG_SHOW
	namedWindow("grayImg", 0);
	imshow("grayImg", grayImg);
	waitKey(0);
#endif
	return SUB_IMAGE_MATCH_OK;
}

int ustc_CalcGrad(Mat grayImg, Mat &gradImg_x, Mat &gradImg_y)
{
	if (NULL == grayImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;


	//计算x方向梯度图
	for (int row_i = 1; row_i < height - 1; row_i += 1)
	{
		for (int col_j = 1; col_j < width - 1; col_j += 1)
		{
			int grad_x =
				grayImg.data[(row_i - 1) * width + col_j + 1]
				+ 2 * grayImg.data[(row_i)* width + col_j + 1]
				+ grayImg.data[(row_i + 1)* width + col_j + 1]
				- grayImg.data[(row_i - 1) * width + col_j - 1]
				- 2 * grayImg.data[(row_i)* width + col_j - 1]
				- grayImg.data[(row_i + 1)* width + col_j - 1];

			((float*)gradImg_x.data)[row_i * width + col_j] = grad_x;

		}
	}

	//计算y方向梯度图
	for (int row_i = 1; row_i < height - 1; row_i += 1)
	{
		for (int col_j = 1; col_j < width - 1; col_j += 1)
		{
			int grad_y =
				-grayImg.data[(row_i - 1) * width + col_j - 1]
				- 2 * grayImg.data[(row_i - 1)* width + col_j]
				- grayImg.data[(row_i - 1)* width + col_j + 1]
				+ grayImg.data[(row_i + 1) * width + col_j - 1]
				+ 2 * grayImg.data[(row_i + 1)* width + col_j]
				+ grayImg.data[(row_i + 1)* width + col_j + 1];

			((float*)gradImg_y.data)[row_i * width + col_j] = grad_y;
		}
	}

#ifdef IMG_SHOW
	Mat gradImg_x_8U(height, width, CV_8UC1);
	gradImg_x_8U.setTo(0);


	//为了方便观察，直接取绝对值
	for (int row_i = 0; row_i < height; row_i++)
	{
		for (int col_j = 0; col_j < width; col_j += 1)
		{
			float a = ((float*)gradImg_x.data)[row_i * width + col_j];
			float b = ((float*)gradImg_y.data)[row_i * width + col_j];
			float val = a*a + b*b;
			int t = *(int*)&val;
			t -= 0x3f800000; t >>= 1;
			t += 0x3f800000;

			gradImg_x_8U.data[row_i * width + col_j] = t;
		}
	}

	namedWindow("aaa", 0);
	imshow("aaa", gradImg_x_8U);
	waitKey(0);
#endif
	return SUB_IMAGE_MATCH_OK;
}

int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y, Mat& angleImg, Mat& magImg)
{
	if (NULL == gradImg_x.data || NULL == gradImg_y.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = gradImg_x.cols;
	int height = gradImg_x.rows;


	//计算角度图

	for (int row_i = 1; row_i < height - 1; row_i++)
	{
		for (int col_j = 1; col_j < width - 1; col_j += 1)
		{
			float grad_x = ((float*)gradImg_x.data)[row_i * width + col_j];
			float grad_y = ((float*)gradImg_y.data)[row_i * width + col_j];
			float angle = atan2(grad_y, grad_x);

			//自己找办法优化三角函数速度，并且转化为角度制，规范化到0-360
			((float*)angleImg.data)[row_i * width + col_j] = angle;
		}
	}

	for (int row_i = 0; row_i < height; row_i++)
	{
		for (int col_j = 0; col_j < width; col_j += 1)
		{
			float a = ((float*)gradImg_x.data)[row_i * width + col_j];
			float b = ((float*)gradImg_y.data)[row_i * width + col_j];
			float val = a*a + b*b;
			int t = *(int*)&val;
			t -= 0x3f800000; t >>= 1;
			t += 0x3f800000;

			magImg.data[row_i * width + col_j] = t;
		}
	}
#ifdef IMG_SHOW
	Mat angleImg_8U(height, width, CV_8UC1);
	//为了方便观察，进行些许变化
	for (int row_i = 0; row_i < height; row_i++)
	{
		for (int col_j = 0; col_j < width; col_j += 1)
		{
			float angle = ((float*)angleImg.data)[row_i * width + col_j];
			angle *= 180 / CV_PI;
			angle += 180;
			//为了能在8U上显示，缩小到0-180之间
			angle /= 2.5;
			angleImg_8U.data[row_i * width + col_j] = angle;
		}




	}

	namedWindow("gradImg_x_8U", 0);
	imshow("gradImg_x_8U", angleImg_8U);
	waitKey();
#endif
}

int ustc_Threshold(Mat grayImg, Mat& binaryImg, int th)
{
	if (NULL == grayImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;

	for (int row_i = 0; row_i < height; row_i++)
	{
		int temp0 = row_i * width;
		for (int col_j = 0; col_j < width; col_j += 2)
		{
			//int pixVal = grayImg.at<uchar>(row_i, col_j);
			int temp1 = temp0 + col_j;
			int pixVal = grayImg.data[temp1];
			int dstVal = (pixVal > th) * 255;

			//binaryImg.at<uchar>(row_i, col_j) = dstVal;
			binaryImg.data[temp1] = dstVal;
		}
	}

#ifdef IMG_SHOW
	namedWindow("binaryImg", 0);
	imshow("binaryImg", binaryImg);
	waitKey();
#endif

	return SUB_IMAGE_MATCH_OK;
}
int ustc_CalcHist(Mat grayImg, int* hist, int hist_len)

{

	if (NULL == grayImg.data)

	{

		cout << "image is NULL." << endl;

		return SUB_IMAGE_MATCH_FAIL;

	}

	int width = grayImg.cols;

	int height = grayImg.rows;

	uchar*p = grayImg.data;

	for (int i = 0; i < hist_len; i++)

	{

		hist[i] = 0;

	}

	for (int row_i = height; row_i >0; row_i--)

	{

		for (int col_j = width; col_j >0; col_j--)

		{

			int pixVal = *p;

			hist[pixVal]++;

			p++;

		}

	}

	return SUB_IMAGE_MATCH_OK;

}

int ustc_SubImgMatch_gray(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (grayImg.channels() != subImg.channels())
	{
		cout << "images have different channels" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	if (width < sub_width || height < sub_height)
	{
		cout << "the subimage is larger ,error input" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}


	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg(height, width, CV_32FC1);
	//匹配误差初始化
	searchImg.setTo(FLT_MAX);

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i <= height - sub_height; i++)
	{
		for (int j = 0; j <= width - sub_width; j++)
		{
			int total_diff = 0;
			//遍历模板图上的每一个像素
			for (int y = 0; y < sub_height; y++)
			{
				for (int x = 0; x < sub_width; x++)
				{
					//大图上的像素位置
					int row_index = i + y;
					int col_index = j + x;
					int bigImg_pix = grayImg.data[row_index * width + col_index];
					//模板图上的像素
					int template_pix = subImg.data[y * sub_width + x];

					total_diff += abs(bigImg_pix - template_pix);
				}
			}
			//存储当前像素位置的匹配误差
			((float*)searchImg.data)[i * width + j] = total_diff;
		}
	}


	int min = searchImg.data[0];
	int min_height = 0, min_width = 0;
	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			if (((float*)searchImg.data)[i * width + j] <= min)
			{
				min = ((float*)searchImg.data)[i * width + j];
				//cout << max;
				min_height = i;
				min_width = j;
			}
		}
	}
	//cout << max_height;
	//cout << max_width;

	Mat Sub_Img(sub_height, sub_width, CV_8UC1);
	for (int i = 0; i<sub_height; i++)
		for (int j = 0; j < sub_width; j++)
		{
			Sub_Img.data[i*sub_width + j] = grayImg.data[(i + min_height)*width + min_width + j];
		}
#ifdef IMG_SHOW
	namedWindow("sub_Img", 0);
	imshow("sub_Img", Sub_Img);
	waitKey(1);
#endif
	return 0;
}
int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int* x, int* y)
{
	if (NULL == colorImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (colorImg.channels() != subImg.channels())
	{
		cout << "images have different channels" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = colorImg.cols;
	int height = colorImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	if (width < sub_width || height < sub_height)
	{
		cout << "the subimage is larger ,error input" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}


	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg(height, width, CV_32FC1);
	//匹配误差初始化
	searchImg.setTo(FLT_MAX);

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i <= height - sub_height; i++)
	{
		for (int j = 0; j <= width - sub_width; j++)
		{
			int total_diffb = 0;
			int total_diffg = 0;
			int total_diffr = 0;
			int total_diff = 0;
			//遍历模板图上的每一个像素
			for (int y = 0; y < sub_height; y++)
			{
				for (int x = 0; x < sub_width; x++)
				{
					//大图上的像素位置
					int row_index = i + y;
					int col_index = j + x;
					int  bigImg_b = colorImg.data[3 * (row_index * width + col_index) + 0];
					int  bigImg_g = colorImg.data[3 * (row_index * width + col_index) + 1];
					int  bigImg_r = colorImg.data[3 * (row_index * width + col_index) + 2];
					//模板图上的像素
					int template_b = subImg.data[3 * (y * sub_width + x) + 0];
					int template_g = subImg.data[3 * (y * sub_width + x) + 1];
					int template_r = subImg.data[3 * (y * sub_width + x) + 2];

					total_diffb += abs(bigImg_b - template_b);
					total_diffg += abs(bigImg_b - template_b);
					total_diffr += abs(bigImg_b - template_b);
					total_diff += (total_diffr + total_diffg + total_diffb);

				}
			}
			//存储当前像素位置的匹配误差
			((float*)searchImg.data)[(i * width + j)] = total_diff;
		}
	}

	float min = ((float*)searchImg.data)[0];
	int min_height = 0, min_width = 0;
	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			if (((float*)searchImg.data)[(i * width + j)] < min)
			{
				min = ((float*)searchImg.data)[i * width + j];
				//cout << max;
				min_height = i;
				min_width = j;
			}
		}
	}

	//cout << max_height;
	//cout << max_width;

	Mat Sub_Img(sub_height, sub_width, CV_8UC3);
	for (int i = 0; i<sub_height; i++)
		for (int j = 0; j < sub_width; j++)
		{
			Sub_Img.data[3 * (i*sub_width + j) + 0] = colorImg.data[3 * ((i + min_height)*width + j + min_width) + 0];
			Sub_Img.data[3 * (i*sub_width + j) + 1] = colorImg.data[3 * ((i + min_height)*width + j + min_width) + 1];
			Sub_Img.data[3 * (i*sub_width + j) + 2] = colorImg.data[3 * ((i + min_height)*width + j + min_width) + 2];
		}
#ifdef IMG_SHOW
	namedWindow("sub_Img", 0);
	imshow("sub_Img", Sub_Img);
	waitKey(1);
#endif
	return 0;
}
int ustc_SubImgMatch_corr(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_OK;
	}

	if (grayImg.channels() != subImg.channels())
	{
		cout << "images have different channels" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	if (width < sub_width || height < sub_height)
	{
		cout << "the subimage is larger ,error input" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}


	float maximum = 0;
	for (int row_big = 0; row_big <= height - sub_height; row_big++)
	{
		for (int col_big = 0; col_big <= width - sub_width; col_big++)
		{

			int sum1 = 0, sum2 = 0, sum3 = 0;
			for (int row_small = 0; row_small < sub_height; row_small++)
			{

				int tempbig1 = (row_big + row_small)*width;
				int tempsma1 = row_small*sub_width;

				for (int col_small = 0; col_small < sub_width; col_small++)
				{

					int tempbig2 = tempbig1 + col_big + col_small;
					int tempsma2 = tempsma1 + col_small;
					sum1 += grayImg.data[tempbig2] * subImg.data[tempsma2]
						sum2 += grayImg.data[tempbig2] * grayImg.data[tempbig2];
					sum3 += subImg.data[tempsma2] * subImg.data[tempsma2];
				}
			}
			float Relation = sum1 / (sqrt(sum2)*sqrt(sum3));
			if (maximum < Relation)
			{
				maximum = Relation;
				*x = row_big;
				*y = col_big;
			}
		}
	}

	return SUB_IMAGE_MATCH_OK;

}
int ustc_SubImgMatch_angle(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (grayImg.channels() != subImg.channels())
	{
		cout << "images have different channels" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	if (width < sub_width || height < sub_height)
	{
		cout << "the subimage is larger ,error input" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}


	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg(height, width, CV_32FC1);
	//匹配误差初始化
	searchImg.setTo(FLT_MAX);

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i <= height - sub_height - 1; i++)
	{
		for (int j = 0; j <= width - sub_width - 1; j++)
		{
			int total_diff = 0;
			//遍历模板图上的每一个像素
			for (int y = 1; y < sub_height - 1; y++)
			{
				for (int x = 1; x < sub_width - 1; x++)
				{
					//大图上的像素位置
					int row_index = i + y;
					int col_index = j + x;

					//计算大图上的像素x梯度
					int bigImg_grad_x =
						grayImg.data[(row_index - 1) * width + col_index + 1]
						+ 2 * grayImg.data[(row_index)* width + col_index + 1]
						+ grayImg.data[(row_index + 1)* width + col_index + 1]
						- grayImg.data[(row_index - 1) * width + col_index - 1]
						- 2 * grayImg.data[(row_index)* width + col_index - 1]
						- grayImg.data[(row_index + 1)* width + col_index - 1];
					//计算大图上的像素y梯度
					int bigImg_grad_y =
						-grayImg.data[(row_index - 1) * width + col_index - 1]
						- 2 * grayImg.data[(row_index - 1)* width + col_index]
						- grayImg.data[(row_index - 1)* width + col_index + 1]
						+ grayImg.data[(row_index + 1) * width + col_index - 1]
						+ 2 * grayImg.data[(row_index + 1)* width + col_index]
						+ grayImg.data[(row_index + 1)* width + col_index + 1];

					float bigImg_angle = atan2(bigImg_grad_y, bigImg_grad_x);

					//计算模板图上的像素x梯度
					int template_grad_x =
						subImg.data[(y - 1) * width + x + 1]
						+ 2 * subImg.data[(y)* width + x + 1]
						+ subImg.data[(y + 1)* width + x + 1]
						- subImg.data[(y - 1) * width + x - 1]
						- 2 * subImg.data[(y)* width + x - 1]
						- subImg.data[(y + 1)* width + x - 1];
					//计算模板图上像素y梯度
					int template_grad_y =
						-subImg.data[(y - 1) * sub_width + x - 1]
						- 2 * subImg.data[(y - 1)* sub_width + x]
						- subImg.data[(y - 1)* sub_width + x + 1]
						+ subImg.data[(y + 1) * sub_width + x - 1]
						+ 2 * subImg.data[(y + 1)* sub_width + x]
						+ subImg.data[(y + 1)* sub_width + x + 1];

					float template_angle = atan2(template_grad_y, template_grad_x);
					total_diff += abs(bigImg_angle - template_angle);
				}
			}
			((float*)searchImg.data)[i * width + j] = total_diff;
		}
	}

	float min = ((float*)searchImg.data)[0];
	int min_height = 0, min_width = 0;
	for (int i = 0; i <= height - sub_height; i++)
	{
		for (int j = 0; j <= width - sub_width; j++)
		{
			if (((float*)searchImg.data)[i * width + j]<min)
			{
				min = ((float*)searchImg.data)[i * width + j];
				//cout << max;
				min_height = i;
				min_width = j;
			}
		}
	}
	//cout << max_height;
	//cout << max_width;

	Mat Sub_Img(sub_height, sub_width, CV_8UC1);
	for (int i = 0; i<sub_height; i++)
		for (int j = 0; j < sub_width; j++)
		{
			Sub_Img.data[i*sub_width + j] = grayImg.data[(i + min_height)*width + j + min_width];
		}
#ifdef IMG_SHOW
	namedWindow("sub_Img", 0);
	imshow("sub_Img", Sub_Img);
	waitKey(1);
#endif
	return 0;
}
int ustc_SubImgMatch_mag(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (grayImg.channels() != subImg.channels())
	{
		cout << "images have different channels" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	if (width < sub_width || height < sub_height)
	{
		cout << "the subimage is larger ,error input" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}


	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg(height, width, CV_32FC1);
	//匹配误差初始化
	searchImg.setTo(FLT_MAX);

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i <= height - sub_height - 1; i++)
	{
		for (int j = 0; j <= width - sub_width - 1; j++)
		{
			int total_diff = 0;
			//遍历模板图上的每一个像素
			for (int y = 1; y < sub_height - 1; y++)
			{
				for (int x = 1; x < sub_width - 1; x++)
				{
					//大图上的像素位置
					int row_index = i + y;
					int col_index = j + x;

					//计算大图上的像素x梯度
					int bigImg_grad_x =
						grayImg.data[(row_index - 1) * width + col_index + 1]
						+ 2 * grayImg.data[(row_index)* width + col_index + 1]
						+ grayImg.data[(row_index + 1)* width + col_index + 1]
						- grayImg.data[(row_index - 1) * width + col_index - 1]
						- 2 * grayImg.data[(row_index)* width + col_index - 1]
						- grayImg.data[(row_index + 1)* width + col_index - 1];
					//计算大图上的像素y梯度
					int bigImg_grad_y =
						-grayImg.data[(row_index - 1) * width + col_index - 1]
						- 2 * grayImg.data[(row_index - 1)* width + col_index]
						- grayImg.data[(row_index - 1)* width + col_index + 1]
						+ grayImg.data[(row_index + 1) * width + col_index - 1]
						+ 2 * grayImg.data[(row_index + 1)* width + col_index]
						+ grayImg.data[(row_index + 1)* width + col_index + 1];


					float bigImg_mag = sqrt(bigImg_grad_y*bigImg_grad_y + bigImg_grad_x*bigImg_grad_x);

					//计算模板图上的像素x梯度
					int template_grad_x =
						subImg.data[(y - 1) * width + x + 1]
						+ 2 * subImg.data[(y)* width + x + 1]
						+ subImg.data[(y + 1)* width + x + 1]
						- subImg.data[(y - 1) * width + x - 1]
						- 2 * subImg.data[(y)* width + x - 1]
						- subImg.data[(y + 1)* width + x - 1];
					//计算模板图上像素y梯度
					int template_grad_y =
						-subImg.data[(y - 1) * sub_width + x - 1]
						- 2 * subImg.data[(y - 1)* sub_width + x]
						- subImg.data[(y - 1)* sub_width + x + 1]
						+ subImg.data[(y + 1) * sub_width + x - 1]
						+ 2 * subImg.data[(y + 1)* sub_width + x]
						+ subImg.data[(y + 1)* sub_width + x + 1];

					float template_mag = sqrt(template_grad_y*template_grad_y + template_grad_x*template_grad_x);
					total_diff += abs(bigImg_mag - template_mag);
				}
			}
			((float*)searchImg.data)[i * width + j] = total_diff;
		}
	}

	float min = ((float*)searchImg.data)[0];
	int min_height = 0, min_width = 0;
	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			if (((float*)searchImg.data)[i * width + j]<min)
			{
				min = ((float*)searchImg.data)[i * width + j];
				//cout << max;
				min_height = i;
				min_width = j;
			}
		}
	}
	//cout << max_height;
	//cout << max_width;

	Mat Sub_Img(sub_height, sub_width, CV_8UC1);
	for (int i = 0; i<sub_height; i++)
		for (int j = 0; j < sub_width; j++)
		{
			Sub_Img.data[i*sub_width + j] = grayImg.data[(i + min_height)*width + j + min_width];
		}
#ifdef IMG_SHOW
	namedWindow("sub_Img", 0);
	imshow("sub_Img", Sub_Img);
	waitKey(1);
#endif
	return 0;
}
int ustc_SubImgMatch_hist(Mat grayImg, Mat subImg, int* x, int* y)

{

	if (NULL == grayImg.data || NULL == subImg.data)

	{

		cout << "image is NULL." << endl;

		return SUB_IMAGE_MATCH_FAIL;

	}
	if (grayImg.channels() != subImg.channels())
	{
		cout << "images have different channels" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}



	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;

	int sub_height = subImg.rows;

	int subhist[256] = { 0 };

	ustc_CalcHist(subImg, subhist, 256);

	int bigImg_pix = 0, total_diff = 0;

	int hist[256] = { 0 };

	int bighist_[256] = { 0 };

	int sub_i, sub_j, i, j;

	int loc_y = 0, loc_x = 0;

	int com_ele;

	for (sub_i = 0; sub_i < sub_height; sub_i++)

	{

		for (sub_j = 0; sub_j < sub_width; sub_j++)

		{

			bigImg_pix = grayImg.data[sub_i * width + sub_j];

			hist[bigImg_pix]++;

		}

	}

	for (int i_ = 0; i_ < 256; i_++)

	{

		bighist_[i_] = hist[i_];

		int temp = hist[i_] - subhist[i_];

		if (temp < 0) temp = -temp;

		total_diff += temp;

	}

	com_ele = total_diff;

	for (i = 0; i <= (height - sub_height); i++)

	{

		if (i != 0)

		{

			total_diff = 0;

			for (sub_i = 0; sub_i < sub_width; sub_i++)

			{

				bigImg_pix = grayImg.data[(i - 1) * width + sub_i];

				hist[bigImg_pix]--;

				bigImg_pix = grayImg.data[(i + sub_height - 1) * width + sub_i];

				hist[bigImg_pix]++;

			}

			for (int _i = 0; _i < 256; _i++)

			{

				bighist_[_i] = hist[_i];

				int temp = hist[_i] - subhist[_i];

				if (temp < 0) temp = -temp;

				total_diff += temp;

			}

			if (com_ele > total_diff)

			{

				com_ele = total_diff;

				loc_y = i;

				loc_x = j;

			}

		}

		for (j = 1; j <= (width - sub_width); j++)

		{

			total_diff = 0;

			for (sub_j = 0; sub_j < sub_height; sub_j++)

			{

				bigImg_pix = grayImg.data[(i + sub_j) * width + j - 1];

				bighist_[bigImg_pix]--;

				bigImg_pix = grayImg.data[(i + sub_j) * width + j + sub_width - 1];

				bighist_[bigImg_pix]++;

			}

			for (int ii = 0; ii < 256; ii++)

			{

				int temp = bighist_[ii] - subhist[ii];

				if (temp < 0) temp = -temp;

				total_diff += temp;

			}

			if (com_ele > total_diff)

			{

				com_ele = total_diff;

				loc_y = i;

				loc_x = j;

			}

		}

	}

	*x = loc_x;

	*y = loc_y;

	return SUB_IMAGE_MATCH_OK;

}



