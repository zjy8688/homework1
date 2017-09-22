#include "opencv2/opencv.hpp"
using namespace cv;
#include <iostream>
using namespace std;
#include <time.h>
#include "SubImageMatch.h"

//#include <image.h>

//#define IMG_SHOW
#define MY_OK 1
#define MY_FAIL -1
#define SUB_IMAGE_MATCH_OK 1
#define SUB_IMAGE_MATCH_FAIL -1

int ustc_ConvertBgr2Gray(Mat bgrImg, Mat& grayImg)
{
	int width = bgrImg.cols;
	int height = bgrImg.rows;

	for (int row_i = 0; row_i < height; row_i++)
	{
		int temp0 = row_i * width;
		for (int col_j = 0; col_j < width; col_j++)
		{
			int b = bgrImg.data[3 * (row_i * width + col_j) + 0];
			int g = bgrImg.data[3 * (row_i * width + col_j) + 1];
			int r = bgrImg.data[3 * (row_i * width + col_j) + 2];

			int grayVal = (b * 114 + g * 587 + r * 229) / 1000;
			grayImg.data[row_i * width + col_j] = grayVal;
		}
	}
	return 1;
}
int ustc_CalcGrad(Mat grayImg, Mat& gradImg_x, Mat& gradImg_y)
{
	int width = grayImg.cols;
	int height = grayImg.rows;

	//计算x方向梯度图
	for (int row_i = 1; row_i < height - 1; row_i++)
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
	for (int row_i = 1; row_i < height - 1; row_i++)
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
	//为了方便观察，直接取绝对值
	for (int row_i = 0; row_i < height; row_i++)
	{
		for (int col_j = 0; col_j < width; col_j += 1)
		{
			int val = ((float*)gradImg_x.data)[row_i * width + col_j];
			gradImg_x_8U.data[row_i * width + col_j] = abs(val);
		}
	}

	namedWindow("gradImg_x_8U", 0);
	imshow("gradImg_x_8U", gradImg_x_8U);
	waitKey();
#endif
	return 1;
}
int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y, Mat& angleImg, Mat& magImg)
{
	if (NULL == gradImg_x.data || NULL == gradImg_y.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
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
			((float*)angleImg.data)[row_i * width + col_j] = angle / 180 * 3.1415926525f;
			float mag1 = grad_x*grad_x + grad_y*grad_y;
			int t = *(int *)&mag1;
			t -= 0x3f800000;
			t >>= 1;
			t += 0x3f800000;
			((float*)magImg.data)[row_i * width + col_j] = *(float*)&t;
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
			angle /= 2;
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
		return MY_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;


	int binary_th = 100;
	for (int row_i = 0; row_i < height; row_i++)
	{
		int temp0 = row_i * width;
		for (int col_j = 0; col_j < width; col_j += 1)
		{
			//int pixVal = grayImg.at<uchar>(row_i, col_j);
			int temp1 = temp0 + col_j;
			int pixVal = grayImg.data[temp1];
			int dstVal = 0;
			if (pixVal > binary_th)
			{
				dstVal = 255;
			}
			else if (pixVal <= binary_th)
			{
				dstVal = 0;
			}
			//binaryImg.at<uchar>(row_i, col_j) = dstVal;
			binaryImg.data[temp1] = dstVal;
		}
	}

#ifdef IMG_SHOW
	namedWindow("binaryImg", 0);
	imshow("binaryImg", binaryImg);
	waitKey();
#endif

	return MY_OK;
}
int ustc_CalcHist(Mat grayImg, int* hist, int hist_len)
{
	if (NULL == grayImg.data || NULL == hist)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;

	//直方图清零
	for (int i = 0; i < hist_len; i++)
	{
		hist[i] = 0;
	}

	//计算直方图
	for (int row_i = 0; row_i < height; row_i++)
	{
		for (int col_j = 0; col_j < width; col_j += 1)
		{
			int pixVal = grayImg.data[row_i * width + col_j];
			hist[pixVal]++;
		}
	}
}
int ustc_SubImgMatch_gray(Mat grayImg, Mat subImg, int* x, int* y)
{

	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int min = 0;

	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg(height, width, CV_32FC1);
	//匹配误差初始化
	searchImg.setTo(FLT_MAX);

	for (int row = 0; row < sub_height; row++)
	{
		for (int col = 0; col < sub_width; col++)
		{

			int row_index = row;
			int col_index = col;
			int bigImg_pix = grayImg.data[row_index * width + col_index];
			//模板图上的像素
			int template_pix = subImg.data[col * sub_width + row];

			min += abs(bigImg_pix - template_pix);

		}
	}

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			int total_diff = 0;
			//遍历模板图上的每一个像素
			for (int a = 0; a < sub_height; a++)
			{
				for (int b = 0; b < sub_width; b++)
				{

					//大图上的像素位置
					int row_index = i + b;
					int col_index = j + a;
					int bigImg_pix = grayImg.data[row_index * width + col_index];
					//模板图上的像素
					int template_pix = subImg.data[b * sub_width + a];

					total_diff += abs(bigImg_pix - template_pix);
				}
			}
			//存储当前像素位置的匹配误差
			((float*)searchImg.data)[i * width + j] = total_diff;

			if (total_diff<min)
			{
				min = total_diff;
				*y = i;
				*x = j;
			}

		}
	}
	return 1;
}
int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int* x, int* y)
{

	if (NULL == colorImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}

	int width = colorImg.cols;
	int height = colorImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int min = 0;
	int  Img_g, Img_b, Img_r, sub_r, sub_g, sub_b, tmp_r, tmp_g, tmp_b;

	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg(height, width, CV_32FC1);
	//匹配误差初始化
	searchImg.setTo(FLT_MAX);

	for (int row = 0; row < sub_height; row++)
	{
		for (int col = 0; col < sub_width; col++)
		{
			Img_b = colorImg.data[3 * row*width + col];
			Img_g = colorImg.data[3 * row*width + col + 1];
			Img_r = colorImg.data[3 * row*width + col + 2];
			//模板图上的像素
			sub_b = subImg.data[3 * row*sub_width + col];
			sub_g = subImg.data[3 * row*sub_width + col + 1];
			sub_r = subImg.data[3 * row*sub_width + col + 2];

			tmp_b = Img_b - sub_b;
			tmp_g = Img_g - sub_g;
			tmp_r = Img_r - sub_r;
			min += abs(tmp_b) + abs(tmp_g) + abs(tmp_r);


		}
	}

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			int total_diff = 0;
			//遍历模板图上的每一个像素
			for (int a = 0; a < sub_height; a++)
			{
				for (int b = 0; b < sub_width; b++)
				{

					Img_b = colorImg.data[((i + a)*width + j + b) * 3];
					Img_g = colorImg.data[((i + a)*width + j + b) * 3 + 1];
					Img_r = colorImg.data[((i + a)*width + j + b) * 3 + 2];
					sub_b = subImg.data[(a*width + b) * 3];
					sub_g = subImg.data[(a*width + b) * 3 + 1];
					sub_r = subImg.data[(a*width + b) * 3 + 2];

					tmp_b = Img_b - sub_b;
					tmp_g = Img_g - sub_g;
					tmp_r = Img_r - sub_r;
					total_diff += abs(tmp_b) + abs(tmp_g) + abs(tmp_r);
					if (total_diff<min)
					{
						min = total_diff;
						*y = i;
						*x = j;
					}
				}
			}




		}
	}

	return 1;
}
int ustc_SubImgMatch_corr(Mat grayImg, Mat subImg, int* x, int* y){
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int m, n, grayImg_pix, template_pix, tmp;
	int s1 = 0;
	int s2 = 0;
	int s3 = 0;
	for (int row = 0; row < sub_height; row++)
	{
		m = row*width;
		n = row*sub_width;
		for (int col = 0; col < sub_width; col++)
		{
			grayImg_pix = grayImg.data[m + col];
			//模板图上的像素
			template_pix = subImg.data[n + col];
			s1 += grayImg_pix*template_pix;
			s2 += grayImg_pix*grayImg_pix;
			s3 += template_pix*template_pix;
		}
	}
	float min = 1.0f*s1 / (sqrt(s2)*sqrt(s3));
	float total_diff;


	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			total_diff = 0;
			s1 = 0;
			s2 = 0;
			s3 = 0;
			for (int a = 0; a < sub_height; a++)
			{
				m = (i + a)*width + j;
				n = a*width;
				for (int b = 0; b < sub_width; b++)
				{
					grayImg_pix = grayImg.data[m + b];
					template_pix = subImg.data[n + b];
					s1 += grayImg_pix*template_pix;
					s2 += grayImg_pix*grayImg_pix;
					s3 += template_pix*template_pix;
				}
			}
			total_diff = 1.0f*s1 / (sqrt(s2)*sqrt(s3));
			if (min>total_diff)
			{
				min = total_diff;
				*x = j;
				*y = i;
			}
		}
	}
	if (*x < 0 || *y < 0)
	{
		return 0;
	}
	else
		return 1;
}
int ustc_SubImgMatch_angle(Mat grayImg, Mat subImg, int* x, int* y){
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}
	int gray_width = grayImg.cols;
	int gray_height = grayImg.rows;
	Mat gray_gradImg_x(gray_height, gray_width, CV_32FC1);
	Mat gray_gradImg_y(gray_height, gray_width, CV_32FC1);
	gray_gradImg_x.setTo(0);
	gray_gradImg_y.setTo(0);


	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	Mat sub_gradImg_x(sub_height, sub_width, CV_32FC1);
	Mat sub_gradImg_y(sub_height, sub_width, CV_32FC1);
	sub_gradImg_x.setTo(0);
	sub_gradImg_y.setTo(0);

	int flag1 = ustc_CalcGrad(grayImg, gray_gradImg_x, gray_gradImg_y);

	int flag2 = ustc_CalcGrad(subImg, sub_gradImg_x, sub_gradImg_y);


	Mat gray_angleImg(gray_height, gray_width, CV_32FC1);
	gray_angleImg.setTo(0);
	Mat gray_magImg(gray_height, gray_width, CV_32FC1);
	gray_magImg.setTo(0);


	Mat sub_angleImg(sub_height, sub_width, CV_32FC1);
	sub_angleImg.setTo(0);
	Mat sub_magImg(sub_height, sub_width, CV_32FC1);
	sub_magImg.setTo(0);

	int flag3 = ustc_CalcAngleMag(gray_gradImg_x, gray_gradImg_y, gray_angleImg, gray_magImg);


	int flag4 = ustc_CalcAngleMag(sub_gradImg_x, sub_gradImg_y, sub_angleImg, sub_magImg);


	float min = 0;
	for (int i = 0; i < sub_height; i++)
	{
		for (int j = 0; j< sub_width; j++)
		{
			int row_index = i;
			int col_index = j;
			float bigImg_pix = ((float*)gray_angleImg.data)[i* gray_width + j];
			float template_pix = ((float*)sub_angleImg.data)[i* sub_width + j];
			//模板图上的像素
			min += abs(bigImg_pix - template_pix);
		}
	}

	for (int i = 0; i < gray_height - sub_height; i++)
	{
		for (int j = 0; j < gray_width - sub_width; j++)
		{
			float total_diff = 0;
			//遍历模板图上的每一个像素
			for (int a = 0; a < sub_height; a++)
			{
				for (int b = 0; b < sub_width; b++)
				{

					//大图上的像素位置
					int row_index = i + b;
					int col_index = j + a;

					float a1 = ((float*)gray_angleImg.data)[row_index * gray_width + col_index];
					float a2 = ((float*)sub_angleImg.data)[b * sub_width + a];

					total_diff += abs(a1 - a2);
				}
			}


			if (total_diff<min)
			{
				min = total_diff;
				*x = j;
				*y = i;
			}

		}
	}

	if (*x < 0 || *y < 0)
	{
		return 0;
	}
	else
		return 1;
}
int ustc_SubImgMatch_mag(Mat grayImg, Mat subImg, int* x, int* y){
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}
	int gray_width = grayImg.cols;
	int gray_height = grayImg.rows;
	Mat gray_gradImg_x(gray_height, gray_width, CV_32FC1);
	Mat gray_gradImg_y(gray_height, gray_width, CV_32FC1);
	gray_gradImg_x.setTo(0);
	gray_gradImg_y.setTo(0);

	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	Mat sub_gradImg_x(sub_height, sub_width, CV_32FC1);
	Mat sub_gradImg_y(sub_height, sub_width, CV_32FC1);
	sub_gradImg_x.setTo(0);
	sub_gradImg_y.setTo(0);

	int flag1 = ustc_CalcGrad(grayImg, gray_gradImg_x, gray_gradImg_y);
	int flag2 = ustc_CalcGrad(subImg, sub_gradImg_x, sub_gradImg_y);


	Mat gray_angleImg(gray_height, gray_width, CV_32FC1);
	gray_angleImg.setTo(0);
	Mat gray_magImg(gray_height, gray_width, CV_32FC1);
	gray_magImg.setTo(0);

	Mat sub_angleImg(sub_height, sub_width, CV_32FC1);
	sub_angleImg.setTo(0);
	Mat sub_magImg(sub_height, sub_width, CV_32FC1);
	sub_magImg.setTo(0);

	int flag3 = ustc_CalcAngleMag(gray_gradImg_x, gray_gradImg_y, gray_angleImg, gray_magImg);
	int flag4 = ustc_CalcAngleMag(sub_gradImg_x, sub_gradImg_y, sub_angleImg, sub_magImg);



	float min = 0;
	for (int i = 0; i < sub_height; i++)
	{
		for (int j = 0; j< sub_width; j++)
		{

			int row_index = i;
			int col_index = j;

			float bigImg_pix = ((float*)gray_magImg.data)[i* gray_width + j];
			float template_pix = ((float*)sub_magImg.data)[i* sub_width + j];
			min += abs(bigImg_pix - template_pix);
		}
	}

	for (int i = 0; i < gray_height - sub_height; i++)
	{
		for (int j = 0; j < gray_width - sub_width; j++)
		{
			float total_diff = 0;
			//遍历模板图上的每一个像素
			for (int a = 0; a < sub_height; a++)
			{
				for (int b = 0; b < sub_width; b++)
				{

					//大图上的像素位置
					int row_index = i + b;
					int col_index = j + a;

					float aa1 = ((float*)gray_magImg.data)[row_index * gray_width + col_index];
					float bb1 = ((float*)sub_magImg.data)[b * sub_width + a];

					total_diff += abs(aa1 - bb1);
				}
			}


			if (total_diff<min)
			{
				min = total_diff;
				*x = j;
				*y = i;
			}

		}
	}
	if (*x < 0 || *y < 0)
	{
		return 0;
	}
	else
		return 1;
}
int ustc_SubImgMatch_hist(Mat grayImg, Mat subImg, int* x, int* y){

	int grayHist[256], subHist[256];
	int a1 = ustc_CalcHist(subImg, subHist, 256);
	long min = 0x7fffffff;
	int x0, y0;
	for (int i = 0; i <= grayImg.rows - subImg.rows; i++)
	{
		for (int j = 0; j < grayImg.cols - subImg.cols; j++)
		{
			int a2 = ustc_CalcHist(grayImg(Rect(j, i, subImg.cols, subImg.rows)).clone(), grayHist, 256);
			long aa = 0;

			for (int k = 0; k < 256; k++)
			{
				aa += abs(grayHist[k] - subHist[k]);
			}
			if (aa< min)
			{
				min = aa;
				x0 = i;
				y0 = j;
			}
		}
	}

	*x = y0;
	*y = x0;
	if (*x < 0 || *y < 0)
	{
		return 0;
	}
	else
		return 1;
}
