#include "opencv2/opencv.hpp"
using namespace cv;
#include <iostream>
#include <stdio.h>
using namespace std;

int ustc_CovertBgr2Gray(Mat bgrImg, Mat &grayImg)
{
	if (NULL == bgrImg.data)
	{
		cout << "image is NULL." << endl;
		return -1;
	}

	int c1 = bgrImg.channels();
	int c2 = grayImg.channels();
	if (c1 != 3 || c2 != 1)
	{
		cout << "channels is wrong." << endl;
		return -1;
	}

	int width = bgrImg.cols;
	int height = bgrImg.rows;


	for (int row_i = 0; row_i < height; row_i++)
	{
		for (int col_j = 0; col_j < width; col_j += 1)
		{
			int expos = row_i * width + col_j;
			int pos = 3 * expos;
			int b = bgrImg.data[pos + 0];
			int g = bgrImg.data[pos + 1];
			int r = bgrImg.data[pos + 2];

			int grayVal = (b * 306 + g * 601 + r * 117) >> 10;
			grayImg.data[expos] = grayVal;
		}
	}
}

int ustc_Threshold(Mat grayImg, Mat &binaryImg, int th)
{
	if (NULL == grayImg.data)
	{
		cout << "image is NULL." << endl;
		return -1;
	}

	int c1 = grayImg.channels();
	int c2 = binaryImg.channels();
	if (c1 != 1 || c2 != 1)
	{
		cout << "channels is wrong." << endl;
		return -1;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;

	int binary_th = th;
	for (int row_i = 0; row_i < height; row_i++)
	{
		int temp0 = row_i * width;
		for (int col_j = 0; col_j < width; col_j += 1)
		{
			//int pixVal = grayImg.at<uchar>(row_i, col_j);
			int temp1 = temp0 + col_j;
			int pixVal = grayImg.data[temp1];
			int dstVal = 0;
			dstVal = (1 - (((pixVal - binary_th) >> 31) & 1)) * 255;
			binaryImg.data[temp1] = dstVal;
		}
	}

	return 1;
}

int ustc_CalcGrad(Mat grayImg, Mat &gradImg_x, Mat &gradImg_y)
{
	if (NULL == grayImg.data)
	{
		cout << "image is NULL." << endl;
		return -1;
	}

	int c1 = grayImg.channels();
	int c2 = gradImg_x.channels();
	int c3 = gradImg_y.channels();
	if (c1 != 1 || c2 != 1 || c3 != 1)
	{
		cout << "channels is wrong." << endl;
		return -1;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	gradImg_x.setTo(0);
	gradImg_y.setTo(0);

	//计算x方向梯度图
	for (int row_i = 1; row_i < height - 1; row_i++)
	{
		for (int col_j = 1; col_j < width - 1; col_j += 1)
		{
			int f_x1 = row_i* width + col_j;
			int grad_x =
				grayImg.data[f_x1 - width + 1]
				+ 2 * grayImg.data[f_x1 + 1]
				+ grayImg.data[f_x1 + width + 1]
				- grayImg.data[f_x1 - width - 1]
				- 2 * grayImg.data[f_x1 - 1]
				- grayImg.data[f_x1 + width - 1];

			((float*)gradImg_x.data)[f_x1] = grad_x;

			int grad_y =
				grayImg.data[f_x1 + width - 1]
				+ 2 * grayImg.data[f_x1 + width]
				+ grayImg.data[f_x1 + width + 1]
				- grayImg.data[f_x1 - width - 1]
				- 2 * grayImg.data[f_x1 - width]
				- grayImg.data[f_x1 - width + 1];

			((float*)gradImg_y.data)[f_x1 - width] = grad_y;
		}
	}

}

int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y, Mat &angleImg, Mat &magImg)
{
	if (NULL == gradImg_x.data || NULL == gradImg_y.data)
	{
		cout << "image is NULL." << endl;
		return -1;
	}

	int c1 = angleImg.channels();
	int c2 = gradImg_x.channels();
	int c3 = gradImg_y.channels();
	int c4 = magImg.channels();
	if (c1 != 1 || c2 != 1 || c3 != 1 || c4 != 1)
	{
		cout << "channels is wrong." << endl;
		return -1;
	}
	int width = angleImg.cols;
	int height = angleImg.rows;
	//int aangle[] = { 11520, 6801, 3593, 1824, 916, 458, 229, 115, 57, 29, 14, 7, 4, 2, 1 };
	//int i = 0;
	//int x_new, y_new;
	angleImg.setTo(0);
	magImg.setTo(0);

	//计算角度图
	for (int row_i = 1; row_i < height - 1; row_i++)
	{
		for (int col_j = 1; col_j < width - 1; col_j += 1)
		{
			int pos = row_i * width + col_j;
			float grad_x = ((float*)gradImg_x.data)[pos];
			float grad_y = ((float*)gradImg_y.data)[pos];
			float squareadd = grad_x*grad_x + grad_y*grad_y;
			float angle = atan2(grad_y, grad_x);

			int ag = angle * 1024;
			angle = angle * 180 / CV_PI + ((ag >> 31) & 1) * 360;
			((float*)angleImg.data)[row_i * width + col_j] = angle;

			float a = squareadd;
			unsigned int i2 = *(unsigned int *)&squareadd;
			i2 = (i2 + 0x3f76cf62) >> 1;
			squareadd = *(float *)&i2;
			squareadd = (squareadd + a / squareadd) * 0.5;
			//float mag= sqrt(squareadd);
			((float*)magImg.data)[row_i * width + col_j] = squareadd;
		}
	}

}

int ustc_CalcHist(Mat grayImg, int* hist, int hist_len)
{
	if (NULL == grayImg.data || NULL == hist)
	{
		cout << "image is NULL." << endl;
		return -1;
	}

	int c1 = grayImg.channels();
	if (c1 != 1)
	{
		cout << "channels is wrong." << endl;
		return -1;
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
		for (int col_j = 0; col_j < width; col_j += 2)
		{
			int pos = row_i * width + col_j;
			int pixVal = grayImg.data[pos];
			hist[pixVal]++;
			++col_j;
			pixVal = grayImg.data[pos + 1];
			hist[pixVal]++;
		}
	}
}



int ustc_SubImgmatch_gray(Mat grayImg, Mat subImg, int *x, int *y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return -1;
	}
	int c1 = grayImg.channels();
	int c2 = subImg.channels();
	if (c1 != 1 || c2 != 1)
	{
		cout << "channels is wrong." << endl;
		return -1;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int min = 999999;

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			int total_diff = 0;
			//遍历模板图上的每一个像素
			for (int x1 = 0; x1 < sub_height; x1++)
			{
				for (int y1 = 0; y1 < sub_width; y1++)
				{
					//大图上的像素位置
					int row_index = i + y1;
					int col_index = j + x1;
					int bigImg_pix = grayImg.data[row_index * width + col_index];
					//模板图上的像素
					int template_pix = subImg.data[y1 * sub_width + x1];

					int pos = bigImg_pix - template_pix;
					total_diff += (1 - 2 * ((pos >> 31) & 1))*pos;
				}
			}
			if (total_diff < min) {
				min = total_diff;
				*x = j;
				*y = i;
			}
		}
	}
}


int ustc_SubImgmatch_bgr(Mat colorImg, Mat subImg, int *mat_x, int *mat_y)
{
	if (NULL == colorImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return -1;
	}
	int c1 = colorImg.channels();
	int c2 = subImg.channels();
	if (c1 != 3 || c2 != 3)
	{
		cout << "channels is wrong." << endl;
		return -1;
	}

	int width = colorImg.cols;
	int height = colorImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int min = 99999999;

	//该图用于记录每一个像素位置的匹配误差
	//Mat searchImg(height, width, CV_32FC1);
	//匹配误差初始化
	//searchImg.setTo(FLT_MAX);

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			int total_diff = 0;
			//遍历模板图上的每一个像素
			for (int x = 0; x < sub_height; x++)
			{
				for (int y = 0; y < sub_width; y++)
				{
					//大图上的像素位置
					int row_index = i + y;
					int col_index = j + x;
					int pos1 = 3 * (row_index * width + col_index);
					int pos2 = 3 * (y * sub_width + x);
					int bigImg_pix1 = colorImg.data[pos1];
					//模板图上的像素
					int bigImg_pix2 = colorImg.data[pos1 + 1];
					//模板图上的像素
					int bigImg_pix3 = colorImg.data[pos1 + 2];
					//模板图上的像素
					int template_pix1 = subImg.data[pos2];
					int template_pix2 = subImg.data[pos2 + 1];
					int template_pix3 = subImg.data[pos2 + 2];

					int pos11 = bigImg_pix1 - template_pix1;
					int pos22 = bigImg_pix2 - template_pix2;
					int pos33 = bigImg_pix3 - template_pix3;
					total_diff += (1 - 2 * ((pos11 >> 31) & 1))*pos11 + (1 - 2 * ((pos22 >> 31) & 1))*pos22 + (1 - 2 * ((pos33 >> 31) & 1))*pos33;
				}
			}
			//存储当前像素位置的匹配误差
			//((float*)searchImg.data)[i * width + j] = total_diff;
			if (total_diff >= min);
			else {
				min = total_diff;
				*mat_x = j;
				*mat_y = i;
			}
		}
	}
}

int ustc_match_hist(Mat grayImg, Mat subImg, int *x, int *y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return -1;
	}
	int c1 = grayImg.channels();
	int c2 = subImg.channels();
	if (c1 != 1 || c2 != 1)
	{
		cout << "channels is wrong." << endl;
		return -1;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int hist_temp[256] = { 0 };
	int hist_sub[256] = { 0 };
	int min = 9999999;

	//该图用于记录每一个像素位置的匹配误差
	//Mat searchImg(height, width, CV_32FC1);
	//匹配误差初始化
	//searchImg.setTo(FLT_MAX);

	//遍历大图每一个像素，注意行列的起始、终止坐标

	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)//  报错！
		{
			//清零
			for (int k = 0; k < 256; k++) {
				hist_temp[k] = 0;
				hist_sub[k] = 0;
			}
			//计算当前位置直方图
			for (int x1 = 0; x1 < sub_height; x1++)
			{
				for (int y1 = 0; y1 < sub_width; y1++)
				{
					//大图上的像素位置
					int row_index = i + y1;
					int col_index = j + x1;
					int bigImg_pix = grayImg.data[row_index * width + col_index];
					//模板图上的像素
					int template_pix = subImg.data[y1 * sub_width + x1];
					hist_temp[bigImg_pix]++;
					hist_sub[template_pix]++;
				}
			}

			//根据直方图计算匹配误差
			int total_diff = 0;
			for (int ii = 0; ii < 256; ii++)
			{
				total_diff += abs(hist_temp[ii] - hist_sub[ii]);
			}
			//存储当前像素位置的匹配误差
			//((float*)searchImg.data)[i * width + j] = total_diff;
			if (total_diff < min) {
				min = total_diff;
				*x = j;
				*y = i;
			}

		}
	}
}

int ustc_SubImgmatch_corr(Mat grayImg, Mat subImg, int *mat_x, int *mat_y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return -1;
	}
	int c1 = grayImg.channels();
	int c2 = subImg.channels();
	if (c1 != 1 || c2 != 1)
	{
		cout << "channels is wrong." << endl;
		return -1;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	float max = 0;

	//该图用于记录每一个像素位置的匹配误差
	//Mat searchImg(height, width, CV_32FC1);
	//匹配误差初始化
	//searchImg.setTo(FLT_MAX);

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			int total_mul = 0;
			int total_xx = 0;
			int total_yy = 0;
			float total = 0;
			//遍历模板图上的每一个像素
			for (int x1 = 0; x1 < sub_height; x1++)
			{
				for (int y1 = 0; y1 < sub_width; y1++)
				{
					//大图上的像素位置
					int row_index = i + y1;
					int col_index = j + x1;
					int bigImg_pix = grayImg.data[row_index * width + col_index];
					//模板图上的像素
					int template_pix = subImg.data[y1 * sub_width + x1];
					total_mul += bigImg_pix*template_pix;
					total_xx += bigImg_pix*bigImg_pix;
					total_yy += template_pix*template_pix;
				}
			}
			//存储当前像素位置的匹配误差
			float squareadd1 = total_xx;
			float a1 = squareadd1;
			unsigned int i1 = *(unsigned int *)&squareadd1;
			i1 = (i1 + 0x3f76cf62) >> 1;
			squareadd1 = *(float *)&i1;
			squareadd1 = (squareadd1 + a1 / squareadd1) * 0.5;

			float squareadd2 = total_yy;
			a1 = squareadd2;
			i1 = *(unsigned int *)&squareadd2;
			i1 = (i1 + 0x3f76cf62) >> 1;
			squareadd2 = *(float *)&i1;
			squareadd2 = (squareadd2 + a1 / squareadd2) * 0.5;

			float temp = squareadd1*squareadd2;
			total = ((float)total_mul * 8) / temp;
			//total = ((float)total_mul*8)/sqrt(total_xx)/sqrt(total_yy);
			//((float*)searchImg.data)[i * width + j] = total;
			if (total> max) {
				max = total;
				*mat_x = j;
				*mat_y = i;
			}
		}
	}
}

int ustc_SubImgMatch_angle(Mat grayImg, Mat subImg, int *mat_x, int *mat_y) {

	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return -1;
	}
	int c1 = grayImg.channels();
	int c2 = subImg.channels();
	if (c1 != 1 || c2 != 1)
	{
		cout << "channels is wrong." << endl;
		return -1;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	Mat gradImg_x1(height, width, CV_32FC1);
	Mat gradImg_y1(height, width, CV_32FC1);
	ustc_CalcGrad(grayImg, gradImg_x1, gradImg_y1);

	Mat angleImg1(height, width, CV_32FC1);
	angleImg1.setTo(0);
	Mat magImg1(height, width, CV_32FC1);
	magImg1.setTo(0);
	ustc_CalcAngleMag(gradImg_x1, gradImg_y1, angleImg1, magImg1);

	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	Mat gradImg_x2(sub_height, sub_width, CV_32FC1);
	Mat gradImg_y2(sub_height, sub_width, CV_32FC1);
	ustc_CalcGrad(subImg, gradImg_x2, gradImg_y2);

	Mat angleImg2(sub_height, sub_width, CV_32FC1);
	angleImg2.setTo(0);
	Mat magImg2(sub_height, sub_width, CV_32FC1);
	magImg2.setTo(0);
	ustc_CalcAngleMag(gradImg_x2, gradImg_y2, angleImg2, magImg2);

	int min = 9999999;

	//该图用于记录每一个像素位置的匹配误差
	//Mat searchImg(height, width, CV_32FC1);
	//匹配误差初始化
	//searchImg.setTo(FLT_MAX);

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 1; i < height - sub_height - 1; i++)
	{
		for (int j = 1; j < width - sub_width - 1; j++)
		{
			int total_diff = 0;
			//遍历模板图上的每一个像素
			for (int x1 = 1; x1 < sub_height - 1; x1++)
			{
				for (int y1 = 1; y1 < sub_width - 1; y1++)
				{
					//大图上的像素位置
					int row_index = i + y1 - 1;
					int col_index = j + x1 - 1;
					int bigImg_pix = ((float*)angleImg1.data)[row_index * width + col_index];
					//模板图上的像素
					int template_pix = ((float*)angleImg2.data)[y1 * sub_width + x1];

					total_diff += abs(bigImg_pix - template_pix);
				}
			}
			//存储当前像素位置的匹配误差
			//((float*)searchImg.data)[i * width + j] = total_diff;
			if (total_diff < min) {
				min = total_diff;
				*mat_x = j - 1;
				*mat_y = i - 1;
			}
		}
	}
}

int ustc_SubImgMatch_mag(Mat grayImg, Mat subImg, int *mat_x, int *mat_y) {

	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return -1;
	}
	int c1 = grayImg.channels();
	int c2 = subImg.channels();
	if (c1 != 1 || c2 != 1)
	{
		cout << "channels is wrong." << endl;
		return -1;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	Mat gradImg_x1(height, width, CV_32FC1);
	Mat gradImg_y1(height, width, CV_32FC1);
	ustc_CalcGrad(grayImg, gradImg_x1, gradImg_y1);

	Mat angleImg1(height, width, CV_32FC1);
	angleImg1.setTo(0);
	Mat magImg1(height, width, CV_32FC1);
	magImg1.setTo(0);
	ustc_CalcAngleMag(gradImg_x1, gradImg_y1, angleImg1, magImg1);

	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	Mat gradImg_x2(sub_height, sub_width, CV_32FC1);
	Mat gradImg_y2(sub_height, sub_width, CV_32FC1);
	ustc_CalcGrad(subImg, gradImg_x2, gradImg_y2);

	Mat angleImg2(sub_height, sub_width, CV_32FC1);
	angleImg2.setTo(0);
	Mat magImg2(sub_height, sub_width, CV_32FC1);
	magImg2.setTo(0);
	ustc_CalcAngleMag(gradImg_x2, gradImg_y2, angleImg2, magImg2);

	int min = 9999999;

	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg(height, width, CV_32FC1);
	//匹配误差初始化
	searchImg.setTo(FLT_MAX);

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 1; i < height - sub_height - 1; i++)
	{
		for (int j = 1; j < width - sub_width - 1; j++)
		{
			int total_diff = 0;
			//遍历模板图上的每一个像素
			for (int x1 = 1; x1 < sub_height - 1; x1++)
			{
				for (int y1 = 1; y1 < sub_width - 1; y1++)
				{
					//大图上的像素位置
					int row_index = i + y1 - 1;
					int col_index = j + x1 - 1;
					int bigImg_pix = ((float*)magImg1.data)[row_index * width + col_index];
					//模板图上的像素
					int template_pix = ((float*)magImg2.data)[y1 * sub_width + x1];

					total_diff += abs(bigImg_pix - template_pix);
				}
			}
			//存储当前像素位置的匹配误差
			((float*)searchImg.data)[i * width + j] = total_diff;
			if (total_diff < min) {
				min = total_diff;
				*mat_x = j - 1;
				*mat_y = i - 1;
			}
		}
	}
}
