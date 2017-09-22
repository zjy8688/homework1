#include "SubImageMatch.h"

#include<iostream>
#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#define SUB_IMAGE_MATCH_OK 1
#define SUB_IMAGE_MATCH_FAIL -1

using namespace cv;
using namespace std;

int ustc_ConvertBgr2Gray(Mat bgrImg, Mat& grayImg) 
{
	if (NULL == bgrImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = bgrImg.cols;
	int height = bgrImg.rows;
	for (int row_i = 0; row_i < height; row_i++)
	{
		int temp0 = row_i * width;
		for (int col_j = 0; col_j < width; col_j++)
		{
			int temp1 = temp0 + col_j;
			int b = bgrImg.data[3 * temp1];
			int g = bgrImg.data[3 * temp1 + 1];
			int r = bgrImg.data[3 * temp1 + 2];
			int grayVal = ((b * 117) + (g * 601) + (r * 234))>>10;
			grayImg.data[temp1] = grayVal;
		}
	}
	return SUB_IMAGE_MATCH_OK;
}

int ustc_CalcGrad(Mat grayImg, Mat& gradImg_x, Mat& gradImg_y) 
{
	if (NULL == grayImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	int temp0,temp1;
	int row_i, col_j;
	int grad_x, grad_y;
	for (row_i = 0; row_i < height; row_i++)
	{
		temp0 = row_i * width;
		gradImg_x.data[temp0] = 0;
		gradImg_x.data[temp0 + width - 1] = 0;
		gradImg_y.data[temp0] = 0;
		gradImg_y.data[temp0 + width - 1] = 0;
	}
	for (col_j = 1; col_j < width; col_j++)
	{
		gradImg_x.data[col_j] = 0;
		gradImg_x.data[temp0 + col_j] = 0;
		gradImg_y.data[col_j] = 0;
		gradImg_y.data[temp0 + col_j] = 0;
	}
	for (row_i = 1; row_i < height - 1; row_i++)
	{
		temp0 = row_i * width;
		for (col_j = 1; col_j < width - 1; col_j ++)
		{
			temp1 = temp0 + col_j;
			grad_x =
				grayImg.data[temp1-width+ 1]
				+ 2 * grayImg.data[temp1 + 1]
				+ grayImg.data[temp1+width + 1]
				- grayImg.data[temp1-width - 1]
				- 2 * grayImg.data[temp1 - 1]
				- grayImg.data[temp1+width - 1];
			((float*)gradImg_x.data)[temp1] = grad_x;

			grad_y =
				grayImg.data[temp1+width + 1]
				+ 2 * grayImg.data[temp1+width]
				+ grayImg.data[temp1+width -1]
				- grayImg.data[temp1-width - 1]
				- 2 * grayImg.data[temp1-width]
				- grayImg.data[temp1-width + 1];
			((float*)gradImg_y.data)[temp1] = grad_y;
		}
	}

	return SUB_IMAGE_MATCH_OK;
}


int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y, Mat& angleImg, Mat& magImg)
{
	if (NULL == gradImg_x.data|| NULL == gradImg_y.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = gradImg_x.cols;
	int height = gradImg_x.rows;
	int temp0, temp1;
	int row_i, col_j;
	float grad_x, grad_y,angle,b,m,ax,ay;
	for (row_i = 0; row_i < height; row_i++)
	{
		temp0 = row_i * width;
		angleImg.data[temp0] = 0;
		angleImg.data[temp0 + width - 1] = 0;
		magImg.data[temp0] = 0;
		magImg.data[temp0 + width - 1] = 0;
	}
	for (col_j = 1; col_j < width; col_j++)
	{
		angleImg.data[col_j] = 0;
		angleImg.data[temp0 + col_j] = 0;
		magImg.data[col_j] = 0;
		magImg.data[temp0 + col_j] = 0;
	}
	for (int row_i = 1; row_i < height - 1; row_i++)
	{
		for (int col_j = 1; col_j < width - 1; col_j += 1)
		{
			grad_x = ((float*)gradImg_x.data)[row_i * width + col_j];
			grad_y = ((float*)gradImg_y.data)[row_i * width + col_j];
			if (grad_x >= 0)
			{
				ax = grad_x;
			}
			else ax = -grad_x;
			if (grad_y >= 0)
			{
				ay = grad_y;
			}
			else ay = -grad_y;
			if (ax > ay)
			{
				b = ay;
				m = ax;
			}
			else
			{
				m = ay;
				b = ax;
			}
			float a = b / (m + (float)DBL_EPSILON);
			float s = a*a;
			angle = ((-0.046496475 * s + 0.15931422) * s - 0.327622764) * s * a + a;
			if (ay > ax) angle = 1.57079637f - angle;
			if (grad_x < 0) angle = 3.14159274f - angle;
			if (grad_y < 0) angle = 6.28318548f - angle;
			angle = angle * 57.29578;
			((float*)angleImg.data)[row_i * width + col_j] = angle;


			float f= grad_x*grad_x + grad_y*grad_y;
			float fhalf = 0.5f*f;
			int h = *(int*)&f;
			h=0x5f3759df-(h>>1);
			f = *(float*)&h;
			f = f*(1.5f - fhalf*f*f);
	     	((float*)magImg.data)[row_i * width + col_j] =1/f;

		}
	}

	return SUB_IMAGE_MATCH_OK;
}
int ustc_Threshold(Mat grayImg, Mat& binaryImg, int th) 
{
	if (NULL == grayImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	uchar tharry[256];
	for (int i = 0; i <= th; i++)
	{
		tharry[i] = 0;
	}
	for (int i =th; i <256; i++)
	{
		tharry[i] =255;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	for (int row_i = 0; row_i < height; row_i++)
	{
		int temp0 = row_i * width;
		for (int col_j = 0; col_j < width; col_j ++)
		{
			int temp1 = temp0 + col_j;
			int pixVal = grayImg.data[temp1];
			pixVal = tharry[pixVal];
			binaryImg.data[temp1] = pixVal;
		}
	}
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
	uchar*p= grayImg.data;
	for (int i = 0; i < hist_len; i++)
	{
		hist[i] = 0;
	}
	for (int row_i = height; row_i >0; row_i--)
	{
		for (int col_j = width; col_j >0; col_j --)
		{
			int pixVal =*p;
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
	int biao[511] = { 0 };
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int sub_i, sub_j, i, j;
	for (i = 0; i <= 255; i++)  biao[i] = 255 - i;
	for (i = 256; i <= 510; i++)  biao[i] = i - 255;
	int endi = 0, endj = 0;
	int value = 256 * sub_width*sub_height;
	for (i = 0; i <= (height - sub_height); i++)
	{
		for (j = 0; j <= (width - sub_width); j++)
		{
			int total_diff = 0;
			for (sub_i = 0; sub_i< sub_height; sub_i++)
			{
				for (sub_j = 0; sub_j< sub_width; sub_j++)
				{
					int row_index = i + sub_i;
					int col_index = j + sub_j;
					int bigImg_pix = grayImg.data[row_index * width + col_index];
					int template_pix = subImg.data[sub_i * sub_width + sub_j];
					total_diff += biao[bigImg_pix - template_pix + 255];
				}
			}
			if (value > total_diff)
			{
				value = total_diff;
				endi = i;
				endj = j;
			}
		}
	}
	*x = endj;
	*y = endi;
	return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int* x, int* y)
{
	if (NULL == colorImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int biao[511] = { 0 };
	int width = colorImg.cols;
	int height = colorImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int sub_i, sub_j, i, j;
	for (i = 0; i <= 255; i++)  biao[i] = 255 - i;
	for (i = 256; i <= 510; i++)  biao[i] = i - 255;
	int endi = 0, endj = 0;
	int value = 768 * sub_width*sub_height;
	for (i = 0; i <= (height - sub_height); i++)
	{
		for (j = 0; j <= (width - sub_width); j++)
		{
			int total_diff = 0;
			for (sub_i = 0; sub_i< sub_height; sub_i++)
			{
				for (sub_j = 0; sub_j< sub_width; sub_j++)
				{
					int row_index = i + sub_i;
					int col_index = j + sub_j;
					int wei1 = 3 * (row_index * width + col_index);
					int wei2 = 3 * (sub_i * sub_width + sub_j);
					int bigImg_pix = colorImg.data[wei1];
					int template_pix = subImg.data[wei2];
					total_diff += biao[bigImg_pix - template_pix +255];
					bigImg_pix = colorImg.data[wei1 + 1];
					template_pix = subImg.data[wei2 + 1];
					total_diff += biao[bigImg_pix - template_pix + 255];
					bigImg_pix = colorImg.data[wei1 + 2];
					template_pix = subImg.data[wei2 + 2];
					total_diff += biao[bigImg_pix - template_pix + 255];
				}
			}
			if (value > total_diff)
			{
				value = total_diff;
				endi = i;
				endj = j;
			}
		}
	}
	*x = endj;
	*y = endi;
	return SUB_IMAGE_MATCH_OK;
}


int ustc_SubImgMatch_corr(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int sub_i, sub_j, i, j;
	int endi = 0, endj = 0;
	float value = 0;
	float st1 = 0, st2 = 0, st3 = 0;
	for (sub_i = 0; sub_i< sub_height; sub_i++)
	{
		for (sub_j = 0; sub_j< sub_width; sub_j++)
		{
			int template_pix = subImg.data[sub_i * sub_width + sub_j];
			st3 += template_pix*template_pix;
		}
	}
	st3 = 1/st3;
	for (i = 0; i <= (height - sub_height); i++)
	{
		for (j = 0; j <= (width - sub_width); j++)
		{
			st1 = 0;
			st2 = 0;
			for (sub_i = 0; sub_i< sub_height; sub_i++)
			{
				for (sub_j = 0; sub_j< sub_width; sub_j++)
				{
					int row_index = i + sub_i;
					int col_index = j + sub_j;
					int bigImg_pix = grayImg.data[row_index * width + col_index];
					int template_pix = subImg.data[sub_i * sub_width + sub_j];
					st1 += bigImg_pix*template_pix;
					st2 += bigImg_pix*bigImg_pix;
				}
			}
			st1 = st1*st1;
			st2 =1/ st2;
			float xiangguan = st1*st2*st3;
			if (xiangguan > value)
			{
				value = xiangguan;
				endi = i;
				endj = j;
			}
		}
	}
	*x = endj;
	*y = endi;
	return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_angle(Mat grayImg, Mat subImg, int* x, int* y)
{
	int biao[721] = { 0 };
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int cha;
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	Mat angleImg(height, width, CV_32FC1);
	Mat magImg(height, width, CV_32FC1);
	Mat subangleImg(sub_height, sub_width, CV_32FC1);
	Mat submagImg(sub_height, sub_width, CV_32FC1);
	Mat gradImg_x(height, width, CV_32FC1);
	Mat gradImg_y(height, width, CV_32FC1);
	Mat subgradImg_x(sub_height, sub_width, CV_32FC1);
	Mat subgradImg_y(sub_height, sub_width, CV_32FC1);
	ustc_CalcGrad(grayImg, gradImg_x, gradImg_y);
	ustc_CalcGrad(subImg, subgradImg_x, subgradImg_y);
	ustc_CalcAngleMag(gradImg_x, gradImg_y, angleImg, magImg);
	ustc_CalcAngleMag(subgradImg_x, subgradImg_y, subangleImg, submagImg);
	Mat zhengangleImg(height, width, CV_32SC1);
	Mat zhengsubangleImg(sub_height, sub_width, CV_32SC1);
	Mat searchImg(height - sub_height + 1, width - sub_width + 1, CV_32SC1);
	int sub_i, sub_j, i, j;
	for (i = 0; i <= 180; i++) biao[i] = i;
	for (i = 181; i <= 360; i++) biao[i] = 360-i;
	for (i = 361; i <= 540; i++) biao[i] = i-360;
	for (i = 541; i <= 720; i++) biao[i] = 720-i;

	float shu;
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			shu = ((float*)angleImg.data)[i * width + j];
			((int*)zhengangleImg.data)[i * width + j] = shu;
		}
	}
	for (i = 0; i < sub_height; i++)
	{
		for (j = 0; j < sub_width; j++)
		{
			shu = ((float*)subangleImg.data)[i * sub_width + j];
			((int*)zhengsubangleImg.data)[i *sub_width + j] = shu;
		}
	}
	for (i = 0; i <= (height - sub_height); i++)
	{
		for (j = 0; j <= (width - sub_width); j++)
		{
			int total_diff = 0;
			for (sub_i = 1; sub_i< sub_height - 1; sub_i++)
			{
				for (sub_j = 1; sub_j< sub_width - 1; sub_j++)
				{
					int row_index = i + sub_i;
					int col_index = j + sub_j;
					int bigImg_pix = ((int*)zhengangleImg.data)[row_index * width + col_index];
					int template_pix = ((int*)zhengsubangleImg.data)[sub_i * sub_width + sub_j];
					cha =biao[ bigImg_pix - template_pix+360];
					total_diff += cha;
				}
			}
			((int*)searchImg.data)[i *(width - sub_width + 1) + j] = total_diff;
		}
	}
	int endi = 0;
	int value = ((int*)searchImg.data)[0];
	int temp0 = (height - sub_height + 1)*(width - sub_width + 1);
	for (i = 0; i < temp0; i++)
	{
		if (((int*)searchImg.data)[i] < value)
		{
			value = ((int*)searchImg.data)[i];
			endi = i;
		}
	}
	*x = endi % (width - sub_width + 1);
	*y = (endi - *x) / (width - sub_width + 1);
	return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_mag(Mat grayImg, Mat subImg, int* x, int* y)
{
	int sign[2] = { 1,-1 };
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int cha;
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	Mat angleImg(height, width, CV_32FC1);
	Mat magImg(height, width, CV_32FC1);
	Mat subangleImg(sub_height, sub_width, CV_32FC1);
	Mat submagImg(sub_height, sub_width, CV_32FC1);
	Mat gradImg_x(height, width, CV_32FC1);
	Mat gradImg_y(height, width, CV_32FC1);
	Mat subgradImg_x(sub_height, sub_width, CV_32FC1);
	Mat subgradImg_y(sub_height, sub_width, CV_32FC1);
	ustc_CalcGrad(grayImg, gradImg_x, gradImg_y);
	ustc_CalcGrad(subImg, subgradImg_x, subgradImg_y);
	ustc_CalcAngleMag(gradImg_x, gradImg_y, angleImg, magImg);
	ustc_CalcAngleMag(subgradImg_x, subgradImg_y, subangleImg, submagImg);
	Mat zhengmagImg(height, width, CV_32SC1);
	Mat zhengsubmagImg(sub_height, sub_width, CV_32SC1);
	int sub_i, sub_j, i, j;
	float shu;
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			shu = ((float*)magImg.data)[i * width + j];
			((int*)zhengmagImg.data)[i * width + j] = shu;
		}
	}
	for (i = 0; i < sub_height; i++)
	{
		for (j = 0; j < sub_width; j++)
		{
			shu = ((float*)submagImg.data)[i * sub_width + j];
			((int*)zhengsubmagImg.data)[i *sub_width + j] = shu;
		}
	}
	Mat searchImg(height - sub_height + 1, width - sub_width + 1, CV_32SC1);
	for (i = 0; i <= (height - sub_height); i++)
	{
		for (j = 0; j <= (width - sub_width); j++)
		{
			int total_diff = 0;
			for (sub_i = 1; sub_i< sub_height - 1; sub_i++)
			{
				for (sub_j = 1; sub_j< sub_width - 1; sub_j++)
				{
					int row_index = i + sub_i;
					int col_index = j + sub_j;
					int bigImg_pix = ((int*)zhengmagImg.data)[row_index * width + col_index];
					int template_pix = ((int*)zhengsubmagImg.data)[sub_i * sub_width + sub_j];
					cha = bigImg_pix - template_pix;
					int ha = cha >> 31;
					cha = sign[-ha] * cha;
					total_diff += cha;
				}
			}
			((int*)searchImg.data)[i *(width - sub_width + 1) + j] = total_diff;
		}
	}
	int endi = 0;
	float value = ((int*)searchImg.data)[0];
	int temp0 = (height - sub_height + 1)*(width - sub_width + 1);
	for (i = 0; i < temp0; i++)
	{
		if (((int*)searchImg.data)[i] < value)
		{
			value = ((int*)searchImg.data)[i];
			endi = i;
		}
	}
	*x = endi % (width - sub_width + 1);
	*y = (endi - *x) / (width - sub_width + 1);
	return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_hist(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int sign[2] = { 1,-1 };
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int subhist[256] = { 0 };
	ustc_CalcHist(subImg, subhist, 256);
	int bigImg_pix = 0, total_diff = 0;
	int hist[256] = { 0 };
	int hist1[256] = { 0 };
	int sub_i, sub_j, i, j;
	int endi = 0, endj = 0;
	int value;
	for (sub_i = 0; sub_i < sub_height; sub_i++)
	{
		for (sub_j = 0; sub_j < sub_width; sub_j++)
		{
			bigImg_pix = grayImg.data[sub_i * width + sub_j];
			hist[bigImg_pix]++;
		}
	}
	for (int ii = 0; ii < 256; ii++)
	{
		hist1[ii] = hist[ii];
		int temp = hist[ii] - subhist[ii];
		if (temp < 0) temp = -temp;
		total_diff += temp;
	}
	value = total_diff;
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
			for (int ii = 0; ii < 256; ii++)
			{
				hist1[ii] = hist[ii];
				int temp = hist[ii] - subhist[ii];
				if (temp < 0) temp = -temp;
				total_diff += temp;
			}
			if (value > total_diff)
			{
				value = total_diff;
				endi = i;
				endj = j;
			}
		}
		for (j = 1; j <= (width - sub_width); j++)
		{
			total_diff = 0;
			for (sub_j = 0; sub_j < sub_height; sub_j++)
			{
				bigImg_pix = grayImg.data[(i + sub_j) * width + j - 1];
				hist1[bigImg_pix]--;
				bigImg_pix = grayImg.data[(i + sub_j) * width + j + sub_width - 1];
				hist1[bigImg_pix]++;
			}
			for (int ii = 0; ii < 256; ii++)
			{
				int temp = hist1[ii] - subhist[ii];
				if (temp < 0) temp = -temp;
				total_diff += temp;
			}
			if (value > total_diff)
			{
				value = total_diff;
				endi = i;
				endj = j;
			}
		}
	}
	*x = endj;
	*y = endi;
	return SUB_IMAGE_MATCH_OK;
}
