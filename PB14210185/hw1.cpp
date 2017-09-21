/************考虑到data指针不能用于不连续存储的图像索引，故采用ptr并针对其优化************/

#include <opencv2/opencv.hpp>
#include <iostream>
#include "SubImageMatch.h"

using namespace cv;
using namespace std;

/************如果define了IMG_SHOW_WZG，函数里将包括显示图像及某些图像（如angleImg）的显示前处理）************/
//#define IMG_SHOW_WZG

int ustc_ConvertBgr2Gray(Mat bgrImg, Mat& grayImg)
{
	if (NULL == bgrImg.data)
	{
		cout << "bgr_image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	if (bgrImg.channels() != 3)
	{
		cout << "not a bgr_image ." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = bgrImg.cols;
	int height = bgrImg.rows;
	if (grayImg.cols != width || grayImg.rows != height)
	{
		cout << "bgr_image and gray_image have different size." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	if (bgrImg.isContinuous() && grayImg.isContinuous())
	{
		width = width * height;
		height = 1;
	}

	for (int row_i = height - 1; row_i >= 0; --row_i)
	{
		uchar* bgrImg_ptr = bgrImg.ptr<uchar>(row_i);
		uchar* grayImg_ptr = grayImg.ptr<uchar>(row_i);
		for (int col_j = width - 1; col_j >= 0; --col_j)
		{
			int b = *(bgrImg_ptr++);
			int g = *(bgrImg_ptr++);
			int r = *(bgrImg_ptr++);
			*(grayImg_ptr++) = ((b * 117) >> 10) + ((g * 601) >> 10) + ((r * 234) >> 10);
		}
	}

#ifdef IMG_SHOW_WZG
	imshow("grayImg", grayImg);
	waitKey();
#endif

	return SUB_IMAGE_MATCH_OK;
}

int ustc_CalcGrad(Mat grayImg, Mat& gradImg_x, Mat& gradImg_y)
{
	if (NULL == grayImg.data)
	{
		cout << "gray_image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	if (grayImg.channels() != 1)
	{
		cout << "not a gray_image ." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	if (gradImg_x.cols != width || gradImg_x.rows != height || gradImg_y.cols != width || gradImg_y.rows != height)
	{
		cout << "gray_image and grad_image have different size." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	gradImg_x.setTo(0);
	gradImg_y.setTo(0);

	for (int row_i = height - 2; row_i >= 1; --row_i)
	{
		float* gradImg_x_ptr = gradImg_x.ptr<float>(row_i);
		float* gradImg_y_ptr = gradImg_y.ptr<float>(row_i);
		uchar* r1 = grayImg.ptr<uchar>(row_i - 1);
		uchar* r2 = grayImg.ptr<uchar>(row_i);
		uchar* r3 = grayImg.ptr<uchar>(row_i + 1);
		for (int col_j = width - 2; col_j >= 1; --col_j)
		{
			uchar v11 = *(r1 + col_j + 1);
			uchar v10 = *(r2 + col_j + 1);
			uchar v1_1 = *(r3 + col_j + 1);
			uchar v_11 = *(r1 + col_j - 1);
			uchar v_10 = *(r2 + col_j - 1);
			uchar v_1_1 = *(r3 + col_j - 1);
			uchar v0_1 = *(r3 + col_j);
			uchar v01 = *(r1 + col_j);
			*(gradImg_x_ptr + col_j) = v11 + 2 * v10 + v1_1 - v_11 - 2 * v_10 - v_1_1;
			*(gradImg_y_ptr + col_j) = v_1_1 + 2 * v0_1 + v1_1 - v_11 - 2 * v01 - v11;
		}
	}

#ifdef IMG_SHOW_WZG
	Mat gradImg_x_8U(height, width, CV_8UC1);
	Mat gradImg_y_8U(height, width, CV_8UC1);
	for (int row_i = height - 1; row_i >= 0; --row_i)
	{
		uchar* gradImg_x_8U_ptr = gradImg_x_8U.ptr<uchar>(row_i);
		uchar* gradImg_y_8U_ptr = gradImg_y_8U.ptr<uchar>(row_i);
		float* gradImg_x_ptr = gradImg_x.ptr<float>(row_i);
		float* gradImg_y_ptr = gradImg_y.ptr<float>(row_i);
		for (int col_j = width - 1; col_j >= 0; --col_j)
		{
			*(gradImg_x_8U_ptr++) = abs(*(gradImg_x_ptr++));
			*(gradImg_y_8U_ptr++) = abs(*(gradImg_y_ptr++));
		}
	}
	imshow("gradImg_x", gradImg_x_8U);
	waitKey();
	imshow("gradImg_y", gradImg_y_8U);
	waitKey();
#endif

	return SUB_IMAGE_MATCH_OK;
}

int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y, Mat& angleImg, Mat& magImg)
{
	if (NULL == gradImg_x.data || NULL == gradImg_y.data)
	{
		cout << "grad_image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = gradImg_x.cols;
	int height = gradImg_x.rows;
	if (angleImg.cols != width || angleImg.rows != height || magImg.cols != width || magImg.rows != height)
	{
		cout << "grad_image and angle_image(mag_image) have different size." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	if (gradImg_x.isContinuous() && gradImg_y.isContinuous() && angleImg.isContinuous() && magImg.isContinuous())
	{
		width = width * height;
		height = 1;
	}

	for (int row_i = height - 1; row_i >= 0; --row_i)
	{
		float* gradImg_x_ptr = gradImg_x.ptr<float>(row_i);
		float* gradImg_y_ptr = gradImg_y.ptr<float>(row_i);
		float* angleImg_ptr = angleImg.ptr<float>(row_i);
		float* magImg_ptr = magImg.ptr<float>(row_i);
		for (int col_j = width - 1; col_j >= 0; --col_j)
		{
			float dx = *(gradImg_x_ptr);
			float dy = *(gradImg_y_ptr);
			//angle
			float ax = (dx >= 0 ? dx : -dx), ay = (dy >= 0 ? dy : -dy);
			float a = (ax >= ay ? ay : ax) / ((ax > ay ? ax : ay) + (float)DBL_EPSILON);
			float s = a*a;
			float angle = ((-0.0464964749 * s + 0.15931422) * s - 0.327622764) * s * a + a;
			if (ay > ax) angle = 1.57079637 - angle;
			if (dx < 0) angle = 3.14159274f - angle;
			if (dy < 0) angle = -angle;
			angle *= 180 / CV_PI;
			angle += 180;
			*(angleImg_ptr++) = angle;

			//mag
			float x = dy * dy + dx * dx;
			float xhalf = 0.5f * x;
			int i = *(int*)&x; // get bits for floating VALUE 
			i = 0x5f375a86 - (i >> 1); // gives initial guess y0
			x = *(float*)&i; // convert bits BACK to float
			x = x * (1.5f - xhalf * x * x); // Newton step, repeating increases accuracy
			x = x * (1.5f - xhalf * x * x); // Newton step, repeating increases accuracy
			x = x * (1.5f - xhalf * x * x); // Newton step, repeating increases accuracy
			*(magImg_ptr++) = 1 / x;

			//
			++gradImg_x_ptr;
			++gradImg_y_ptr;
		}
	}

#ifdef IMG_SHOW_WZG
	width = gradImg_x.cols;
	height = gradImg_x.rows;
	Mat angleImg_8U(height, width, CV_8UC1);
	for (int row_i = height - 1; row_i >= 0; --row_i)
	{
		float* angleImg_ptr = angleImg.ptr<float>(row_i);
		uchar* angleImg_8U_ptr = angleImg_8U.ptr<uchar>(row_i);
		for (int col_j = width - 1; col_j >= 0; --col_j)
		{
			*(angleImg_8U_ptr++) = *(angleImg_ptr++) / 2;
		}
	}
	imshow("angleImg", angleImg_8U);
	waitKey();
#endif

	return SUB_IMAGE_MATCH_OK;
}

int ustc_Threshold(Mat grayImg, Mat& binaryImg, int th)
{
	if (NULL == grayImg.data)
	{
		cout << "gray_image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	if (grayImg.channels() != 1)
	{
		cout << "not a gray_image ." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	if (binaryImg.cols != width || binaryImg.rows != height)
	{
		cout << "binary_image and gray_image have different size." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int th_array[256];
	for (int i = 255; i >= 0; --i)
	{
		th_array[i] = i >= th ? 255 : 0;
	}

	if (grayImg.isContinuous() && binaryImg.isContinuous())
	{
		width = width * height;
		height = 1;
	}

	for (int row_i = height - 1; row_i >= 0; --row_i)
	{
		uchar* grayImg_ptr = grayImg.ptr<uchar>(row_i);
		uchar* binaryImg_ptr = binaryImg.ptr<uchar>(row_i);
		for (int col_j = width - 1; col_j >= 0; --col_j)
		{
			*(binaryImg_ptr++) = th_array[*(grayImg_ptr++)];
		}
	}

#ifdef IMG_SHOW_WZG
	imshow("binaryImg", binaryImg);
	waitKey();
#endif

	return SUB_IMAGE_MATCH_OK;
}

int ustc_CalcHist(Mat grayImg, int* hist, int hist_len)
{
	if (NULL == grayImg.data)
	{
		cout << "gray_image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	if (grayImg.channels() != 1)
	{
		cout << "not a gray_image ." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;

	for (int i = hist_len - 1; i >= 0; --i)
	{
		hist[i] = 0;
	}

	if (grayImg.isContinuous())
	{
		width = width * height;
		height = 1;
	}

	for (int row_i = height - 1; row_i >= 0; --row_i)
	{
		uchar* grayImg_ptr = grayImg.ptr<uchar>(row_i);
		for (int col_j = width - 1; col_j >= 0; --col_j)
		{
			++hist[*(grayImg_ptr++)];
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

	if (grayImg.channels() != 1)
	{
		cout << "not a gray_image ." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	if (1 != subImg.channels())
	{
		cout << "different channels." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	if (sub_width > width || sub_height > height)
	{
		cout << "invalid subImg." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int total_diff_min = INT_MAX;
	for (int row_i = height - sub_height; row_i >= 0; --row_i)
	{ 
		for (int col_j = width - sub_width; col_j >= 0; --col_j)
		{
			int total_diff = 0;
			for (int sub_row_i = sub_height - 1; sub_row_i >= 0; --sub_row_i)
			{
				int row_index = row_i + sub_row_i;
				uchar* grayImg_ptr = grayImg.ptr<uchar>(row_index);
				uchar* subImg_ptr = subImg.ptr<uchar>(sub_row_i);
				for (int sub_col_j = sub_width - 1; sub_col_j >= 0; --sub_col_j)
				{			
					int col_index = col_j + sub_col_j;
					int diff = *(grayImg_ptr + col_index) - *(subImg_ptr + sub_col_j);
					total_diff += diff >= 0 ? diff : -diff;
				}
			}
			if (total_diff < total_diff_min)
			{
				total_diff_min = total_diff;
				*y = row_i;
				*x = col_j;
			}
		}
	}
	
	return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int* x, int* y)
{
	if (NULL == colorImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	if (colorImg.channels() != 3)
	{
		cout << "not a bgr_image ." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	if (3 != subImg.channels())
	{
		cout << "different channels." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = colorImg.cols;
	int height = colorImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	if (sub_width > width || sub_height > height)
	{
		cout << "invalid subImg." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int total_diff_min = INT_MAX;
	for (int row_i = height - sub_height; row_i >= 0; --row_i)
	{
		for (int col_j = width - sub_width; col_j >= 0; --col_j)
		{
			int total_diff = 0;
			for (int sub_row_i = sub_height - 1; sub_row_i >= 0; --sub_row_i)
			{
				int row_index = row_i + sub_row_i;
				uchar* colorImg_ptr = colorImg.ptr<uchar>(row_index);
				uchar* subImg_ptr = subImg.ptr<uchar>(sub_row_i);
				for (int sub_col_j = sub_width - 1; sub_col_j >= 0; --sub_col_j)
				{
					int col_index = col_j + sub_col_j;
					uchar* color_pixel = colorImg_ptr + 3 * col_index;
					uchar* sub_pixel = subImg_ptr + 3 * sub_col_j;
					int b_diff = *color_pixel - *sub_pixel;
					int g_diff = *(color_pixel + 1) - *(sub_pixel + 1);
					int r_diff = *(color_pixel + 2) - *(sub_pixel + 2);
					total_diff += (b_diff >= 0 ? b_diff : -b_diff) + (g_diff >= 0 ? g_diff : -g_diff) + (r_diff >= 0 ? r_diff : -r_diff);
				}
			}
			if (total_diff < total_diff_min)
			{
				total_diff_min = total_diff;
				*y = row_i;
				*x = col_j;
			}
		}
	}

	return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_corr(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	if (grayImg.channels() != 1)
	{
		cout << "not a gray_image ." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	if (1 != subImg.channels())
	{
		cout << "different channels." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	if (sub_width > width || sub_height > height)
	{
		cout << "invalid subImg." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	float r_1 = 0;
	for (int row_i = height - sub_height; row_i >= 0; --row_i)
	{
		for (int col_j = width - sub_width; col_j >= 0; --col_j)
		{
			int s_t = 0;
			int s_2 = 0;
			int t_2 = 0;
			for (int sub_row_i = sub_height - 1; sub_row_i >= 0; --sub_row_i)
			{
				int row_index = row_i + sub_row_i;
				uchar* grayImg_ptr = grayImg.ptr<uchar>(row_index);
				uchar* subImg_ptr = subImg.ptr<uchar>(sub_row_i);
				for (int sub_col_j = sub_width - 1; sub_col_j >= 0; --sub_col_j)
				{
					int col_index = col_j + sub_col_j;
					int val = *(grayImg_ptr + col_index);
					int sub_val = *(subImg_ptr + sub_col_j);
					s_t += val * sub_val;
					s_2 += val * val;
					t_2 += sub_val * sub_val;
				}
			}
			//
			float s_2f = (float)s_2;
			float s_2half = 0.5f * s_2f;
			int i = *(int*)&s_2f; // get bits for floating VALUE 
			i = 0x5f375a86 - (i >> 1); // gives initial guess y0
			s_2f = *(float*)&i; // convert bits BACK to float
			s_2f = s_2f * (1.5f - s_2half * s_2f * s_2f); // Newton step, repeating increases accuracy
			s_2f = s_2f * (1.5f - s_2half * s_2f * s_2f); // Newton step, repeating increases accuracy
			s_2f = s_2f * (1.5f - s_2half * s_2f * s_2f); // Newton step, repeating increases accuracy
			s_2f = 1 / s_2f;
			//
			float t_2f = (float)t_2;
			float t_2half = 0.5f * t_2f;
			int j = *(int*)&t_2f; // get bits for floating VALUE 
			j = 0x5f375a86 - (j >> 1); // gives initial guess y0
			t_2f = *(float*)&j; // convert bits BACK to float
			t_2f = t_2f * (1.5f - t_2half * t_2f * t_2f); // Newton step, repeating increases accuracy
			t_2f = t_2f * (1.5f - t_2half * t_2f * t_2f); // Newton step, repeating increases accuracy
			t_2f = t_2f * (1.5f - t_2half * t_2f * t_2f); // Newton step, repeating increases accuracy
			t_2f = 1 / t_2f;

			float r = (float)s_t / (s_2f * t_2f);
			if (1.0 - r < 1.0 - r_1)
			{
				r_1 = r;
				*y = row_i;
				*x = col_j;
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

	if (grayImg.channels() != 1)
	{
		cout << "not a gray_image ." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	if (1 != subImg.channels())
	{
		cout << "different channels." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	if (sub_width > width || sub_height > height)
	{
		cout << "invalid subImg." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	Mat angleImg(height, width, CV_16UC1);
	angleImg.setTo(0);

	for (int row_i = height - 2; row_i >= 1; --row_i)
	{
		uchar* r1 = grayImg.ptr<uchar>(row_i - 1);
		uchar* r2 = grayImg.ptr<uchar>(row_i);
		uchar* r3 = grayImg.ptr<uchar>(row_i + 1);
		ushort* angleImg_ptr = angleImg.ptr<ushort>(row_i);
		for (int col_j = width - 2; col_j >= 1; --col_j)
		{
			uchar v11 = *(r1 + col_j + 1);
			uchar v10 = *(r2 + col_j + 1);
			uchar v1_1 = *(r3 + col_j + 1);
			uchar v_11 = *(r1 + col_j - 1);
			uchar v_10 = *(r2 + col_j - 1);
			uchar v_1_1 = *(r3 + col_j - 1);
			uchar v0_1 = *(r3 + col_j);
			uchar v01 = *(r1 + col_j);
			int dx = v11 + 2 * v10 + v1_1 - v_11 - 2 * v_10 - v_1_1;
			int dy = v_1_1 + 2 * v0_1 + v1_1 - v_11 - 2 * v01 - v11;
			
			//angle
			float ax = (dx >= 0 ? dx : -dx), ay = (dy >= 0 ? dy : -dy);
			float a = (ax >= ay ? ay : ax) / ((ax > ay ? ax : ay) + (float)DBL_EPSILON);
			float s = a*a;
			float angle = ((-0.0464964749 * s + 0.15931422) * s - 0.327622764) * s * a + a;
			if (ay > ax) angle = 1.57079637 - angle;
			if (dx < 0) angle = 3.14159274f - angle;
			if (dy < 0) angle = -angle;
			angle *= 180 / CV_PI;
			angle += 180;
			*(angleImg_ptr + col_j) = (ushort)angle;
		}
	}

	Mat sub_angleImg(sub_height, sub_width, CV_16UC1);
	sub_angleImg.setTo(0);

	for (int sub_row_i = sub_height - 2; sub_row_i >= 1; --sub_row_i)
	{
		uchar* r1 = subImg.ptr<uchar>(sub_row_i - 1);
		uchar* r2 = subImg.ptr<uchar>(sub_row_i);
		uchar* r3 = subImg.ptr<uchar>(sub_row_i + 1);
		ushort* sub_angleImg_ptr = sub_angleImg.ptr<ushort>(sub_row_i);
		for (int sub_col_j = sub_width - 2; sub_col_j >= 1; --sub_col_j)
		{
			uchar v11 = *(r1 + sub_col_j + 1);
			uchar v10 = *(r2 + sub_col_j + 1);
			uchar v1_1 = *(r3 + sub_col_j + 1);
			uchar v_11 = *(r1 + sub_col_j - 1);
			uchar v_10 = *(r2 + sub_col_j - 1);
			uchar v_1_1 = *(r3 + sub_col_j - 1);
			uchar v0_1 = *(r3 + sub_col_j);
			uchar v01 = *(r1 + sub_col_j);
			int dx = v11 + 2 * v10 + v1_1 - v_11 - 2 * v_10 - v_1_1;
			int dy = v_1_1 + 2 * v0_1 + v1_1 - v_11 - 2 * v01 - v11;

			//angle
			float ax = (dx >= 0 ? dx : -dx), ay = (dy >= 0 ? dy : -dy);
			float a = (ax >= ay ? ay : ax) / ((ax > ay ? ax : ay) + (float)DBL_EPSILON);
			float s = a*a;
			float angle = ((-0.0464964749 * s + 0.15931422) * s - 0.327622764) * s * a + a;
			if (ay > ax) angle = 1.57079637 - angle;
			if (dx < 0) angle = 3.14159274f - angle;
			if (dy < 0) angle = -angle;
			angle *= 180 / CV_PI;
			angle += 180;
			*(sub_angleImg_ptr + sub_col_j) = (ushort)angle;
		}
	}

	int total_diff_min = INT_MAX;
	for (int row_i = height - sub_height; row_i >= 0; --row_i)
	{
		for (int col_j = width - sub_width; col_j >= 0; --col_j)
		{
			int total_diff = 0;
			for (int sub_row_i = sub_height - 1; sub_row_i >= 0; --sub_row_i)
			{
				int row_index = row_i + sub_row_i;
				ushort* angleImg_ptr = angleImg.ptr<ushort>(row_index);
				ushort* sub_angleImg_ptr = sub_angleImg.ptr<ushort>(sub_row_i);
				for (int sub_col_j = sub_width - 1; sub_col_j >= 0; --sub_col_j)
				{
					int col_index = col_j + sub_col_j;
					int diff = *(angleImg_ptr + col_index) - *(sub_angleImg_ptr + sub_col_j);
					diff = diff >= 0 ? diff : -diff;
					diff = diff > 180 ? 360 - diff : diff;
					total_diff += diff;
				}
			}
			if (total_diff < total_diff_min)
			{
				total_diff_min = total_diff;
				*y = row_i;
				*x = col_j;
			}
		}
	}

	return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_mag(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	if (grayImg.channels() != 1)
	{
		cout << "not a gray_image ." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	if (1 != subImg.channels())
	{
		cout << "different channels." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	if (sub_width > width || sub_height > height)
	{
		cout << "invalid subImg." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	Mat magImg(height, width, CV_16UC1);
	magImg.setTo(0);

	for (int row_i = height - 2; row_i >= 1; --row_i)
	{
		uchar* r1 = grayImg.ptr<uchar>(row_i - 1);
		uchar* r2 = grayImg.ptr<uchar>(row_i);
		uchar* r3 = grayImg.ptr<uchar>(row_i + 1);
		ushort* magImg_ptr = magImg.ptr<ushort>(row_i);
		for (int col_j = width - 2; col_j >= 1; --col_j)
		{
			uchar v11 = *(r1 + col_j + 1);
			uchar v10 = *(r2 + col_j + 1);
			uchar v1_1 = *(r3 + col_j + 1);
			uchar v_11 = *(r1 + col_j - 1);
			uchar v_10 = *(r2 + col_j - 1);
			uchar v_1_1 = *(r3 + col_j - 1);
			uchar v0_1 = *(r3 + col_j);
			uchar v01 = *(r1 + col_j);
			int dx = v11 + 2 * v10 + v1_1 - v_11 - 2 * v_10 - v_1_1;
			int dy = v_1_1 + 2 * v0_1 + v1_1 - v_11 - 2 * v01 - v11;

			//mag
			float m = dy * dy + dx * dx;
			float mhalf = 0.5f * m;
			int i = *(int*)&m; // get bits for floating VALUE 
			i = 0x5f375a86 - (i >> 1); // gives initial guess y0
			m = *(float*)&i; // convert bits BACK to float
			m = m * (1.5f - mhalf * m * m); // Newton step, repeating increases accuracy
			m = m * (1.5f - mhalf * m * m); // Newton step, repeating increases accuracy
			m = m * (1.5f - mhalf * m * m); // Newton step, repeating increases accuracy
			*(magImg_ptr + col_j) = (ushort)(1 / m);
		}
	}

	Mat sub_magImg(sub_height, sub_width, CV_16UC1);
	sub_magImg.setTo(0);

	for (int sub_row_i = sub_height - 2; sub_row_i >= 1; --sub_row_i)
	{
		uchar* r1 = subImg.ptr<uchar>(sub_row_i - 1);
		uchar* r2 = subImg.ptr<uchar>(sub_row_i);
		uchar* r3 = subImg.ptr<uchar>(sub_row_i + 1);
		ushort* sub_magImg_ptr = sub_magImg.ptr<ushort>(sub_row_i);
		for (int sub_col_j = sub_width - 2; sub_col_j >= 1; --sub_col_j)
		{
			uchar v11 = *(r1 + sub_col_j + 1);
			uchar v10 = *(r2 + sub_col_j + 1);
			uchar v1_1 = *(r3 + sub_col_j + 1);
			uchar v_11 = *(r1 + sub_col_j - 1);
			uchar v_10 = *(r2 + sub_col_j - 1);
			uchar v_1_1 = *(r3 + sub_col_j - 1);
			uchar v0_1 = *(r3 + sub_col_j);
			uchar v01 = *(r1 + sub_col_j);
			int dx = v11 + 2 * v10 + v1_1 - v_11 - 2 * v_10 - v_1_1;
			int dy = v_1_1 + 2 * v0_1 + v1_1 - v_11 - 2 * v01 - v11;

			//mag
			float m = dy * dy + dx * dx;
			float mhalf = 0.5f * m;
			int i = *(int*)&m; // get bits for floating VALUE 
			i = 0x5f375a86 - (i >> 1); // gives initial guess y0
			m = *(float*)&i; // convert bits BACK to float
			m = m * (1.5f - mhalf * m * m); // Newton step, repeating increases accuracy
			m = m * (1.5f - mhalf * m * m); // Newton step, repeating increases accuracy
			m = m * (1.5f - mhalf * m * m); // Newton step, repeating increases accuracy
			*(sub_magImg_ptr + sub_col_j) = (ushort)(1 / m);
		}
	}

	int total_diff_min = INT_MAX;
	for (int row_i = height - sub_height; row_i >= 0; --row_i)
	{
		for (int col_j = width - sub_width; col_j >= 0; --col_j)
		{
			int total_diff = 0;
			for (int sub_row_i = sub_height - 1; sub_row_i >= 0; --sub_row_i)
			{
				int row_index = row_i + sub_row_i;
				ushort* magImg_ptr = magImg.ptr<ushort>(row_index);
				ushort* sub_magImg_ptr = sub_magImg.ptr<ushort>(sub_row_i);
				for (int sub_col_j = sub_width - 1; sub_col_j >= 0; --sub_col_j)
				{
					int col_index = col_j + sub_col_j;
					int diff = *(magImg_ptr + col_index) - *(sub_magImg_ptr + sub_col_j);
					total_diff += diff >= 0 ? diff : -diff;
				}
			}
			if (total_diff < total_diff_min)
			{
				total_diff_min = total_diff;
				*y = row_i;
				*x = col_j;
			}
		}
	}

	return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_hist(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	if (grayImg.channels() != 1)
	{
		cout << "not a gray_image ." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	if (1 != subImg.channels())
	{
		cout << "different channels." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	if (sub_width > width || sub_height > height)
	{
		cout << "invalid subImg." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int hist_len = 256;
	int sub_hist[256];
	int hist[256];

	for (int i = hist_len - 1; i >= 0; --i)
	{
		hist[i] = sub_hist[i] = 0;
	}

	for (int sub_row_i = sub_height - 1; sub_row_i >= 0; --sub_row_i)
	{
		uchar* subImg_ptr = subImg.ptr<uchar>(sub_row_i);
		for (int sub_col_j = sub_width - 1; sub_col_j >= 0; --sub_col_j)
		{
			++sub_hist[*(subImg_ptr++)];
		}
	}

	int total_diff_min = INT_MAX;
	for (int row_i = height - sub_height; row_i >= 0; --row_i)
	{
		for (int col_j = width - sub_width; col_j >= 0; --col_j)
		{
			int total_diff = 0;
			for (int sub_row_i = sub_height - 1; sub_row_i >= 0; --sub_row_i)
			{
				int row_index = row_i + sub_row_i;
				uchar* grayImg_ptr = grayImg.ptr<uchar>(row_index);
				for (int sub_col_j = sub_width - 1; sub_col_j >= 0; --sub_col_j)
				{
					int col_index = col_j + sub_col_j;
					++hist[*(grayImg_ptr + col_index)];
				}
			}
			for (int i = hist_len - 1; i >= 0; --i)
			{
				int diff = hist[i] - sub_hist[i];
				total_diff += diff >= 0 ? diff : -diff;
				hist[i] = 0;
			}
			if (total_diff < total_diff_min)
			{
				total_diff_min = total_diff;
				*y = row_i;
				*x = col_j;
			}
		}
	}

	return SUB_IMAGE_MATCH_OK;
}
