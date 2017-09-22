#include"SubImageMatch.h"

int ustc_ConvertBgr2Gray(Mat bgrImg, Mat& grayImg)
{
	if (NULL == bgrImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (3!=bgrImg.channels())
    {
        cout << "It is not a colour image ." << endl;
        return SUB_IMAGE_MATCH_FAIL;
    }
	unsigned short int width, length;
	width = bgrImg.cols;
	length = bgrImg.rows;
	grayImg = Mat(length, width, CV_8UC1, Scalar(0));
	for (unsigned short int i = 0; i < length; i++)
		for (unsigned short int j = 0; j < width; j++)
		{
			uchar *datapoint = &bgrImg.data[3 * (i*width + j) + 0];
			grayImg.data[i*width + j] = (*datapoint * 117 + *(datapoint + 1) * 601 + *(datapoint + 2) * 306) / 1024;
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
	if (1!= grayImg.channels())
	{
		cout << "It is not a gray image ." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	
	unsigned short int width = grayImg.cols, length = grayImg.rows;
	short int ux, uy;
	short int a00, a01, a02;
	short int a10, a11, a12;
	short int a20, a21, a22;
	gradImg_x = Mat(length, width, CV_16SC1, Scalar(0));
	gradImg_y = Mat(length, width, CV_16SC1, Scalar(0));
	for (unsigned short int i = 1; i < length - 1; ++i)
	{
		for (unsigned short int j = 1; j < width - 1; ++j)
		{
			uchar* datapoint = &grayImg.data[(i)*width + j];
			a00 = *(datapoint - width - 1);
			a01 = *(datapoint - width);
			a02 = *(datapoint - width + 1);
			a10 = *(datapoint - 1);
			a11 = *(datapoint);
			a12 = *(datapoint + 1);
			a20 = *(datapoint + width - 1);
			a21 = *(datapoint + width);
			a22 = *(datapoint + width + 1);
			ux = a20*(1) + a10*(2) + a00*(1) + a02*(-1) + a12*(-2) + a22*(-1);
			uy = a02*(1) + a01*(2) + a00*(1) + a20*(-1) + a21*(-2) + a22*(-1);
			((short int*)gradImg_x.data)[i*width + j] = ux;
			((short int*)gradImg_y.data)[i*width + j] = uy;
		}
	}

	return SUB_IMAGE_MATCH_OK;
}




int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y, Mat& angleImg, Mat& magImg)
{
	if (NULL == gradImg_x.data || NULL == gradImg_y.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if ((1 != gradImg_x.channels())|| (1 != gradImg_y.channels()))
	{
		cout << "It is not a gray image ." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	
	if ((gradImg_x.cols != gradImg_y.cols) || (gradImg_x.rows != gradImg_y.rows))
	{
		cout << "The image is illegal!" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = gradImg_x.cols, length = gradImg_x.rows;
	float ux, uy, angle;
	angleImg = Mat(length, width, CV_32FC1, Scalar(0));
	magImg = Mat(length, width, CV_32FC1, Scalar(0));
	for (unsigned short int i = 0; i<length; i++)
		for (unsigned short int j = 0; j < width; j++)
		{
			ux = ((short int*)gradImg_x.data)[i*width + j];
			uy = ((short int*)gradImg_y.data)[i*width + j];

			float number = ux*ux + uy*uy;
			long m;
			float xx, yy;
			const float f = 1.5f;
			xx = number*0.5f;
			yy = number;
			m = *(long *)&yy;
			m = 0x5f3759df - (m >> 1);
			yy = *(float *)&m;
			yy = yy*(f - (xx*yy*yy));
			yy = yy*(f - (xx*yy*yy));
			((float*)magImg.data)[i*width + j] = number*yy;




			float ax = std::abs(ux), ay = std::abs(uy);
			float a = std::min(ax, ay) / (std::max(ax, ay) + (float)DBL_EPSILON);
			float s = a*a;
			float r = ((-0.0464964749 * s + 0.15931422) * s - 0.327622764) * s * a + a;
			if (ay > ax) r = 1.57079637 - r;
			if (ux < 0) r = 3.14159274f - r;
			if (uy < 0) r = 6.28318548f - r;
			((float*)angleImg.data)[i*width + j] = r * 36000 / 628;

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
	if (1 != grayImg.channels())
	{
		cout << "It is not a gray image ." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	
	if (th < 0 || th>255)
	{
		cout << "Are you fuck......dding me?" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	unsigned short int width = grayImg.cols, length = grayImg.rows;
	binaryImg = Mat(length, width, CV_8UC1, Scalar(0));
	for (unsigned short int i = 0; i<length; i++)
		for (unsigned short int j = 0; j < width; j++)
		{
			short int flag = 1 + (grayImg.data[i*width + j] - th >> 15);
			binaryImg.data[i*width + j] = flag * 255;
		}
	return SUB_IMAGE_MATCH_OK;
}



int ustc_CalcHist(Mat grayImg, int* hist, int hist_len)
{
	if (NULL == grayImg.data || NULL == hist)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (1 != grayImg.channels())
	{
		cout << "It is not a gray image ." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	

	unsigned short int width = grayImg.cols;
	unsigned short int height = grayImg.rows;
	for (unsigned short int i = 0; i < hist_len; i++)
	{
		hist[i] = 0;
	}
	for (unsigned short int row_i = 0; row_i < height; row_i++)
	{
		for (unsigned short int col_j = 0; col_j < width; col_j += 1)
		{
			unsigned short int pixVal = grayImg.data[row_i * width + col_j];
			hist[pixVal]++;
		}
	}
	return SUB_IMAGE_MATCH_OK;
}



int ustc_SubImgMatch_gray(Mat grayImg, Mat subImg, int *x, int *y)
{

	if (NULL == grayImg.data || NULL == subImg.data || NULL == x || NULL == y)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (1 != grayImg.channels())
	{
		cout << "It is not a gray image ." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (grayImg.rows < subImg.rows || grayImg.cols < subImg.cols)
	{
		cout << "The match is illegal!." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	unsigned short int width = grayImg.cols;
	unsigned short int height = grayImg.rows;
	unsigned short int sub_width = subImg.cols;
	unsigned short int sub_height = subImg.rows;
	Mat searchImg(height, width, CV_32SC1, Scalar(INT_MAX));
	unsigned short int testheight = height - sub_height;
	unsigned short int testwidth = width - sub_width;
	for (unsigned short int i = 0; i <= testheight; i++)
	{
		for (unsigned short int j = 0; j <= testwidth; j++)
		{
			int total_diff = 0;
			for (unsigned short int p = 0; p < sub_height; p++)
			{
				for (unsigned short int q = 0; q < sub_width; q++)
				{
					total_diff += abs(grayImg.data[(i + p)* width + j + q] - subImg.data[p * sub_width + q]);
				}
			}
			((int*)searchImg.data)[i * width + j] = total_diff;
		}
	}
	int min = ((int*)searchImg.data)[0];
	for (unsigned short int i = 0; i <= testheight; i++)
		for (unsigned short int j = 0; j <= testwidth; j++)
		{
			if (((int*)searchImg.data)[i*width + j] <= min)
			{
				min = ((int*)searchImg.data)[i*width + j];
				*x = j, *y = i;
			}
		}
	return SUB_IMAGE_MATCH_OK;

}



int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int* x, int* y)
{
	if (NULL == colorImg.data || NULL == subImg.data || NULL == x || NULL == y)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if ((3!= colorImg.channels())||(3!=subImg.channels()))
	{
		cout << "It is not a bgr image ." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	
	if (colorImg.rows < subImg.rows || colorImg.cols < subImg.cols)
	{
		cout << "The match is illegal!." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	unsigned short int width = colorImg.cols;
	unsigned short int height = colorImg.rows;
	unsigned short int sub_width = subImg.cols;
	unsigned short int sub_height = subImg.rows;
	Mat searchImg(height, width, CV_32SC1);
	searchImg.setTo(INT_MAX);
	unsigned short int testheight = height - sub_height;
	unsigned short int testwidth = width - sub_width;
	for (unsigned short int i = 0; i <= testheight; i++)
	{
		for (unsigned short int j = 0; j <= testwidth; j++)
		{
			unsigned int total_diff_b = 0;
			unsigned int total_diff_g = 0;
			unsigned int total_diff_r = 0;
			for (unsigned short int p = 0; p < sub_height; p++)
			{
				for (unsigned short int q = 0; q < sub_width; q++)
				{
					uchar* datapoint = &colorImg.data[3 * ((i + p) * width + j + q) + 0];
					uchar* datapoint2 = &subImg.data[3 * (p * sub_width + q) + 0];
					unsigned short int bigImg_pix_b = *(datapoint);
					unsigned short int bigImg_pix_g = *(datapoint + 1);
					unsigned short int bigImg_pix_r = *(datapoint + 2);
					unsigned short int template_pix_b = *(datapoint2);
					unsigned short int template_pix_g = *(datapoint2 + 1);
					unsigned short int template_pix_r = *(datapoint2 + 2);

					total_diff_b += abs(bigImg_pix_b - template_pix_b);
					total_diff_g += abs(bigImg_pix_g - template_pix_g);
					total_diff_r += abs(bigImg_pix_r - template_pix_r);

				}
			}
			((int*)searchImg.data)[i * width + j] = total_diff_b + total_diff_g + total_diff_r;
		}
	}
	int min = ((int*)searchImg.data)[0];
	for (unsigned short int i = 0; i <= testheight; i++)
		for (unsigned short int j = 0; j <= testwidth; j++)
		{
			if (((int*)searchImg.data)[i*width + j] <= min)
			{
				min = ((int*)searchImg.data)[i*width + j];
				*x = j, *y = i;
			}
		}
	return SUB_IMAGE_MATCH_OK;

}




int ustc_SubImgMatch_corr(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data || NULL == x || NULL == y)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if ((1 != grayImg.channels())||(1!=subImg.channels()))
	{
		cout << "It is not a gray image ." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (grayImg.rows < subImg.rows || grayImg.cols < subImg.cols)
	{
		cout << "The match is illegal!." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	unsigned short int width = grayImg.cols;
	unsigned short int height = grayImg.rows;
	unsigned short int sub_width = subImg.cols;
	unsigned short int sub_height = subImg.rows;
	Mat searchImg(height, width, CV_32FC1, Scalar(FLT_MAX));
	unsigned short int testheight = height - sub_height;
	unsigned short int testwidth = width - sub_width;
	for (unsigned short int i = 0; i <= testheight; i++)
	{
		for (unsigned short int j = 0; j <= testwidth; j++)
		{

			float Img_R = 0;
			int ST = 0, SS = 0, TT = 0;
			for (unsigned short int p = 0; p <sub_height; p++)
			{
				for (unsigned short int q = 0; q <sub_width; q++)
				{
					unsigned short int row_index = i + p;
					unsigned short int col_index = j + q;
					ST += ((grayImg.data[row_index*width + col_index])*(subImg.data[p*sub_width + q]));
					SS += ((grayImg.data[row_index*width + col_index])*(grayImg.data[row_index*width + col_index]));
					TT += ((subImg.data[p*sub_width + q])*(subImg.data[p*sub_width + q]));
				}
			}
			float sqrtSS = SS, sqrtTT = TT;//开始给SS和TT开方
			long m;
			float xx, yy;
			const float f = 1.5f;
			xx = sqrtSS*0.5f;
			yy = sqrtSS;
			m = *(long *)&yy;
			m = 0x5f3759df - (m >> 1);
			yy = *(float *)&m;
			yy = yy*(f - (xx*yy*yy));
			yy = yy*(f - (xx*yy*yy));
			sqrtSS = sqrtSS*yy;

			xx = sqrtTT*0.5f;
			yy = sqrtTT;
			m = *(long *)&yy;
			m = 0x5f3759df - (m >> 1);
			yy = *(float *)&m;
			yy = yy*(f - (xx*yy*yy));
			yy = yy*(f - (xx*yy*yy));
			sqrtTT = sqrtTT*yy;
			((float*)searchImg.data)[i * width + j] = ST / (sqrtSS*sqrtTT);
		}
	}
	float max = ((float*)searchImg.data)[0];
	for (unsigned short int i = 0; i <= testheight; i++)
		for (unsigned short int j = 0; j <= testwidth; j++)
		{
			if (((float*)searchImg.data)[i*width + j] >= max)
			{
				max = ((float*)searchImg.data)[i*width + j];
				*x = j, *y = i;
			}
		}
	return SUB_IMAGE_MATCH_OK;
}



int ustc_SubImgMatch_angle(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data || NULL == x || NULL == y)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if ((1 != grayImg.channels()) || (1 != subImg.channels()))
	{
		cout << "It is not a gray image ." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (grayImg.rows<subImg.rows || grayImg.cols<subImg.cols)
	{
		cout << "The image is illegal." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	unsigned short int width = grayImg.cols;
	unsigned short int length = grayImg.rows;
	unsigned short int sub_width = subImg.cols;
	unsigned short int sub_length = subImg.rows;
	int ux, uy, sub_ux, sub_uy;
	unsigned short int a00, a01, a02;
	unsigned short int a10, a11, a12;
	unsigned short int a20, a21, a22;
	Mat angleImg_big = Mat(length, width, CV_16SC1, Scalar(0));
	Mat angleImg_small = Mat(sub_length, sub_width, CV_16SC1, Scalar(0));
	for (unsigned short int i = 1; i < length - 1; ++i)
	{
		for (unsigned short int j = 1; j < width - 1; ++j)
		{
			uchar* datapoint = &grayImg.data[(i)*width + j];
			a00 = *(datapoint - width - 1);
			a01 = *(datapoint - width);
			a02 = *(datapoint - width + 1);
			a10 = *(datapoint - 1);
			a11 = *(datapoint);
			a12 = *(datapoint + 1);
			a20 = *(datapoint + width - 1);
			a21 = *(datapoint + width);
			a22 = *(datapoint + width + 1);
			ux = a20*(1) + a10*(2) + a00*(1) + a02*(-1) + a12*(-2) + a22*(-1);
			uy = a02*(1) + a01*(2) + a00*(1) + a20*(-1) + a21*(-2) + a22*(-1);
			//快速计算角度的方法
			float ax = std::abs(ux), ay = std::abs(uy);
			float a = std::min(ax, ay) / (std::max(ax, ay) + (float)DBL_EPSILON);
			float s = a*a;
			float r = ((-0.0464964749 * s + 0.15931422) * s - 0.327622764) * s * a + a;
			if (ay > ax) r = 1.57079637 - r;
			if (ux < 0) r = 3.14159274f - r;
			if (uy < 0) r = 6.28318548f - r;
			((short int*)angleImg_big.data)[i*width + j] = r * 36000 / 628;
		}
	}
	for (unsigned short int i = 1; i < sub_length - 1; ++i)
	{
		for (unsigned short int j = 1; j < sub_width - 1; ++j)
		{
			uchar* datapoint2 = &subImg.data[(i)*sub_width + j];
			a00 = *(datapoint2 - sub_width - 1);
			a01 = *(datapoint2 - sub_width);
			a02 = *(datapoint2 - sub_width + 1);
			a10 = *(datapoint2 - 1);
			a11 = *(datapoint2);
			a12 = *(datapoint2 + 1);
			a20 = *(datapoint2 + sub_width - 1);
			a21 = *(datapoint2 + sub_width);
			a22 = *(datapoint2 + sub_width + 1);
			sub_ux = a20*(1) + a10*(2) + a00*(1) + a02*(-1) + a12*(-2) + a22*(-1);
			sub_uy = a02*(1) + a01*(2) + a00*(1) + a20*(-1) + a21*(-2) + a22*(-1);
			//快速计算角度的方法
			float ax = std::abs(sub_ux), ay = std::abs(sub_uy);
			float a = std::min(ax, ay) / (std::max(ax, ay) + (float)DBL_EPSILON);
			float s = a*a;
			float r = ((-0.0464964749 * s + 0.15931422) * s - 0.327622764) * s * a + a;
			if (ay > ax) r = 1.57079637 - r;
			if (sub_ux < 0) r = 3.14159274f - r;
			if (sub_uy < 0) r = 6.28318548f - r;
			((short int*)angleImg_small.data)[i*sub_width + j] = r * 36000 / 628;
		}
	}

	Mat searchImg(length, width, CV_32SC1);
	searchImg.setTo(INT_MAX);
	unsigned short int testlength = length - sub_length;
	unsigned short int testwidth = width - sub_width;
	for (unsigned short int i = 0; i <= testlength; i++)
	{
		for (unsigned short int j = 0; j <= testwidth; j++)
		{
			int total_diff = 0;
			for (unsigned short int p = 1; p < sub_length - 1; p++)
			{
				for (unsigned short int q = 1; q < sub_width - 1; q++)
				{
					unsigned short int row_index = i + p;
					unsigned short int col_index = j + q;
					short int bigImg_pix = ((short int*)angleImg_big.data)[row_index * width + col_index];
					short int template_pix = ((short int*)angleImg_small.data)[p * sub_width + q];
					short int diff = abs(bigImg_pix - template_pix);
					if (diff > 180)
					{
						diff = 360 - diff;
					}
					total_diff += diff;
				}
			}
			((int*)searchImg.data)[i * width + j] = total_diff;
		}
	}
	int min = ((int*)searchImg.data)[0];
	for (unsigned short int i = 0; i <= testlength; i++)
		for (unsigned short int j = 0; j <= testwidth; j++)
		{
			if (((int*)searchImg.data)[i*width + j] <= min)
			{
				min = ((int*)searchImg.data)[i*width + j];
				*x = j; *y = i;
			}
		}
	return SUB_IMAGE_MATCH_OK;

}



int ustc_SubImgMatch_mag(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data || NULL == x || NULL == y)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if ((1 != grayImg.channels()) || (1 != subImg.channels()))
	{
		cout << "It is not a gray image ." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (grayImg.rows<subImg.rows || grayImg.cols<subImg.cols)
	{
		cout << "The image is illegal!" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	unsigned short int width = grayImg.cols;
	unsigned short int length = grayImg.rows;
	unsigned short int sub_width = subImg.cols;
	unsigned short int sub_length = subImg.rows;
	unsigned short int ux, uy, sub_ux, sub_uy;
	unsigned short int a00, a01, a02;
	unsigned short int a10, a11, a12;
	unsigned short int a20, a21, a22;
	Mat magImg_big = Mat(length, width, CV_16UC1, Scalar(0));
	Mat magImg_small = Mat(sub_length, sub_width, CV_16UC1, Scalar(0));
	for (unsigned short int i = 1; i < length - 1; ++i)
	{
		for (unsigned short int j = 1; j < width - 1; ++j)
		{
			uchar* datapoint = &grayImg.data[(i)*width + j];
			a00 = *(datapoint - width - 1);
			a01 = *(datapoint - width);
			a02 = *(datapoint - width + 1);
			a10 = *(datapoint - 1);
			a11 = *(datapoint);
			a12 = *(datapoint + 1);
			a20 = *(datapoint + width - 1);
			a21 = *(datapoint + width);
			a22 = *(datapoint + width + 1);
			ux = a20*(1) + a10*(2) + a00*(1) + a02*(-1) + a12*(-2) + a22*(-1);
			uy = a02*(1) + a01*(2) + a00*(1) + a20*(-1) + a21*(-2) + a22*(-1);
			//快速开方的方法
			float number = ux*ux + uy*uy;
			long m;
			float xx, yy;
			const float f = 1.5f;
			xx = number*0.5f;
			yy = number;
			m = *(long *)&yy;
			m = 0x5f3759df - (m >> 1);
			yy = *(float *)&m;
			yy = yy*(f - (xx*yy*yy));
			yy = yy*(f - (xx*yy*yy));
			((short int*)magImg_big.data)[i*width + j] = number*yy;
		}
	}
	for (unsigned short int i = 1; i < sub_length - 1; ++i)
	{
		for (unsigned short int j = 1; j < sub_width - 1; ++j)
		{
			uchar* datapoint2 = &subImg.data[(i)*sub_width + j];
			a00 = *(datapoint2 - sub_width - 1);
			a01 = *(datapoint2 - sub_width);
			a02 = *(datapoint2 - sub_width + 1);
			a10 = *(datapoint2 - 1);
			a11 = *(datapoint2);
			a12 = *(datapoint2 + 1);
			a20 = *(datapoint2 + sub_width - 1);
			a21 = *(datapoint2 + sub_width);
			a22 = *(datapoint2 + sub_width + 1);
			sub_ux = a20*(1) + a10*(2) + a00*(1) + a02*(-1) + a12*(-2) + a22*(-1);
			sub_uy = a02*(1) + a01*(2) + a00*(1) + a20*(-1) + a21*(-2) + a22*(-1);
			//快速开方的方法
			float number = sub_ux*sub_ux + sub_uy*sub_uy;
			long m;
			float xx, yy;
			const float f = 1.5f;
			xx = number*0.5f;
			yy = number;
			m = *(long *)&yy;
			m = 0x5f3759df - (m >> 1);
			yy = *(float *)&m;
			yy = yy*(f - (xx*yy*yy));
			yy = yy*(f - (xx*yy*yy));
			((short int*)magImg_small.data)[i*sub_width + j] = number*yy;
		}
	}

	Mat searchImg(length, width, CV_32SC1, Scalar(INT_MAX));
	unsigned short int testlength = length - sub_length;
	unsigned short int testwidth = width - sub_width;
	for (unsigned short int i = 0; i <= testlength; i++)
	{
		for (unsigned short int j = 0; j <= testwidth; j++)
		{
			long total_diff = 0;
			for (unsigned short int p = 1; p < sub_length - 1; p++)
			{
				for (unsigned short int q = 1; q < sub_width - 1; q++)
				{
					unsigned short int row_index = i + p;
					unsigned short int col_index = j + q;
					unsigned short int bigImg_pix = ((short int*)magImg_big.data)[row_index * width + col_index];
					unsigned short int template_pix = ((short int*)magImg_small.data)[p * sub_width + q];

					total_diff += abs(bigImg_pix - template_pix);
				}
			}
			((int*)searchImg.data)[i * width + j] = total_diff;
		}
	}
	int min = ((int*)searchImg.data)[0];
	for (unsigned short int i = 0; i <= testlength; i++)
		for (unsigned short int j = 0; j <= testwidth; j++)
		{
			if (((int*)searchImg.data)[i * width + j] <= min)
			{
				min = ((int*)searchImg.data)[i * width + j];
				*x = j; *y = i;
			}
		}
	return SUB_IMAGE_MATCH_OK;

}



int ustc_SubImgMatch_hist(Mat grayImg, Mat subImg, int* x, int* y)

{
	if (NULL == grayImg.data || NULL == subImg.data || NULL == x || NULL == y)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if ((1 != grayImg.channels()) || (1 != subImg.channels()))
	{
		cout << "It is not a gray image ." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (grayImg.rows < subImg.rows || grayImg.cols < subImg.cols)
	{
		cout << "The match is illegal!." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	unsigned short int width = grayImg.cols;
	unsigned short int height = grayImg.rows;
	unsigned short int sub_width = subImg.cols;
	unsigned short int sub_height = subImg.rows;
	unsigned short int histsmall[256] = { 0 }, *p = histsmall;
	unsigned short int histbig[256], *q = histbig;
	for (unsigned short int i = 0; i< sub_height; i++)
	{
		for (unsigned short int j = 0; j < sub_width; j++)
		{
			unsigned short int pixVal = subImg.data[i*sub_width + j];
			histsmall[pixVal]++;
		}
	}
	Mat searchImg(height, width, CV_32SC1, Scalar(INT_MAX));
	unsigned short int testheight = height - sub_height;
	unsigned short int testwidth = width - sub_width;
	for (unsigned short int i = 0; i <= testheight; i++)
	{
		for (unsigned short int j = 0; j <= testwidth; j++)
		{
			int total_diff = 0;
			for (unsigned short int m = 0; m < 256; m++)
			{
				*(q++) = 0;
			}
			q = histbig;
			for (unsigned short int m = 0; m< sub_height; m++)
				for (unsigned short int n = 0; n < sub_width; n++)
				{
					unsigned short int pixVal = grayImg.data[(i + m)*width + j + n];
					(*(q + pixVal))++;
				}
			for (unsigned short int m = 0; m<256; m++)
			{
				total_diff += abs(*(p++) - *(q++));
			}
			p = histsmall;
			q = histbig;
			((int*)searchImg.data)[i * width + j] = total_diff;
		}
	}
	int min = ((int*)searchImg.data)[0];
	for (unsigned short int i = 0; i <= testheight; i++)
		for (unsigned short int j = 0; j <= testwidth; j++)
		{
			if (((int*)searchImg.data)[i*width + j] <= min)
			{
				min = ((int*)searchImg.data)[i*width + j];
				*x = j, *y = i;
			}
		}
	return SUB_IMAGE_MATCH_OK;

}
