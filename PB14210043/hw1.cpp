#include "SubImageMatch.h"
float Arctan(float x, float y)
{
	float ax = abs(x), ay = abs(y);
	float a = min(ax, ay) / (max(ax, ay) + (float)DBL_EPSILON);
	float s = a*a;
	float r = ((-0.0464964749 * s + 0.15931422) * s - 0.327622764) * s * a + a;
	if (ay > ax) r = 1.57079637 - r;
	if (x < 0) r = 3.14159274f - r;
	if (y < 0) r = 6.28318548f - r;
	return r;
}



int ustc_ConvertBgr2Gray(Mat bgrImg, Mat& grayImg)
{
	if (NULL == bgrImg.data)
	{
		cout << "image input failed" << '\n';
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = bgrImg.cols;
	int height = bgrImg.rows;
	int k1, k2, k3, t;
	for (int row_i = 0; row_i < height; row_i++)
	{
		k1 = row_i*width;
		for (int col_j = 0; col_j < width; col_j++)
		{
			k2 = k1 + col_j;
			k3 = k2 * 3;
			t = bgrImg.data[k3] * 114 + bgrImg.data[k3 + 1] * 587 + bgrImg.data[k3 + 2] * 299;
			grayImg.data[k2] = t / 1000;
		}
	}
	return SUB_IMAGE_MATCH_OK;
}


int ustc_CalcGrad(Mat grayImg, Mat& gradImg_x, Mat& gradImg_y)
{
	if (NULL == grayImg.data)
	{
		cout << "image input failed" << '\n';
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	int k[9], k0, grad;
	for (int row_i = 1; row_i < height - 1; row_i++)
	{
		k0 = row_i*width;
		for (int col_j = 1;col_j < width - 1;col_j++)
		{
			k[0] = k0 + col_j;
			k[1] = k[0] - 1 - width;
			k[2] = k[0] - 1;
			k[3] = k[2] + width;
			k[4] = k[1] + 2;
			k[5] = k[2] + 2;
			k[6] = k[3] + 2;
			k[7] = k[1] + 1;
			k[8] = k[3] + 1;
			grad = grayImg.data[k[4]] + grayImg.data[k[5]] + grayImg.data[k[5]] + grayImg.data[k[6]] - grayImg.data[k[1]] - grayImg.data[k[2]] - grayImg.data[k[2]] - grayImg.data[k[3]];
			((float *)gradImg_x.data)[k[0]] = grad;
			grad = grayImg.data[k[3]] + grayImg.data[k[8]] + grayImg.data[k[8]] + grayImg.data[k[6]] - grayImg.data[k[1]] - grayImg.data[k[7]] - grayImg.data[k[7]] - grayImg.data[k[4]];
			((float *)gradImg_y.data)[k[0]] = grad;
		}
	}
	k0 = width*(height - 1);
	for (int col_j = 0; col_j < width;col_j++)
	{
		((float *)gradImg_x.data)[col_j] = 0;
		((float *)gradImg_y.data)[col_j] = 0;
		((float *)gradImg_x.data)[k0 + col_j] = 0;
		((float *)gradImg_y.data)[k0 + col_j] = 0;
	}
	k0 = width;
	for (int row_i = 0;row_i < height;row_i++)
	{
		((float *)gradImg_x.data)[k0] = 0;
		((float *)gradImg_y.data)[k0] = 0;
		((float *)gradImg_x.data)[k0 - 1] = 0;
		((float *)gradImg_y.data)[k0 - 1] = 0;
		k0 += width;
	}
	return SUB_IMAGE_MATCH_OK;
}

int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y, Mat& angleImg, Mat& magImg)
{
	if (NULL == gradImg_x.data || NULL == gradImg_y.data)
	{
		cout << "image input failed" << '\n';
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = gradImg_x.cols;
	int height = gradImg_y.rows;
	float grad_x, grad_y, angle;
	int k1, k2;
	for (int row_i = 1; row_i < height - 1; row_i++)
	{
		k1 = row_i*width;
		for (int col_j = 1; col_j < width - 1; col_j += 1)
		{
			k2 = k1 + col_j;
			grad_x = ((float*)gradImg_x.data)[k2];
			grad_y = ((float*)gradImg_y.data)[k2];
			angle = Arctan(grad_y, grad_x);
			((float*)angleImg.data)[k2] = angle * 180 / 3.1415926525f;
			((float*)magImg.data)[k2] = sqrtf(grad_x*grad_x + grad_y*grad_y);
		}
	}

#ifdef IMG_SHOW
	Mat angleImg_8U(height, width, CV_8UC1);
	//Ϊ�˷���۲죬����Щ���仯
	for (int row_i = 0; row_i < height; row_i++)
	{
		for (int col_j = 0; col_j < width; col_j += 1)
		{
			float angle = ((float*)angleImg.data)[row_i * width + col_j];
			//Ϊ������8U����ʾ����С��0-180֮��
			angle /= 2;
			angleImg_8U.data[row_i * width + col_j] = angle;
		}
	}

	namedWindow("gradImg_x_8U", 0);
	imshow("gradImg_x_8U", angleImg_8U);
	waitKey();
#endif
	return SUB_IMAGE_MATCH_OK;
}

int ustc_CalcMag(Mat gradImg_x, Mat gradImg_y, Mat&magImg)
{
	float*pX = (float*)gradImg_x.data;
	float*pY = (float*)gradImg_y.data;
	uchar*pMag = magImg.data;
	while (pMag < magImg.dataend)
	{
		*pMag = sqrtf((*pX) * (*pX) + (*pY) * (*pY)) / 4 / 1.41421356;
		pX++, pY++, pMag++;
	}
	return 0;
}

int ustc_CalcAngle(Mat gradImg_x, Mat gradImg_y, Mat&angleImg)
{
	float*pX = (float*)gradImg_x.data;
	float*pY = (float*)gradImg_y.data;
	uchar*pAngle = angleImg.data;
	while (pAngle < angleImg.dataend)
	{
		*pAngle = Arctan(*pX, *pY) * 255.0 / 2 / 3.1415927;
		pX++, pY++, pAngle++;
	}
	return 0;
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
	int conv_table[256];
	for (int i = 0;i < th;i++)
	{
		conv_table[i] = 0;
	}
	for (int i = th; i < 256;i++)
	{
		conv_table[i] = 255;
	}

	int temp0, temp1, pixVal;
	for (int row_i = 0; row_i < height; row_i++)
	{
		temp0 = row_i * width;
		for (int col_j = 0; col_j < width; col_j++)
		{
			temp1 = temp0 + col_j;
			pixVal = grayImg.data[temp1];
			binaryImg.data[temp1] = conv_table[pixVal];
		}
	}

#ifdef IMG_SHOW
	namedWindow("grayImg", 0);
	namedWindow("binaryImg", 0);
	imshow("binaryImg", binaryImg);
	imshow("grayImg", grayImg);
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
	int histogram[256] = { 0 }, k1, k2;
	for (int row_i = 0; row_i < height; row_i++)
	{
		k1 = row_i*width;
		for (int col_j = 0; col_j < width; col_j++)
		{
			k2 = k1 + col_j;
			histogram[grayImg.data[k2]]++;
		}
	}
	for (int i = 0;i < hist_len;i++)
	{
		hist[i] = 0;
	}
	float k0 = 1.0f*hist_len / 256;
	for (int i = 0;i < 256;i++)
	{
		hist[(int)(k0*(i + 1) - 0.5f)] += histogram[i];
	}
}
int ustc_SubImgMatch_gray(Mat grayImg, Mat subImg, int* x, int* y)
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
	int min_dis = 0;
	int k0, k1, k2, bigImg_pix, template_pix, tmp;
	int dis;
	min_dis = 0x7fffffff;
	*x = 0;
	*y = 0;

	for (int big_row = 0; big_row < height - sub_height; big_row++)
	{
		for (int big_col = 0;big_col < width - sub_width;big_col++)
		{
			dis = 0;
			for (int sub_row = 0; sub_row < sub_height;sub_row++)
			{
				k0 = (big_row + sub_row)*width + big_col;
				k1 = sub_row*sub_width;
				for (int sub_col = 0;sub_col < sub_width;sub_col++)
				{
					bigImg_pix = grayImg.data[k0 + sub_col];
					template_pix = subImg.data[k1 + sub_col];
					tmp = bigImg_pix - template_pix;
					dis += (tmp>0) ? tmp : -tmp;
				}
			}
			if (dis<min_dis)
			{
				min_dis = dis;
				*x = big_row;
				*y = big_col;
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

	int width = colorImg.cols;
	int height = colorImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int min_dis = 0;
	int k0, k1, k2, k3, bigImg_pix_b, bigImg_pix_g, bigImg_pix_r, sub_r, sub_g, sub_b, tmp_r, tmp_g, tmp_b;
	min_dis = 0x7fffffff;
	int dis;

	*x = 0;
	*y = 0;

	for (int big_row = 0; big_row < height - sub_height; big_row++)
	{
		for (int big_col = 0;big_col < width - sub_width;big_col++)
		{
			dis = 0;
			for (int sub_row = 0; sub_row < sub_height;sub_row++)
			{
				k0 = ((big_row + sub_row)*width + big_col) * 3;
				k1 = sub_row*sub_width * 3;
				for (int sub_col = 0;sub_col < sub_width;sub_col++)
				{
					bigImg_pix_b = colorImg.data[k0];
					bigImg_pix_g = colorImg.data[k0 + 1];
					bigImg_pix_r = colorImg.data[k0 + 2];
					sub_b = subImg.data[k1];
					sub_g = subImg.data[k1 + 1];
					sub_r = subImg.data[k1 + 2];
					k1 += 3;
					k0 += 3;

					tmp_b = bigImg_pix_b - sub_b;
					tmp_g = bigImg_pix_g - sub_g;
					tmp_r = bigImg_pix_r - sub_r;
					dis += ((tmp_b>0) ? tmp_b : -tmp_b) + ((tmp_g>0) ? tmp_g : -tmp_g) + ((tmp_r>0) ? tmp_r : -tmp_r);
				}
			}
			if (dis<min_dis)
			{
				min_dis = dis;
				*x = big_row;
				*y = big_col;
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
	int sigma1, sigma2;
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int k0, k1, k2, bigImg_pix, template_pix, tmp;
	float dis, max_dis;
	max_dis = 0x7fffffff + 1;
	*x = 0;
	*y = 0;

	for (int big_row = 0; big_row < height - sub_height; big_row++)
	{
		for (int big_col = 0;big_col < width - sub_width;big_col++)
		{
			dis = 0;
			sigma1 = 0;
			sigma2 = 0;
			for (int sub_row = 0; sub_row < sub_height;sub_row++)
			{
				k0 = (big_row + sub_row)*width + big_col;
				k1 = sub_row*sub_width;
				for (int sub_col = 0;sub_col < sub_width;sub_col++)
				{
					bigImg_pix = grayImg.data[k0 + sub_col];
					template_pix = subImg.data[k1 + sub_col];
					sigma1 += bigImg_pix*template_pix;
					sigma2 += bigImg_pix*bigImg_pix;
				}
			}
			dis = 1.0f*sigma1 / sqrtf(sigma2);
			if (dis>max_dis)
			{
				max_dis = dis;
				*x = big_row;
				*y = big_col;
			}
		}
	}
	return SUB_IMAGE_MATCH_OK;
}


int ustc_SubImgMatch_angle(Mat grayImg, Mat subImg, int* x, int* y)
{
	int width = grayImg.cols, height = grayImg.rows;
	int sub_width = subImg.cols, sub_height = subImg.rows;
	Mat gradImg_x(height, width, CV_32SC1);
	Mat gradImg_y(height, width, CV_32SC1);
	Mat angleImg(height, width, CV_8UC1);
	Mat gradSub_x(sub_height, sub_width, CV_32SC1);
	Mat gradSub_y(sub_height, sub_width, CV_32SC1);
	Mat Sub_angle(sub_height, sub_width, CV_8UC1);
	ustc_CalcGrad(grayImg, gradImg_x, gradImg_y);
	ustc_CalcGrad(subImg, gradSub_x, gradSub_y);
	ustc_CalcAngle(gradImg_x, gradImg_y, angleImg);
	ustc_CalcAngle(gradSub_x, gradSub_y, Sub_angle);
	int min_dis = 0x7fffffff;
	int abs_angle[256] = { 0 };
	for (int i = 0;i < 129;i++)
	{
		abs_angle[i] = i;
	}
	for (int i = 129;i < 256;i++)
	{
		abs_angle[i] = 256 - i;
	}
	int a = subImg.rows - 1, b = subImg.cols - 1;
	for (int i = 0; i <= grayImg.rows - subImg.rows; i++)
	{
		for (int j = 0; j <= grayImg.cols - subImg.cols; j++)
		{
			int d = 0;
			for (int k = 1; k < a; k++)
			{
				uchar*p = grayImg.data + (i + k) * grayImg.cols + j;
				uchar*q = subImg.data + k * subImg.cols;
				for (int l = 1; l < b; l++)
				{
					d += abs_angle[uchar(p[l] - q[l])];
				}
			}
			if (d < min_dis)
			{
				min_dis = d;
				*x = i;
				*y = j;
			}
		}
	}
	return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_mag(Mat grayImg, Mat subImg, int* x, int* y)
{
	int width = grayImg.cols, height = grayImg.rows;
	int sub_width = subImg.cols, sub_height = subImg.rows;
	Mat gradImg_x(height, width, CV_32SC1);
	Mat gradImg_y(height, width, CV_32SC1);
	Mat magImg(height, width, CV_8UC1);
	Mat gradSub_x(sub_height, sub_width, CV_32SC1);
	Mat gradSub_y(sub_height, sub_width, CV_32SC1);
	Mat Sub_mag(sub_height, sub_width, CV_8UC1);

	ustc_CalcGrad(grayImg, gradImg_x, gradImg_y);
	ustc_CalcGrad(subImg, gradSub_x, gradSub_y);
	ustc_CalcMag(gradImg_x, gradImg_y, magImg);
	ustc_CalcMag(gradSub_x, gradSub_y, Sub_mag);

	int min_dis = 0x7fffffff;
	for (int i = 0; i <= grayImg.rows - subImg.rows; i++)
	{
		for (int j = 0; j <= grayImg.cols - subImg.cols; j++)
		{
			int d = 0;
			int t;
			int a = subImg.rows - 1, b = subImg.cols - 1;
			for (int k = 1; k < a; k++)
			{
				uchar*p = grayImg.data + (i + k) * grayImg.cols + j;
				uchar*q = subImg.data + k * subImg.cols;
				for (int l = 1; l < b; l++)
				{
					t = p[l] - q[l];
					d += (t > 0) ? t : -t;
				}
			}
			if (d < min_dis)
			{
				min_dis = d;
				*x = i;
				*y = j;
			}
		}
	}
	return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_hist(Mat grayImg, Mat subImg, int* x, int* y)
{
	int width = grayImg.cols, height = grayImg.rows;
	int sub_width = subImg.cols, sub_height = subImg.rows;
	int hist[256] = { 0 }, sub_hist[256] = { 0 };
	for (uchar* p = subImg.data; p < subImg.dataend; p++)
	{
		sub_hist[*p]++;
	}
	int dis, min_dis, t;
	min_dis = 0x7fffffff;
	*x = 0;
	*y = 0;
	uchar *k0;
	int length = sizeof(int) * 256;
	for (int row_i = 0; row_i < height - sub_height; row_i++)
	{
		for (int col_j = 0; col_j < width - sub_width; col_j++)
		{
			memset(hist, 0, length);
			dis = 0;
			for (int row_2 = 0;row_2 < sub_height;row_2++)
			{
				k0 = grayImg.data + (row_i + row_2)*width + col_j;
				for (int col_2 = 0;col_2 < sub_width; col_2++)
				{
					hist[*(k0 + col_2)]++;
				}
			}

			for (int i = 0;i < 256;i++)
			{
				t = hist[i] - sub_hist[i];
				dis += (t > 0) ? t : -t;
			}

			if (dis<min_dis)
			{
				min_dis = dis;
				*x = row_i;
				*y = col_j;
			}
		}
	}
	return SUB_IMAGE_MATCH_OK;
}
