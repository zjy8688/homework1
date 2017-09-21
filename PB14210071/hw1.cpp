
#include "SubImageMatch.h"

#define IMG_SHOW
using namespace cv;
int ustc_ConvertBgr2Gray(Mat bgrImg, Mat& grayImg)
{
	Mat raw_img = bgrImg;

	if (NULL == raw_img.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	Mat gray = grayImg;
	int len = raw_img.rows*raw_img.cols;
	uchar*p = raw_img.data;
	uchar*q = gray.data;
	for (int i = 0; i < len; i++)
	{
		*(q) = *(p)*0.114 + *(p + 1)*0.587 + *(p + 2)*0.229;
		q++;
		p = p + 3;
	}
#ifdef IMG_SHOW
	namedWindow("grayImg", 0);
	imshow("grayImg", grayImg);
#endif
	return SUB_IMAGE_MATCH_OK;
}

int ustc_CalcGrad(Mat grayImg, Mat& gradImg_x, Mat& gradImg_y)
{
	Mat raw_img = grayImg;
	if (NULL == raw_img.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	uchar* raw_img1 = raw_img.data;
	int row = raw_img.rows;
	int col = raw_img.cols;
	int a00, a01, a02;
	int a10, a11, a12;
	int a20, a21, a22;
	for (int i = 1; i < row - 1; i++)
	{
		for (int j = 1; j < col - 1; j++)
		{
			a00 = raw_img1[(i - 1)*col + j - 1];
			a01 = raw_img1[(i - 1)*col + j];
			a02 = raw_img1[(i - 1)*col + j + 1];
			a10 = raw_img1[i*col + j - 1];
			a11 = raw_img1[i*col + j];
			a12 = raw_img1[i*col + j + 1];
			a20 = raw_img1[(i + 1)*col + j - 1];
			a21 = raw_img1[(i + 1)*col + j];
			a22 = raw_img1[(i + 1)*col + j + 1];
			((float*)gradImg_x.data)[i*col + j] = (a02 - a00 + 2 * (a12 - a10) + a22 - a20);
			((float*)gradImg_y.data)[i*col + j] = (a20 - a00 + 2 * (a21 - a01) + a22 - a02);
		}
	}

#ifdef IMG_SHOW
	Mat show_imgx(row, col, CV_8UC1);
	Mat show_imgy(row, col, CV_8UC1);
	int t1, t2;
	for (int row_i = 0; row_i < row; row_i++)
	{
		t1 = row_i*col;
		for (int col_j = 0; col_j < col; col_j++)
		{
			t2 = t1 + col_j;
			int valx = ((float*)gradImg_x.data)[t2];
			int valy = ((float*)gradImg_y.data)[t2];
			show_imgx.data[t2] = valx;
			show_imgy.data[t2] = valy;
		}
	}
	namedWindow("gradimg_x", 0);
	imshow("gradimg_x", show_imgx);
	cvNamedWindow("gradimg_y", 0);
	imshow("gradimg_y", show_imgy);
#endif // IMG_SHOW

	return SUB_IMAGE_MATCH_OK;

}
int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y, Mat& angleImg, Mat& magImg)
{
	if (NULL == gradImg_x.data || NULL == gradImg_y.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int row = gradImg_x.rows;
	int col = gradImg_x.cols;
	int t1, t2;
	float dx;
	float dy;
	for (int row_i = 1; row_i < row - 1; row_i++)
	{
		t1 = row_i*col;
		for (int col_j = 1; col_j < col - 1; col_j++)
		{
			t2 = t1 + col_j;
			dy = ((float*)(gradImg_y.data))[t2];
			dx = ((float*)(gradImg_x.data))[t2];
			((float*)angleImg.data)[t2] = (float)atan2(dy, dx);
			((float*)magImg.data)[t2] = (float)sqrt(dx*dx+dy*dy);
		}
	}
	for (int row_i = 0; row_i < row; row_i++)
	{
		t1 = row_i*col;
		for (int col_j = 0; col_j < col; col_j++)
		{
			t2 = t1 + col_j;
			float val_angle = ((float*)angleImg.data)[t2];
			val_angle *= 180 / CV_PI;
			val_angle += 180;
			val_angle /= 2;
			((float*)angleImg.data)[t2] = val_angle;
		}
	}
#ifdef IMG_SHOW
	Mat angleImg_8U(row, col, CV_8UC1);
	Mat magImg_8U(row, col, CV_8UC1);
	for (int row_i = 0; row_i < row; row_i++)
	{
		t1 = row_i*col;
		for (int col_j = 0; col_j < col; col_j++)
		{
			t2 = t1 + col_j;
			int val_angle_8U = ((float*)angleImg.data)[t2];
			int val_mag = ((float*)magImg.data)[t2];
			angleImg_8U.data[t2] = val_angle_8U;
			magImg_8U.data[t2] = val_mag;
		}
	}
	namedWindow("angleImg", 0);
	imshow("angleImg", angleImg_8U);
	namedWindow("magImg", 0);
	imshow("magImg", magImg_8U);
#endif // IMG_SHOW

	return SUB_IMAGE_MATCH_OK;
}
int ustc_Threshold(Mat grayImg, Mat& binaryImg, int th)
{
	if (NULL == grayImg.data)
	{
		cout << "image is NULL" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	Mat raw_img = grayImg;
	int row = raw_img.rows;
	int col = raw_img.cols;
	int t1, t2;
	int c;
	for (int i = 0; i < row; i++)
	{
		t1 = i*col;
		for (int j = 0; j < col; j++)
		{
			t2 = t1 + j;
			c = (raw_img.data[t2] - th) >> 63;
			binaryImg.data[t2] = 255 * (c + 1);
		}
	}
#ifdef IMG_SHOW
	namedWindow("binaryImg", 0);
	imshow("binaryImg", binaryImg);
#endif // 
	return SUB_IMAGE_MATCH_OK;
}
int ustc_CalcHist(Mat grayImg, int* hist, int hist_len)
{
	if (NULL == grayImg.data)
	{
		cout << "image is NULL" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	Mat raw_img = grayImg;
	uchar* raw_img1 = grayImg.data;
	int hen = raw_img.rows*raw_img.cols;
	for (int i; i < hist_len; i++)
	{
		hist[i] = 0;
	}
	for (int i; i < hen; i++)
	{
		hist[*raw_img1]++;
		raw_img1++;
	}
	return SUB_IMAGE_MATCH_OK;
}
int ustc_SubImgMatch_gray(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data)
	{
		cout << "image is NULL" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int row = grayImg.rows;
	int col = grayImg.cols;
	int sub_row = subImg.rows;
	int sub_col = subImg.cols;
	if ((sizeof(grayImg.data) / (row*col)) != (sizeof(subImg.data) / (sub_row*sub_col)))
	{
		cout << "the channels number are different" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	float differece;
	float min;
	int t1, t2, t3, t4;
	int T1, T2;
	Mat match_difference(row - sub_row, col - sub_col, CV_32FC1);
	for (int row_i = 0; row_i < row - sub_row; row_i++)
	{
		t1 = row_i*col;
		for (int col_j = 0; col_j < col - sub_col; col_j++)
		{
			t2 = t1 + col_j;
			differece = 0;
			for (int subrow_i = 0; subrow_i < sub_row; subrow_i++)
			{
				T1 = subrow_i*sub_col;
				t3 = subrow_i*col + t2;
				for (int subcol_j = 0; subcol_j < sub_col; subcol_j++)
				{
					T2 = T1 + subcol_j;
					t4 = t3 + subcol_j;
					differece += abs(grayImg.data[t4] - subImg.data[T2]);
				}
			}
			((float*)match_difference.data)[t2-row_i*sub_col] = differece;
		}
	}
	min = ((float*)match_difference.data)[0];
	*x = 0;
	*y = 0;
	row = row - sub_row;
	col = col - sub_col;
	for (int row_i=0; row_i < row; row_i++)
	{
		t1 = row_i*col;
		for (int col_j=0; col_j < col; col_j++)
		{
			t2 = t1 + col_j;
			differece = ((float*)match_difference.data)[t2];
			if (min < differece)
			{
				
			}
			else
			{
				*x = row_i;
				*y = col_j;
				min = differece;
			}
		}
	}
#ifdef IMG_SHOW
	cout << "x=" << *x << endl;
	cout << "y=" << *y << endl;
#endif // IMG_SHOW
	return SUB_IMAGE_MATCH_OK;
}
int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int* x, int* y)
{
	if (NULL == colorImg.data)
	{
		cout << "image is NULL" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int row = colorImg.rows;
	int col = colorImg.cols;
	int sub_row = subImg.rows;
	int sub_col = subImg.cols;
	if ((sizeof(colorImg.data) / (row*col)) != (sizeof(subImg.data) / (sub_row*sub_col)))
	{
		cout << "the channels are different" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int t1, t2, t3, t4;
	int T1, T2;
	float differece;
	float min;
	Mat match_difference(row - sub_row, col - sub_col, CV_32FC1);
	for (int row_i = 0; row_i < row - sub_row; row_i++)
	{
		t1 = row_i*col * 3;
		for (int col_j = 0; col_j < col - sub_col; col_j++)
		{
			t2 = t1 + col_j * 3;
			differece = 0;
			for (int subrow_i = 0; subrow_i < sub_row; subrow_i++)
			{
				T1 = subrow_i*sub_col*3;
				t3 = t2 + subrow_i*col * 3;
				for (int subcol_j = 0; subcol_j < sub_col; subcol_j++)
				{
					T2 = T1 + subcol_j * 3;
					t4 = t3 + subcol_j * 3;
					differece += abs(colorImg.data[t4] - subImg.data[T2]) + abs(colorImg.data[t4 + 1] - subImg.data[T2 + 1]) + abs(colorImg.data[t4 + 2] - subImg.data[T2 + 2]);
				}
			}
			((float*)match_difference.data)[row_i*(col-sub_col) + col_j] = differece;
		}
	}
	min = ((float*)match_difference.data)[0];
	*x = 0;
	*y = 0;
	row = row - sub_row;
	col = col - sub_col;
	for (int row_i=0; row_i < row; row_i++)
	{
		t1 = row_i*col;
		for (int col_j=0; col_j < col; col_j++)
		{
			t2 = t1 + col_j;
			differece = ((float*)match_difference.data)[t2];
			if (min < differece)
			{

			}
			else
			{
				*x = row_i;
				*y = col_j;
				min = differece;
			}
		}
	}
#ifdef IMG_SHOW
	cout << "x=" << *x << endl;
	cout << "y=" << *y << endl;
#endif // IMG_SHOW

	return SUB_IMAGE_MATCH_OK;
}
int ustc_SubImgMatch_corr(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data)
	{
		cout << "image is NULL" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int row = grayImg.rows;
	int col = grayImg.cols;
	int sub_row = subImg.rows;
	int sub_col = subImg.cols;
	if ((sizeof(grayImg.data) / (row*col)) != (sizeof(subImg.data) / (sub_row*sub_col)))
	{
		cout << "the channel are different" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int temp;
	int T1, T2;
	double T_sqar_root=0;
	for (int subrow_i=0; subrow_i < sub_row; subrow_i++)
	{
		T1 = subrow_i*sub_col;
		for (int subcol_j=0; subcol_j < sub_col; subcol_j++)
		{
			T2 = T1 + subcol_j;
			temp = subImg.data[T2];
			T_sqar_root += temp*temp;
		}
	}
	T_sqar_root = sqrt(T_sqar_root);
	int t1, t2, t3, t4;
	int temp1, temp2;
	double S_sqrt_root=0;
	double S_T=0;
	double R;
	Mat match_difference(row - sub_row, col - sub_col, CV_64FC1);
	for (int row_i = 0; row_i < row - sub_row; row_i++)
	{
		t1 = row_i*col;
		for (int col_j = 0; col_j < col - sub_col; col_j++)
		{
			t2 = t1 + col_j;
			S_T = 0;
			S_sqrt_root = 0;
			for (int subrow_i = 0; subrow_i < sub_row; subrow_i++)
			{
				t3 = t2 + subrow_i*col;
				T1 = subrow_i*sub_col;
				for (int subcol_j = 0; subcol_j < sub_col; subcol_j++)
				{
					t4 = t3 + subcol_j;
					T2 = T1 + subcol_j;
					temp1 = grayImg.data[t4];
					temp2 = subImg.data[T2];
					S_T += temp1*temp2;
					S_sqrt_root += temp1*temp1;
				}
			}
			S_sqrt_root = sqrt(S_sqrt_root);
			R = S_T / (S_sqrt_root*T_sqar_root);
			((double*)match_difference.data)[t2-row_i*sub_col] = R;
		}
	}
	double max =((double*)match_difference.data)[0];
	double difference;
	*x = 0; 
	*y = 0;
	row = row - sub_row;
	col = col - sub_col;
	for (int row_i = 0; row_i < row; row_i++)
	{
		t1 = row_i*col;
		for (int col_j = 0; col_j < col; col_j++)
		{
			t2 = t1 + col_j;
			difference = ((double*)match_difference.data)[t2];
			if (max > difference)
			{
				
			}
			else
			{
				*x = row_i;
				*y = col_j;
				max = difference;
			}
		}
	}
#ifdef IMG_SHOW
	cout << "x=" << *x << endl;
	cout << "y=" << *y << endl;
#endif // IMG_SHOW

	return SUB_IMAGE_MATCH_OK;
}
int ustc_SubImgMatch_angle(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data)
	{
		cout << "image is NULL" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int row = grayImg.rows;
	int col = grayImg.cols;
	int sub_row = subImg.rows;
	int sub_col = subImg.cols;
	if (row < sub_row || col < sub_col)
	{
		cout << "subimg is larger than grayimg" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	Mat gradImg_x = Mat::zeros(row, col, CV_32FC1);
	Mat gradImg_y = Mat::zeros(row, col, CV_32FC1);
	Mat sub_gradImg_x = Mat::zeros(sub_row, sub_col, CV_32FC1);
	Mat sub_gradImg_y = Mat::zeros(sub_row, sub_col, CV_32FC1);
	ustc_CalcGrad(grayImg, gradImg_x, gradImg_y);
	ustc_CalcGrad(subImg, sub_gradImg_x, sub_gradImg_y);
	Mat angleImg = Mat::zeros(row, col, CV_32FC1);
	Mat magImg = Mat::zeros(row, col, CV_32FC1);
	Mat sub_angleImg = Mat::zeros(sub_row, sub_col, CV_32FC1);
	Mat sub_magImg = Mat::zeros(sub_row, sub_col, CV_32FC1);
	ustc_CalcAngleMag(gradImg_x, gradImg_y, angleImg, magImg);
	ustc_CalcAngleMag(sub_gradImg_x, sub_gradImg_y, sub_angleImg, sub_magImg);
	int t1, t2, t3, t4;
	int T1, T2;
	float difference;
	Mat match_difference = Mat::zeros(row - sub_row, col - sub_col, CV_32FC1);
	for (int row_i = 0; row_i < row - sub_row ; row_i++)
	{
		t1 = row_i*col;
		for (int col_j = 0; col_j < col - sub_col ; col_j++)
		{
			t2 = t1 + col_j;
			difference = 0;
			for (int subrow_i = 1; subrow_i < sub_row - 1; subrow_i++)
			{
				t3 = t2 + subrow_i*col;
				T1 = subrow_i*sub_col;
				for (int subcol_j = 1; subcol_j < sub_col - 1; subcol_j++)
				{
					t4 = t3 + subcol_j;
					T2 = T1 + subcol_j;
					difference += fabsf(((float*)angleImg.data)[t4] - ((float*)sub_angleImg.data)[T2]);
				}
			}
			((float*)match_difference.data)[t2-row_i*sub_col] = (float)difference;
		}
	}
	row = row - sub_row;
	col = col - sub_col;
	float min = ((float*)match_difference.data)[0];
	*x = 0;
	*y = 0;
	for (int row_i = 0; row_i < row ; row_i++)
	{
		t1 = row_i*col;
		for (int col_j = 0; col_j < col ; col_j++)
		{
			t2 = t1 + col_j;
			difference = ((float*)match_difference.data)[t2];
			if (min < difference)
			{

			}
			else
			{
				*x = row_i;
				*y = col_j;
				min = difference;
			}
		}
	}
#ifdef IMG_SHOW
	cout << "x=" << *x << endl;
	cout << "y=" << *y << endl;
#endif // IMG_SHOW

	return SUB_IMAGE_MATCH_OK;
}
int ustc_SubImgMatch_mag(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data)
	{
		cout << "image is NULL" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int row = grayImg.rows;
	int col = grayImg.cols;
	int sub_row = subImg.rows;
	int sub_col = subImg.cols;
	if (row < sub_row || col < sub_col)
	{
		cout << "subimg is larger than grayimg" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	Mat gradImg_x = Mat::zeros(row, col, CV_32FC1);
	Mat gradImg_y = Mat::zeros(row, col, CV_32FC1);
	Mat sub_gradImg_x = Mat::zeros(sub_row, sub_col, CV_32FC1);
	Mat sub_gradImg_y = Mat::zeros(sub_row, sub_col, CV_32FC1);
	ustc_CalcGrad(grayImg, gradImg_x, gradImg_y);
	ustc_CalcGrad(subImg, sub_gradImg_x, sub_gradImg_y);
	Mat angleImg = Mat::zeros(row, col, CV_32FC1);
	Mat magImg = Mat::zeros(row, col, CV_32FC1);
	Mat sub_angleImg = Mat::zeros(sub_row, sub_col, CV_32FC1);
	Mat sub_magImg = Mat::zeros(sub_row, sub_col, CV_32FC1);
	ustc_CalcAngleMag(gradImg_x, gradImg_y, angleImg, magImg);
	ustc_CalcAngleMag(sub_gradImg_x, sub_gradImg_y, sub_angleImg, sub_magImg);
	int t1, t2, t3, t4;
	int T1, T2;
	float difference;
	Mat match_difference = Mat::zeros(row - sub_row, col - sub_col, CV_32FC1);
	for (int row_i = 0; row_i < row - sub_row ; row_i++)
	{
		t1 = row_i*col;
		for (int col_j = 0; col_j < col - sub_col ; col_j++)
		{
			t2 = t1 + col_j;
			difference = 0;
			for (int subrow_i = 1; subrow_i < sub_row - 1; subrow_i++)
			{
				t3 = t2 + subrow_i*col;
				T1 = subrow_i*sub_col;
				for (int subcol_j = 1; subcol_j < sub_col - 1; subcol_j++)
				{
					t4 = t3 + subcol_j;
					T2 = T1 + subcol_j;
					(float)difference += fabsf(((float*)magImg.data)[t4] - ((float*)sub_magImg.data)[T2]);
				}
			}
			((float*)match_difference.data)[t2-row_i*sub_col] = (float)difference;
		}
	}
	row = row - sub_row;
	col = col - sub_col;
	float min = ((float*)match_difference.data)[0];
	*x = 0;
	*y = 0;
	for (int row_i = 0; row_i < row ; row_i++)
	{
		t1 = row_i*col;
		for (int col_j = 0; col_j < col ; col_j++)
		{
			t2 = t1 + col_j;
			(float)difference = ((float*)match_difference.data)[t2];
			if (min < difference)
			{

			}
			else
			{
				*x = row_i;
				*y = col_j;
				min = difference;
			}
		}
	}
#ifdef IMG_SHOW
	cout << "x=" << *x << endl;
	cout << "y=" << *y << endl;
#endif // IMG_SHOW

	return SUB_IMAGE_MATCH_OK;
}
int ustc_SubImgMatch_hist(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data)
	{
		cout << "image is NULL" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int row = grayImg.rows;
	int col = grayImg.cols;
	int sub_row = subImg.rows;
	int sub_col = subImg.cols;
	if (row < sub_row || col < sub_col)
	{
		cout << "subimg is larger than grayimg" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int hist_len = 256;
	int hist[256] = { 0 };
	int sub_hist[256] = { 0 };
	int t1, t2, t3, t4;
	int T1, T2;
	float difference;
	Mat match_difference(row - sub_row, col - sub_col, CV_32FC1);
	float min;
	for (int subrow_i = 0; subrow_i < sub_row; subrow_i++)
	{
		T1 = subrow_i*sub_col;
		for (int subcol_j = 0; subcol_j < sub_col; subcol_j++)
		{
			T2 = T1 + subcol_j;
			sub_hist[subImg.data[T2]]++;
		}
	}
	for (int row_i = 0; row_i < row-sub_row; row_i++)
	{
		t1 = row_i*col;
		for (int col_j = 0; col_j < col-sub_col; col_j++)
		{
			t2 = t1 + col_j;
			difference = 0;
			for (int hist_i = 0; hist_i < hist_len; hist_i++)
			{
				hist[hist_i] = 0;
			}
			for (int subrow_i = 0; subrow_i < sub_row; subrow_i++)
			{
				t3 = t2 + subrow_i*col;
				for (int subcol_j = 0; subcol_j < sub_col; subcol_j++)
				{
					t4 = t3 + subcol_j;
					int val = grayImg.data[t4];
					hist[val]++;
				}
			}
			for (int hist_i = 0; hist_i < hist_len; hist_i++)
			{
				difference += abs(hist[hist_i] - sub_hist[hist_i]);
			}
			((float*)match_difference.data)[t2-row_i*sub_col] = difference;
		}
	}
	row = row - sub_row;
	col = col - sub_col;
	min = ((float*)match_difference.data)[0];
	*x = 0;
	*y = 0;
	for (int row_i = 0; row_i < row; row_i++)
	{
		t1 = row_i*col;
		for (int col_j = 0; col_j < col; col_j++)
		{
			t2 = t1 + col_j;
			difference = ((float*)match_difference.data)[t2];
			if (min < difference)
			{

			}
			else
			{
				*x = row_i;
				*y = col_j;
				min = difference;
			}
		}
	}
#ifdef IMG_SHOW
	cout << "x=" << *x << endl;
	cout << "y=" << *y << endl;
#endif // IMG_SHOW

	return SUB_IMAGE_MATCH_OK;
}
