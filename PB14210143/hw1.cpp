#include "SubImageMatch.h"

float my_sqrt(float number)
{
	int i;
	float x2, y;
	const float threehalfs = 1.5f;

	x2 = number * 0.5f;
	y = number;
	i = *(int *)&y;
	i = 0x5f375a86 - (i >> 1);
	y = *(float *)&i;
	y = y * (threehalfs - (x2 * y * y));
	y = y * (threehalfs - (x2 * y * y));
	y = y * (threehalfs - (x2 * y * y));
	return number * y;
}

float my_atan(float dx, float dy)
{
	float ax = (dx > 0) ? dx : -dx;
	float ay = (dy > 0) ? dy : -dy;
	float min = (ax < ay) ? ax : ay;
	float max = (ax > ay) ? ax : ay;
	float a = min / (max + (float)DBL_EPSILON);
	float s = a * a;
	float r = ((-0.0464964749f * s + 0.15931422f) * s - 0.327622764f) * s * a + a;
	if (ay > ax)
	{
		r = 1.57079637f - r;
	}
	if (dx < 0)
	{
		r = 3.14159274f - r;
	}
	if (dy < 0)
	{
		r = 6.28318548f - r;
	}
	return r / CV_PI * 180.0f;
}

int ustc_ConvertBgr2Gray(Mat bgrImg, Mat& grayImg)
{
	if (NULL == bgrImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	uint nRows = bgrImg.rows;
	uint nCols = bgrImg.cols;
	uchar *data = (uchar *)bgrImg.data;
	uchar *gray_data = (uchar *)grayImg.data;

	for (uint i = 0; i < nRows; i++)
	{
		for (uint j = 0; j < nCols; j++)
		{
			gray_data[i*nCols + j] = (114 * data[3 * (i * nCols + j) + 0] + \
				587 * data[3 * (i * nCols + j) + 1] + \
				299 * data[3 * (i * nCols + j) + 2]) >> 10;
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

	uchar *data = (uchar *)grayImg.data;
	float *data_x = (float *)gradImg_x.data;
	float *data_y = (float *)gradImg_y.data;
	uint nRows = grayImg.rows;
	uint nCols = grayImg.cols;

	for (uint i = 1; i < nRows - 1; i++)
	{
		for (uint j = 1; j < nCols - 1; j++)
		{
			uint left_top = (i - 1) * nCols + (j - 1);
			uint left_mid = left_top + nCols;
			uint left_down = left_mid + nCols;
			uint index = i * nCols + j;

			data_x[index] = (-data[left_top]       /*+ 0 * data[(left_top + 1)]*/ + data[(left_top + 2)]     \
				- data[left_mid] * 2   /*+ 0 * data[(left_mid + 1)]*/ + data[(left_mid + 2)] * 2 \
				- data[left_down]      /*+ 0 * data[(left_down + 1)]*/ + data[(left_down + 2)]);

			data_y[index] = (-data[left_top] - data[(left_top + 1)] * 2 - data[(left_top + 2)] \
				/* 0 * data[left_mid] + 0 * data[(left_mid + 1)] + 0 * data[(left_mid + 2)] */
				+ data[left_down] + data[(left_down + 1)] * 2 + data[(left_down + 2)]);
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

	float *data_x = (float *)gradImg_x.data;
	float *data_y = (float *)gradImg_y.data;
	float *data_angle = (float *)angleImg.data;
	float *data_mag = (float *)magImg.data;
	uint nRows = gradImg_x.rows;
	uint nCols = gradImg_x.cols;

	for (uint i = 1; i < nRows - 1; i++)
	{
		for (uint j = 1; j < nCols - 1; j++)
		{
			uint index = i * nCols + j;
			float dx = data_x[index];
			float dy = data_y[index];
			data_angle[index] = my_atan(dx, dy);
			data_mag[index] = my_sqrt(dx*dx + dy*dy);
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

	uchar *data = (uchar *)grayImg.data;
	uchar *data_bi = (uchar *)binaryImg.data;
	uint nRows = grayImg.rows;
	uint nCols = grayImg.cols;

	for (uint i = 0; i < nRows; i++)
	{
		for (uint j = 0; j < nCols; j++)
		{
			uint index = i * nCols + j;
			int val = data[index];
			val = -((th - val) >> 31) * 255;
			data_bi[index] = val;
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

	uchar *data = (uchar *)grayImg.data;
	int nRows = grayImg.rows;
	int nCols = grayImg.cols;

	for (int i = 0; i < hist_len; i++)
	{
		hist[i] = 0;
	}

	for (int i = 0; i < nRows; i++)
	{
		for (int j = 0; j < nCols; j++)
		{
			int pix = data[(i * nCols + j)];
			hist[pix]++;
		}
	}
}

int ustc_SubImgMatch_gray(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	*x = *y = -1;

	uchar *data = (uchar *)grayImg.data;
	uchar *data_sub = (uchar *)subImg.data;
	int nCols = grayImg.cols;
	int nRows = grayImg.rows;
	int sub_nCols = subImg.cols;
	int sub_nRows = subImg.rows;

	if (sub_nCols > nCols || sub_nRows > nRows)
	{
		cout << "subImg is too big." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	if (sub_nCols == nCols && sub_nRows == nRows)
	{
		*x = 0;
		*y = 0;
		return SUB_IMAGE_MATCH_OK;
	}

	int min_err = 0x7FFFFFFF;

	for (int i = 0; i < nRows - sub_nRows; i++)
	{
		for (int j = 0; j < nCols - sub_nCols; j++)
		{
			int total_err = 0;
			for (int sub_i = 0; sub_i < sub_nRows; sub_i++)
			{
				for (int sub_j = 0; sub_j < sub_nCols; sub_j++)
				{
					int err = data[(i + sub_i)*nCols + (j + sub_j)] - data_sub[sub_i*sub_nCols + sub_j];
					total_err += (err & 0x80000000) ? -err : err;
				}
			}

			if (min_err > total_err)
			{
				min_err = total_err;
				*x = i;
				*y = j;
			}
		}
	}

	if (*x == -1 == *y) {
		cout << "match failed." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	else
	{
		return SUB_IMAGE_MATCH_OK;
	}
}

int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int* x, int* y)
{
	if (NULL == colorImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	*x = *y = -1;

	uchar *data = (uchar *)colorImg.data;
	uchar *data_sub = (uchar *)subImg.data;
	int nCols = colorImg.cols;
	int nRows = colorImg.rows;
	int sub_nCols = subImg.cols;
	int sub_nRows = subImg.rows;

	if (sub_nCols > nCols || sub_nRows > nRows)
	{
		cout << "subImg is too big." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	if (sub_nCols == nCols && sub_nRows == nRows)
	{
		*x = 0;
		*y = 0;
		return SUB_IMAGE_MATCH_OK;
	}

	int min_err = 0x7FFFFFFF;

	for (int i = 0; i < nRows - sub_nRows; i++)
	{
		for (int j = 0; j < nCols - sub_nCols; j++)
		{
			int total_err = 0;
			for (int sub_i = 0; sub_i < sub_nRows; sub_i++)
			{
				for (int sub_j = 0; sub_j < sub_nCols; sub_j++)
				{
					int index = (i + sub_i)*nCols + (j + sub_j);
					int sub_index = sub_i*sub_nCols + sub_j;
					int err_b = data[3 * index + 0] - data_sub[3 * sub_index + 0];
					int err_g = data[3 * index + 1] - data_sub[3 * sub_index + 1];
					int err_r = data[3 * index + 2] - data_sub[3 * sub_index + 2];
					total_err += (err_b & 0x80000000) ? -err_b : err_b;
					total_err += (err_g & 0x80000000) ? -err_g : err_g;
					total_err += (err_r & 0x80000000) ? -err_r : err_r;
				}
			}

			if (min_err > total_err) {
				min_err = total_err;
				*x = i;
				*y = j;
			}
		}
	}

	if (*x == -1 == *y) {
		cout << "match failed." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	else
	{
		return SUB_IMAGE_MATCH_OK;
	}
}

int ustc_SubImgMatch_corr(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	*x = *y = -1;

	int nCols = grayImg.cols;
	int nRows = grayImg.rows;
	int sub_nCols = subImg.cols;
	int sub_nRows = subImg.rows;
	uchar *data = (uchar *)grayImg.data;
	uchar *data_sub = (uchar *)subImg.data;

	if (sub_nCols > nCols || sub_nRows > nRows)
	{
		cout << "subImg is too big." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	if (sub_nCols == nCols && sub_nRows == nRows)
	{
		*x = 0;
		*y = 0;
		return SUB_IMAGE_MATCH_OK;
	}

	float max_R = 0;

	float sub_sum = 0;
	int sub_scale = sub_nRows * sub_nCols;
	for (int i = 0; i < sub_scale; i++)
	{
		uchar pix = data_sub[i];
		sub_sum += (pix * pix);
	}
	sub_sum = my_sqrt(sub_sum);

	for (int i = 0; i < nRows - sub_nRows; i++)
	{
		for (int j = 0; j < nCols - sub_nCols; j++)
		{
			float sum = 0;
			unsigned long long cov_sum = 0;
			float now_R = 0;

			for (int sub_i = 0; sub_i < sub_nRows; sub_i++)
			{
				for (int sub_j = 0; sub_j < sub_nCols; sub_j++)
				{
					int index = (i + sub_i)*nCols + (j + sub_j);
					int sub_index = sub_i*sub_nCols + sub_j;
					int pix = data[index];
					int sub_pix = data_sub[sub_index];
					sum += pix*pix;
					cov_sum += pix*sub_pix;
				}
			}
			sum = my_sqrt(sum);
			now_R = cov_sum / (sum * sub_sum);

			if (max_R < now_R)
			{
				max_R = now_R;
				*x = i;
				*y = j;
			}
		}
	}

	if (*x == -1 == *y) {
		cout << "match failed." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	else
	{
		return SUB_IMAGE_MATCH_OK;
	}
}

int ustc_SubImgMatch_angle(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	*x = *y = -1;

	int nCols = grayImg.cols;
	int nRows = grayImg.rows;
	int sub_nCols = subImg.cols;
	int sub_nRows = subImg.rows;

	if (sub_nCols > nCols || sub_nRows > nRows)
	{
		cout << "subImg is too big." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	if (sub_nCols == nCols && sub_nRows == nRows)
	{
		*x = 0;
		*y = 0;
		return SUB_IMAGE_MATCH_OK;
	}

	Mat grad_x(nRows, nCols, CV_32FC1);
	Mat grad_y(nRows, nCols, CV_32FC1);
	Mat angleImg(nRows, nCols, CV_32FC1);
	Mat magImg(nRows, nCols, CV_32FC1);

	Mat sub_grad_x(sub_nRows, sub_nCols, CV_32FC1);
	Mat sub_grad_y(sub_nRows, sub_nCols, CV_32FC1);
	Mat sub_angleImg(sub_nRows, sub_nCols, CV_32FC1);
	Mat sub_magImg(sub_nRows, sub_nCols, CV_32FC1);

	ustc_CalcGrad(grayImg, grad_x, grad_y);
	ustc_CalcAngleMag(grad_x, grad_y, angleImg, magImg);

	ustc_CalcGrad(subImg, sub_grad_x, sub_grad_y);
	ustc_CalcAngleMag(sub_grad_x, sub_grad_y, sub_angleImg, sub_magImg);

	float *data = (float *)angleImg.data;
	float *sub_data = (float *)sub_angleImg.data;

	float min_err = FLT_MAX;

	for (int i = 0; i < nRows - sub_nRows; i++)
	{
		for (int j = 0; j < nCols - sub_nCols; j++)
		{
			float total_err = 0;
			for (int sub_i = 1; sub_i < sub_nRows - 1; sub_i++)
			{
				for (int sub_j = 1; sub_j < sub_nCols - 1; sub_j++)
				{
					int index = (i + sub_i)*nCols + (j + sub_j);
					int sub_index = sub_i*sub_nCols + sub_j;
					int err = data[index] - sub_data[sub_index];
					total_err += (err & 0x80000000) ? -err : err;
				}
			}

			if (min_err > total_err) {
				min_err = total_err;
				*x = i;
				*y = j;
			}
		}
	}

	if (*x == -1 == *y) {
		cout << "match failed." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	else
	{
		return SUB_IMAGE_MATCH_OK;
	}
}

int ustc_SubImgMatch_mag(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	*x = *y = -1;

	int nCols = grayImg.cols;
	int nRows = grayImg.rows;
	int sub_nCols = subImg.cols;
	int sub_nRows = subImg.rows;

	if (sub_nCols > nCols || sub_nRows > nRows)
	{
		cout << "subImg is too big." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	if (sub_nCols == nCols && sub_nRows == nRows)
	{
		*x = 0;
		*y = 0;
		return SUB_IMAGE_MATCH_OK;
	}

	Mat grad_x(nRows, nCols, CV_32FC1);
	Mat grad_y(nRows, nCols, CV_32FC1);
	Mat angleImg(nRows, nCols, CV_32FC1);
	Mat magImg(nRows, nCols, CV_32FC1);

	Mat sub_grad_x(sub_nRows, sub_nCols, CV_32FC1);
	Mat sub_grad_y(sub_nRows, sub_nCols, CV_32FC1);
	Mat sub_angleImg(sub_nRows, sub_nCols, CV_32FC1);
	Mat sub_magImg(sub_nRows, sub_nCols, CV_32FC1);

	ustc_CalcGrad(grayImg, grad_x, grad_y);
	ustc_CalcAngleMag(grad_x, grad_y, angleImg, magImg);

	ustc_CalcGrad(subImg, sub_grad_x, sub_grad_y);
	ustc_CalcAngleMag(sub_grad_x, sub_grad_y, sub_angleImg, sub_magImg);

	float *data = (float *)magImg.data;
	float *sub_data = (float *)sub_magImg.data;

	float min_err = FLT_MAX;

	for (int i = 0; i < nRows - sub_nRows; i++)
	{
		for (int j = 0; j < nCols - sub_nCols; j++)
		{
			float total_err = 0;
			for (int sub_i = 1; sub_i < sub_nRows - 1; sub_i++)
			{
				for (int sub_j = 1; sub_j < sub_nCols - 1; sub_j++)
				{
					int index = (i + sub_i)*nCols + (j + sub_j);
					int sub_index = sub_i*sub_nCols + sub_j;
					int err = data[index] - sub_data[sub_index];
					total_err += (err & 0x80000000) ? -err : err;
				}
			}

			if (min_err > total_err) {
				min_err = total_err;
				*x = i;
				*y = j;
			}
		}
	}

	if (*x == -1 == *y) {
		cout << "match failed." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	else
	{
		return SUB_IMAGE_MATCH_OK;
	}
}

int ustc_SubImgMatch_hist(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	*x = *y = -1;

	int min_err = 0x7FFFFFFF;
	int nCols = grayImg.cols;
	int nRows = grayImg.rows;
	int sub_nCols = subImg.cols;
	int sub_nRows = subImg.rows;

	if (sub_nCols > nCols || sub_nRows > nRows)
	{
		cout << "subImg is too big." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	if (sub_nCols == nCols && sub_nRows == nRows)
	{
		*x = 0;
		*y = 0;
		return SUB_IMAGE_MATCH_OK;
	}

	int hist_len = 0;
	uchar *data = (uchar *)grayImg.data;
	uchar *sub_data = (uchar *)subImg.data;

	for (int i = 0; i < nRows; i++)
	{
		for (int j = 0; j < nCols; j++)
		{
			int pix = data[i*nCols + j];
			if (pix > hist_len)
			{
				hist_len = pix;
			}
		}
	}
	hist_len++;

	int *hist = new int[hist_len];
	int *sub_hist = new int[hist_len];
	for (int i = 0; i < hist_len; i++)
	{
		sub_hist[i] = 0;
	}

	for (int i = 0; i < sub_nRows; i++)
	{
		for (int j = 0; j < sub_nCols; j++)
		{
			int sub_pix = sub_data[i*sub_nCols + j];
			sub_hist[sub_pix]++;
		}
	}

	for (int i = 0; i < nRows - sub_nRows; i++)
	{
		for (int j = 0; j < nCols - sub_nCols; j++)
		{
			unsigned long long total_err = 0;
			for (int i = 0; i < hist_len; i++)
			{
				hist[i] = 0;
			}
			for (int sub_i = 0; sub_i < sub_nRows; sub_i++)
			{
				for (int sub_j = 0; sub_j < sub_nCols; sub_j++)
				{
					int pix = data[(i + sub_i)*nCols + (j + sub_j)];
					hist[pix]++;
				}
			}
			for (int i = 0; i < hist_len; i++)
			{
				int err = hist[i] - sub_hist[i];
				total_err += (err & 0x80000000) ? -err : err;
			}

			if (min_err > total_err)
			{
				min_err = total_err;
				*x = i;
				*y = j;
			}
		}
	}

	if (*x == -1 == *y) {
		cout << "match failed." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	else
	{
		return SUB_IMAGE_MATCH_OK;
	}
}
