
#include "SubImageMatch.h"

using namespace std;
using namespace cv;

inline float mySqrt(float x)
{
	float a = x;
	unsigned int i = *(unsigned int *)&x;
	i = (i + 0x3f76cf62) >> 1;
	x = *(float *)&i;
	x = (x + a / x) * 0.5f;
	return x;
}
int ustc_ConvertBgr2Gray(Mat bgrImg, Mat& grayImg)
{
//	int a = 114, b = 587, c = 299;
	int i = 0, j = 0, row = bgrImg.rows, col = bgrImg.cols;
	grayImg = Mat(row, col, CV_8UC1, 1);
	for (i = 0;i < row;i++)
	{
		for (j = 0;j < col;j++)
		{
			grayImg.at<uchar>(i, j) = bgrImg.at<Vec3b>(i, j)[0] * 114 / 1000 + bgrImg.at<Vec3b>(i, j)[1] * 587 / 1000 + bgrImg.at<Vec3b>(i, j)[2] * 299 / 1000;
		}
	}
	return 1;
}

int ustc_CalcGrad(Mat grayImg, Mat& gradImg_x, Mat& gradImg_y)
{
	int i = 0, j = 0, row = grayImg.rows, col = grayImg.cols;
	gradImg_x = Mat(row, col, CV_32FC1, 1);
	gradImg_y = Mat(row, col, CV_32FC1, 1);
	row--;
	col--;
	for (i = 1;i < row;i++)
	{
		for (j = 1;j < col;j++)
		{
			gradImg_x.at<float>(i, j) = (grayImg.at<uchar>(i - 1, j + 1) + 2 * grayImg.at<uchar>(i, j + 1) + grayImg.at<uchar>(i + 1, j + 1) - grayImg.at<uchar>(i - 1, j - 1) - 2 * grayImg.at<uchar>(i, j - 1) - grayImg.at<uchar>(i + 1, j - 1))/256.0f;
			gradImg_y.at<float>(i, j) = (grayImg.at<uchar>(i + 1, j - 1) + 2 * grayImg.at<uchar>(i + 1, j) + grayImg.at<uchar>(i + 1, j + 1) - grayImg.at<uchar>(i - 1, j - 1) - 2 * grayImg.at<uchar>(i - 1, j) - grayImg.at<uchar>(i - 1, j + 1))/256.0f;
		}
	}
	return 1;
}

int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y, Mat& angleImg, Mat& magImg)
{
	int i = 0, j = 0, row = gradImg_x.rows, col = gradImg_x.cols;
	float temp;
	angleImg = Mat(row, col, CV_32FC1, 1);
	magImg = Mat(row, col, CV_32FC1, 1);
	for (i = 0;i < row;i++)
	{
		for (j = 0;j < col;j++)
		{
			magImg.at<float>(i, j) = mySqrt(gradImg_x.at<float>(i, j)*gradImg_x.at<float>(i, j) + gradImg_y.at<float>(i, j)*gradImg_x.at<float>(i, j));
			temp = gradImg_y.at<float>(i, j) / gradImg_x.at<float>(i, j);
			if (temp <= 0.5)
			{
				angleImg.at<float>(i, j) = (temp*(1.0f + temp*temp*(-1.0f / 3 + temp*temp*(1.0f / 5 - temp*temp / 7))))*57.295f;
			}
			else if (temp <= 2.0)
			{
				temp--;
				angleImg.at<float>(i, j) = (3.14159f / 4 + temp*(1.0f / 2 + temp*(-1.0f / 4 + temp*(1.0f / 12 + temp*temp*(1.0f / 40 + temp / 48)))))*57.295f;
			}
			else
			{
				angleImg.at<float>(i, j) = (3.14159f / 2 + (((((1.0f / (11 * temp*temp) - 1.0f / 9) / (temp*temp) + 1.0f / 7) / (temp*temp) - 1.0f / 5) / (temp*temp) + 1.0f / 3) / (temp*temp) - 1.0f) / temp)*57.295f;
			}
		}
	}
	return 1;
}

int ustc_Threshold(Mat grayImg, Mat& binaryImg, int th)
{
	int i = 0, j = 0, row = grayImg.rows, col = grayImg.cols;
	binaryImg = Mat(row, col, CV_8UC1, 1);
	for (i = 0;i < row;i++)
	{
		for (j = 0;j < col;j++)
		{
			binaryImg.at<uchar>(i, j) = (grayImg.at<uchar>(i, j) > th) * 255;
		}
	}
	return 1;
}

int ustc_CalcHist(Mat grayImg, int* hist, int hist_len)
{
	int i = 0, j = 0, row = grayImg.rows, col = grayImg.cols;
	memset(hist, 0, hist_len);
	for (i = 0;i < row;i++)
	{
		for (j = 0;j < col;j++)
		{
			hist[grayImg.at<uchar>(i, j)*256/hist_len]++;
		}
	}
	return 1;
}

int ustc_SubImgMatch_gray(Mat grayImg, Mat subImg, int* x, int* y)
{
	int i = 0, j = 0, k = 0, l = 0, m = 0;
	int mx = 0, my = 0;
	uchar *pg(NULL), *ps(NULL);
	long long temp = 0;
	long long ans = 0xffffffffffffffff;
	int row = grayImg.rows, col = grayImg.cols;
	int rrr = subImg.rows, ccc = subImg.cols;
	row -= rrr;
	col -= ccc;
	for (i = 0;i < row;i++)
	{
		for (j = 0;j < col;j++)
		{
			temp = 0;
			for (k = 0;k < rrr;k++)
			{
				pg = grayImg.ptr<uchar>(i + k);
				ps = subImg.ptr<uchar>(k);
				for (l = 0;l < ccc;l++)
				{
					m = pg[j + l] - ps[l];
					temp += m > 0 ? m : -m;
				}
			}
			if (temp < ans)
			{
				ans = temp;
				mx = i, my = j;
			}
		}
	}
	*x = mx;
	*y = my;
	return 1;
}

int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int* x, int* y)
{
	int i = 0, j = 0, k = 0, l = 0, m = 0;
	int mx = 0, my = 0;
	Vec3b *pc(NULL),*ps(NULL);
	long long temp = 0;
	long long ans = 0xffffffffffffffff;
	int row = colorImg.rows, col = colorImg.cols;
	int rrr = subImg.rows, ccc = subImg.cols;
	row -= rrr;
	col -= ccc;
	for (i = 0;i < row;i++)
	{
//		cout << i << endl;
		for (j = 0;j < col;j++)
		{
			temp = 0;
			for (k = 0;k < rrr;k++)
			{
				pc = colorImg.ptr<Vec3b>(i + k);
				ps = subImg.ptr<Vec3b>(k);
				for (l = 0;l < ccc;l++)
				{
					m = pc[j + l][0] - pc[l][0];
					temp += m > 0 ? m : -m;
					m = pc[j + l][1] - pc[l][1];
					temp += m > 0 ? m : -m;
					m = pc[j + l][2] - pc[l][2];
					temp += m > 0 ? m : -m;
				}
			}
			if (temp < ans)
			{
				ans = temp;
				mx = i, my = j;
			}
		}
	}
	*x = mx;
	*y = my;
	return 1;
}

int ustc_SubImgMatch_corr(Mat grayImg, Mat subImg, int* x, int* y)
{
	int i = 0, j = 0, k = 0, l = 0, m = 0;
	int mx = 0, my = 0;
	int row = grayImg.rows, col = grayImg.cols;
	int rrr = subImg.rows, ccc = subImg.cols;
	float gg=0, ss=0;
	float temp, ans=128.0;
	Mat gsquare(row, col, CV_32FC1, 1);
	Mat ssquare(rrr, ccc, CV_32FC1, 1);
	float *pp(NULL), *pg(NULL), *ps(NULL);
	for (i = 0;i < row;i++)
	{
		pp = grayImg.ptr<float>(i);
		pg = gsquare.ptr<float>(i);
		for (j = 0;j < col;j++)
		{
			pg[j] = pp[i] * pp[i];
		}
	}
	for (i = 0;i < rrr;i++)
	{
		pp = subImg.ptr<float>(i);
		ps = ssquare.ptr<float>(i);
		for (j = 0;j < col;j++)
		{
			ps[j] = pp[i] * pp[i];
		}
	}
	row -= rrr;
	col -= ccc;
	for (i = 0;i < row;i++)
	{
		for (j = 0;j < col;j++)
		{
			temp = 0.0;
			ss = gg = 0;
			for (k = 0;k < rrr;k++)
			{
				pg = grayImg.ptr<float>(i + k);
				ps = subImg.ptr<float>(k);
				for (l = 0;l < ccc;l++)
				{
					temp += pg[j + l] * ps[l];
					ss += ssquare.at<float>(i + k, j + l);
					gg += gsquare.at<float>(k, l);
				}
			}
			temp /= (mySqrt(ss)*mySqrt(gg));
			if (temp < ans)
			{
				ans = temp;
				mx = i, my = j;
			}
		}
	}
	*x = mx;
	*y = my;
	return 1;
}

int ustc_SubImgMatch_angle(Mat grayImg, Mat subImg, int* x, int* y)
{
	int i = 0, j = 0, k = 0, l = 0, m = 0;
	int mx = 0, my = 0;
	int row = grayImg.rows, col = grayImg.cols;
	int rrr = subImg.rows, ccc = subImg.cols;
	row -= rrr;
	col -= ccc;
	int temp, ans = 0x7fffffff;
	float *pg(NULL), *ps(NULL);
	Mat gangle, gmag, sangle, smag;
	Mat ggradx, ggrady, sgradx, sgrady;
	ustc_CalcGrad(grayImg, ggradx, ggrady);
	ustc_CalcGrad(subImg, sgradx, sgrady);
	ustc_CalcAngleMag(ggradx, ggrady, gangle, gmag);
	ustc_CalcAngleMag(sgradx, sgrady, sangle, smag);
	for (i = 0;i < row;i++)
	{
		for (j = 0;j < col;j++)
		{
			temp = 0;
			for (k = 0;k < rrr;k++)
			{
				pg = gangle.ptr<float>(i + k);
				ps = sangle.ptr<float>(k);
				for (l = 0;l < ccc;l++)
				{
					m = pg[j + l] - ps[l];
					temp += m > 0 ? m : -m;
				}
			}
			if (temp < ans)
			{
				ans = temp;
				mx = i, my = j;
			}
		}
	}
	*x = mx;
	*y = my;
	return 1;
}

int ustc_SubImgMatch_mag(Mat grayImg, Mat subImg, int* x, int* y)
{
	int i = 0, j = 0, k = 0, l = 0;
	int mx = 0, my = 0;
	int row = grayImg.rows, col = grayImg.cols;
	int rrr = subImg.rows, ccc = subImg.cols;
	row -= rrr;
	col -= ccc;
	float temp, ans = 128.0, m = 0.0;
	float *pg(NULL), *ps(NULL);
	Mat gangle, gmag, sangle, smag;
	Mat ggradx, ggrady, sgradx, sgrady;
	ustc_CalcGrad(grayImg, ggradx, ggrady);
	ustc_CalcGrad(subImg, sgradx, sgrady);
	ustc_CalcAngleMag(ggradx, ggrady, gangle, gmag);
	ustc_CalcAngleMag(sgradx, sgrady, sangle, smag);
	for (i = 0;i < row;i++)
	{
		for (j = 0;j < col;j++)
		{
			temp = 0.0;
			for (k = 0;k < rrr;k++)
			{
				pg = gmag.ptr<float>(i + k);
				ps = smag.ptr<float>(k);
				for (l = 0;l < ccc;l++)
				{
					m = pg[j + l] - ps[l];
					temp += m > 0 ? m : -m;
				}
			}
			if (temp < ans)
			{
				ans = temp;
				mx = i, my = j;
			}
		}
	}
	*x = mx;
	*y = my;
	return 1;
}

int ustc_SubImgMatch_hist(Mat grayImg, Mat subImg, int* x, int* y)
{
	int ghist[1000], shist[1000];
	int mx = 0, my = 0;
	int i = 0, j = 0, k = 0, l = 0, m = 0;
	int row = grayImg.rows, col = grayImg.cols;
	int rrr = subImg.rows, ccc = subImg.cols;
	int temp, ans = 0x7fffffff;
	uchar *pg(NULL), *ps(NULL);
	row -= rrr;
	col -= ccc;
	for (i = 0;i < row;i++)
	{
		for (j = 0;j < col;j++)
		{
			temp = 0;
			memset(ghist, 0, 256);
			memset(shist, 0, 256);
			for (k = 0;k < rrr;k++)
			{
				pg = grayImg.ptr<uchar>(i + k);
				ps = subImg.ptr<uchar>(k);
				for (l = 0;l < ccc;l++)
				{
					ghist[pg[j + l]]++;
					shist[ps[l]]++;
				}
			}
			for (k = 0;k < 256;k++)
			{
				m = ghist[k] - shist[k];
				temp += m > 0 ? m : -m;
			}
			if (temp < ans)
			{
				ans = temp;
				mx = i, my = j;
			}
		}
	}
	return 1;
}
