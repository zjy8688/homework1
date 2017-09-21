#include "SubImageMatch.h"

inline int Abs(int i)
{
	int j = i >> 31;
	return (i ^ j) - j;
}

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

float Sqrt(float x)
{
	int i = *(int*)&x;
	i -= 0x3f800000;
	i >>= 1;
	i += 0x3f800000;
	return *(float*)&i;
}

int ustc_ConvertBgr2Gray(Mat bgrImg, Mat& grayImg)
{
	if (!(bgrImg.data && bgrImg.depth() == CV_8U && bgrImg.channels() == 3
		&& grayImg.size() == bgrImg.size() && grayImg.depth() == CV_8U && grayImg.channels() == 1)) return -1;
	uchar*pBgr = bgrImg.data;
	uchar*pGray = grayImg.data;
	int coefRed = 0.299 * 65536;
	int coefGreen = 0.587 * 65536;
	int coefBlue = 0.114 * 65536;
	while (pBgr < bgrImg.dataend)
	{
		*pGray = ((pBgr[0] * coefBlue) >> 16) + ((pBgr[1] * coefGreen) >> 16) + ((pBgr[2] * coefRed) >> 16);
		pBgr += 3;
		pGray++;
	}
	return 1;
}

int ustc_CalcGrad(Mat grayImg, Mat& gradImg_x, Mat& gradImg_y)
{
	if (!(grayImg.data && grayImg.depth() == CV_8U && grayImg.channels() == 1
		&& gradImg_x.size() == grayImg.size() && gradImg_x.depth() == CV_32F && gradImg_x.channels() == 1
		&& gradImg_y.size() == grayImg.size() && gradImg_y.depth() == CV_32F && gradImg_y.channels() == 1)) return -1;
	int width = grayImg.cols;
	int height = grayImg.rows;
	uchar*rowG;
	float*rowX = gradImg_x.ptr<float>(0);
	float*rowY = gradImg_y.ptr<float>(0);
 	for (int j = 0; j < width; j++) rowX[j] = rowY[j] = 0;
	for (int i = 1; i < height - 1; i++)
	{
		rowG = grayImg.ptr<uchar>(i);
		rowX = gradImg_x.ptr<float>(i);
		rowY = gradImg_y.ptr<float>(i);
		rowX[0] = rowY[0] = rowX[width] = rowY[width] = 0;
		for (int j = 1; j < width - 1; j++)
		{
			rowX[j] = (rowG - width)[j - 1] - (rowG - width)[j + 1] + 2 * rowG[j - 1] - 2 * rowG[j + 1]
				+ (rowG + width)[j - 1] - (rowG + width)[j + 1];
			rowY[j] = (rowG - width)[j + 1] - (rowG + width)[j + 1] + 2 * (rowG - width)[j] - 2 * (rowG + width)[j]
				+ (rowG - width)[j - 1] - (rowG + width)[j - 1];
		}
	}
	rowX = gradImg_x.ptr<float>(height - 1);
	rowY = gradImg_y.ptr<float>(height - 1);
	for (int j = 0; j < width; j++) rowX[j] = rowY[j] = 0;
	return 1;
}

int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y, Mat&angleImg, Mat&magImg)
{
	if (!(gradImg_x.data && gradImg_x.depth() == CV_32F && gradImg_x.channels() == 1
		&& gradImg_y.data && gradImg_y.depth() == CV_32F && gradImg_y.channels() == 1
		&& angleImg.size() == gradImg_x.size() && angleImg.depth() == CV_32F && angleImg.channels() == 1
		&& magImg.size() == gradImg_x.size() && magImg.depth() == CV_32F && magImg.channels() == 1)) return -1;
	float*pX = (float*)gradImg_x.data;
	float*pY = (float*)gradImg_y.data;
	float*pAngle = (float*)angleImg.data;
	float*pMag = (float*)magImg.data;
	while (pX < (float*)gradImg_x.dataend)
	{
		*pMag = Sqrt((*pX) * (*pX) + (*pY) * (*pY));
		*pAngle = Arctan(*pX, *pY) * 180.0 / 3.1415927;
		pX++, pY++, pAngle++, pMag++;
	}
	return 0;
}

int ustc_CalcMag(Mat gradImg_x, Mat gradImg_y, Mat&magImg)
{
	float*pX = (float*)gradImg_x.data;
	float*pY = (float*)gradImg_y.data;
	uchar*pMag = magImg.data;
	while (pMag < magImg.dataend)
	{
		*pMag = Sqrt((*pX) * (*pX) + (*pY) * (*pY)) / 4 / 1.41421356;
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

int ustc_Threshold(Mat grayImg, Mat&binaryImg, int th)
{
	if (!(grayImg.data && grayImg.depth() == CV_8U && grayImg.channels() == 1
		&& binaryImg.size() == grayImg.size() && binaryImg.depth() == CV_8U && binaryImg.channels() == 1)) return -1;
	uchar*pG = grayImg.data;
	uchar*pB = binaryImg.data;
	uchar whiteBlack[256];
	int i;
	for (i = 0; i < th; i++) whiteBlack[i] = 0;
	for (i = th; i < 256; i++) whiteBlack[i] = 255;
	while (pG < grayImg.dataend)
	{
		*pB = whiteBlack[*pG];
		*pG++, *pB++;
	}
	return 1;
}

int ustc_CalcHist(Mat grayImg, int*hist, int hist_len)
{
	if (!(grayImg.data && grayImg.depth() == CV_8U && grayImg.channels() == 1 && hist_len == 256)) return -1;
	for (int i = 0; i < hist_len; i++) hist[i] = 0;
	for (uchar*p = grayImg.data; p < grayImg.dataend; p++) hist[*p]++;
	return 1;
}

int ustc_SubImgMatch_gray(Mat grayImg, Mat subImg, int*x, int*y)
{
	if (!(grayImg.data && grayImg.depth() == CV_8U && grayImg.channels() == 1
		&& subImg.data && subImg.depth() == CV_8U && subImg.channels() == 1
		&& subImg.rows <= grayImg.rows && subImg.cols <= grayImg.cols)) return -1;
	long d0 = (subImg.dataend - subImg.datastart) * 255;
	int i0, j0;
	for (int i = 0; i <= grayImg.rows - subImg.rows; i++)
	{
		for (int j = 0; j <= grayImg.cols - subImg.cols; j++)
		{
			long d = 0;
			for (int k = 0; k < subImg.rows; k++)
			{
				uchar*p = grayImg.data + (i + k) * grayImg.cols + j;
				uchar*q = subImg.data + k * subImg.cols;
				for (int l = 0; l < subImg.cols; l++) d += Abs(p[l] - q[l]);
			}
 			if (d < d0) d0 = d, i0 = i, j0 = j;
		}
	}
	*x = j0, *y = i0;
	return 1;
}

int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int* x, int* y)
{
	if (!(colorImg.data && colorImg.depth() == CV_8U && colorImg.channels() == 3
		&& subImg.data && subImg.depth() == CV_8U && subImg.channels() ==3
		&& subImg.rows <= colorImg.rows && subImg.cols <= colorImg.cols)) return -1;
	long d0 = (subImg.dataend - subImg.datastart) * 3 * 255;
	int i0, j0;
	for (int i = 0; i <= colorImg.rows - subImg.rows; i++)
	{
		for (int j = 0; j <= colorImg.cols - subImg.cols; j++)
		{
			long d = 0;
			for (int k = 0; k < subImg.rows; k++)
			{
				uchar*p = colorImg.data + ((i + k) * colorImg.cols + j) * 3;
				uchar*q = subImg.data + k * subImg.cols * 3;
				for (int l = subImg.cols * 3 - 1; l >= 0; l--) d += Abs(p[l] - q[l]);
			}
			if (d < d0) d0 = d, i0 = i, j0 = j;
		}
	}
	*x = j0, *y = i0;
	return 1;
}

int ustc_SubImgMatch_corr(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (!(grayImg.data && grayImg.depth() == CV_8U && grayImg.channels() == 1
		&& subImg.data && subImg.depth() == CV_8U && subImg.channels() == 1
		&& subImg.rows <= grayImg.rows && subImg.cols <= grayImg.cols)) return -1;
	float r02 = 0;
	int i0, j0;
	for (int i = 0; i <= grayImg.rows - subImg.rows; i++)
	{
		for (int j = 0; j <= grayImg.cols - subImg.cols; j++)
		{
			int sigmaS2 = 0, sigmaT2 = 0, sigmaST = 0;
			for (int k = 0; k < subImg.rows; k++)
			{
				uchar*p = grayImg.data + (i + k) * grayImg.cols + j;
				uchar*q = subImg.data + k * subImg.cols;
				for (int l = 0; l < subImg.cols; l++)
				{
					sigmaS2 += p[l] * p[l];
					sigmaT2 += q[l] * q[l];
					sigmaST += p[l] * q[l];
				}
			}
			float r2 = ((float)sigmaST * (float)sigmaST) / ((float)sigmaS2 * (float)sigmaT2);
			if (r2 > r02) r02 = r2, i0 = i, j0 = j;
		}
	}
	*x = j0, *y = i0;
	return 1;
}

int ustc_SubImgMatch_angle(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (!(grayImg.data && grayImg.depth() == CV_8U && grayImg.channels() == 1
		&& subImg.data && subImg.depth() == CV_8U && subImg.channels() == 1
		&& subImg.rows <= grayImg.rows && subImg.cols <= grayImg.cols)) return -1;
	Mat grayX(grayImg.size(), CV_32FC1), grayY(grayImg.size(), CV_32FC1);
	ustc_CalcGrad(grayImg, grayX, grayY);
	Mat grayAngle(grayImg.size(), CV_8UC1);
	ustc_CalcAngle(grayX, grayY, grayAngle);
	Mat subX(subImg.size(), CV_32FC1), subY(subImg.size(), CV_32FC1);
	ustc_CalcGrad(subImg, subX, subY);
	Mat subAngle(subImg.size(), CV_8UC1);
	ustc_CalcAngle(subX, subY, subAngle);
	int d0 = (subImg.dataend - subImg.datastart) * 128;
	int i0, j0;
	int angle[256];
	for (int i = 0; i < 129; i++) angle[i] = i;
	for (int i = 129; i < 256; i++) angle[i] = 256 - i;
	for (int i = 0; i <= grayImg.rows - subImg.rows; i++)
	{
		for (int j = 0; j <= grayImg.cols - subImg.cols; j++)
		{
			int d = 0;
			for (int k = 0; k < subImg.rows; k++)
			{
				uchar*p = grayImg.data + (i + k) * grayImg.cols + j;
				uchar*q = subImg.data + k * subImg.cols;
				for (int l = 0; l < subImg.cols; l++) d += angle[uchar(p[l] - q[l])];
			}
			if (d < d0) d0 = d, i0 = i, j0 = j;
		}
	}
	*x = j0, *y = i0;
	return 1;
}

int ustc_SubImgMatch_mag(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (!(grayImg.data && grayImg.depth() == CV_8U && grayImg.channels() == 1
		&& subImg.data && subImg.depth() == CV_8U && subImg.channels() == 1
		&& subImg.rows <= grayImg.rows && subImg.cols <= grayImg.cols)) return -1;
	Mat grayX(grayImg.size(), CV_32FC1), grayY(grayImg.size(), CV_32FC1);
	ustc_CalcGrad(grayImg, grayX, grayY);
	Mat grayMag(grayImg.size(), CV_8UC1);
	ustc_CalcMag(grayX, grayY, grayMag);
	Mat subX(subImg.size(), CV_32FC1), subY(subImg.size(), CV_32FC1);
	ustc_CalcGrad(subImg, subX, subY);
	Mat subMag(subImg.size(), CV_8UC1);
	ustc_CalcMag(subX, subY, subMag);
	long d0 = (subImg.dataend - subImg.datastart) * 255;
	int i0, j0;
	for (int i = 0; i <= grayImg.rows - subImg.rows; i++)
	{
		for (int j = 0; j <= grayImg.cols - subImg.cols; j++)
		{
			long d = 0;
			for (int k = 0; k < subImg.rows; k++)
			{
				uchar*p = grayImg.data + (i + k) * grayImg.cols + j;
				uchar*q = subImg.data + k * subImg.cols;
				for (int l = 0; l < subImg.cols; l++) d += Abs(p[l] - q[l]);
			}
			if (d < d0) d0 = d, i0 = i, j0 = j;
		}
	}
	*x = j0, *y = i0;
	return 1;
}

int ustc_SubImgMatch_hist(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (!(grayImg.data && grayImg.depth() == CV_8U && grayImg.channels() == 1
		&& subImg.data && subImg.depth() == CV_8U && subImg.channels() == 1
		&& subImg.rows <= grayImg.rows && subImg.cols <= grayImg.cols)) return -1;
	int grayHist[256], subHist[256];
	ustc_CalcHist(subImg, subHist, 256);
	long d0 = 0x7fffffff;
	int i0, j0;
	for (int i = 0; i <= grayImg.rows - subImg.rows; i++)
	{
		for (int j = 0; j < grayImg.cols - subImg.cols; j++)
		{
			ustc_CalcHist(grayImg(Rect(j, i, subImg.cols, subImg.rows)).clone(), grayHist, 256);
			long d = 0;
			for (int k = 0; k < 256; k++) d += Abs(grayHist[k] - subHist[k]);
			if (d < d0) d0 = d, i0 = i, j0 = j;
		}
	}
	*x = j0, *y = i0;
	return 1;
}
