#include "SubImageMatch.h"
#include<stdio.h>
#include<time.h>
#include<opencv2\opencv.hpp>
using namespace cv;
#define pi 3.14159
#include<math.h>

int ustc_ConvertBgr2Gray(Mat bgrImg, Mat& grayImg)
{
	if (NULL == bgrImg.data)
	{
		printf("bgrImage is NULL!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}

	uchar *data = bgrImg.data;
	int nr = bgrImg.rows;
	int nc = bgrImg.cols;
	int num = nr*nc;
	grayImg = Mat(nr,nc,CV_8UC1,Scalar(0));
	uchar *gData = grayImg.data;

	for(int i=0;i<num;i++)
	{
		*gData = uchar(*data * 0.114 + *(data++) * 0.587 + *(data++) * 0.299);
		gData++; data++;
	}

	return SUB_IMAGE_MATCH_OK;
}

int ustc_CalcGrad(Mat grayImg, Mat& gradImg_x, Mat& gradImg_y)	
{
	if (NULL == grayImg.data)
	{
		printf("grayImage is NULL!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}

	int Gx[3][3] = { {-1,0,1},{-2,0,2},{-1,0,1} };
	int Gy[3][3] = { {-1,-2,-1},{0,0,0},{1,2,1} };
	int nr = grayImg.rows;
	int nc = grayImg.cols;
	float *px,*py;
	uchar *q1,*q2,*q3;
	gradImg_x = Mat(nr, nc, CV_32FC1, Scalar(0));
	gradImg_y = Mat(nr, nc, CV_32FC1, Scalar(0));

	for (int i = 1; i < nr - 1; i++)
	{
		px = gradImg_x.ptr<float>(i);
		py = gradImg_y.ptr<float>(i);
		q1 = grayImg.ptr<uchar>(i - 1);
		q2 = grayImg.ptr<uchar>(i);
		q3 = grayImg.ptr<uchar>(i + 1);
		for (int j = 1; j < nc - 1; j++)
		{
			px[j] = (float)(q1[j + 1] + 2 * q2[j + 1] + q3[j + 1]) - (float)(q1[j - 1] + 2 * q2[j - 1] + q3[j - 1]);
			py[j] = (float)(q3[j - 1] + 2 * q3[j] + q3[j + 1]) - (float)(q1[j - 1] + 2 * q1[j] + q1[j + 1]);
		}
	}

	return SUB_IMAGE_MATCH_OK;
}

int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y, Mat& angleImg, Mat& magImg)	
{
	if (NULL == gradImg_x.data || NULL == gradImg_y.data)
	{
		printf("gradImage is NULL!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}
	float *px, *py, *qa,*qm;
	float mag;
	int nr = gradImg_x.rows;
	int nc = gradImg_x.cols;
	int nr_y = gradImg_y.rows;
	int nc_y = gradImg_y.cols;

	if (nr != nr_y || nc != nc_y)
	{
		printf("The size of gradImg_x does not match the size of gradImg_y!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}

	angleImg = Mat(nr, nc, CV_32FC1, Scalar(0));
	magImg = Mat(nr, nc, CV_32FC1, Scalar(0));
	
	for (int i = 1; i < nr - 1; i++)
	{
		px = gradImg_x.ptr<float>(i);
		py = gradImg_y.ptr<float>(i);
		qa = angleImg.ptr<float>(i);
		qm = magImg.ptr<float>(i);
		for (int j = 1; j < nc - 1; j++)
		{
			mag = py[j] * py[j] + px[j] * px[j];

			qa[j] = atan2(py[j], px[j]);
			if (qa[j] < 0)
				qa[j] += 360;

			qm[j] = sqrt(mag);
		}
	}

	return SUB_IMAGE_MATCH_OK;
}

int ustc_Threshold(Mat grayImg, Mat& binaryImg, int th)	
{
	if (NULL == grayImg.data)
	{
		printf("grayImage is NULL!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}
	int nr = grayImg.rows;
	int nc = grayImg.cols;
	binaryImg = Mat(nr, nc, CV_8UC1, Scalar(0));
	uchar *pg, *pb;

	for (int i = 0; i < nr; i++)
	{
		pg = grayImg.ptr<uchar>(i);
		pb = binaryImg.ptr<uchar>(i);
		for (int j = 0; j < nc; j++)
		{
			pb[j] = (pg[j] - th >= 0) ? 255 : 0;
		}
	}
	return SUB_IMAGE_MATCH_OK;
}

int ustc_CalcHist(Mat grayImg, int* hist, int hist_len)
{
	if (NULL == grayImg.data)
	{
		printf("grayImage is NULL!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}

	if (hist_len < 256)
	{
		printf("The length of hist is not enough!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}

	for (int i = 0; i < 256; i++)
		hist[i] = 0;

	uchar *p;
	int nr = grayImg.rows;
	int nc = grayImg.cols;

	for (int i = 0; i < nr; i++)
	{
		p = grayImg.ptr<uchar>(i);
		for (int j = 0; j < nc; j++)
		{
			hist[p[j]]++;
		}
	}

	return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_gray(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		printf("Image is NULL!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}

	int nc = grayImg.cols;
	int nr = grayImg.rows;
	int sub_nc = subImg.cols;
	int sub_nr = subImg.rows;

	if (nc < sub_nc || nr < sub_nr)
	{
		printf("SubImg is larger than grayImg!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}

	int tdi_nc = nc - sub_nc + 1;
	int tdi_nr = nr - sub_nr + 1;
	int totalDiff,diff;
	int minDiff=255;
	Mat totalDiffImg(tdi_nr, tdi_nc, CV_32FC1, Scalar(255));

	for (int i_out = 0; i_out < tdi_nr; i_out++)
	{
		for (int j_out = 0; j_out < tdi_nc; j_out++)
		{
			totalDiff = 0;
			for (int i_in = 0; i_in < sub_nr; i_in++)
			{
				for (int j_in = 0; j_in < sub_nc; j_in++)
				{
					int rIndex = i_out + i_in;
					int cIndex = j_out + j_in;
					//totalDiff += abs(grayImg.data[rIndex*nc + cIndex] - subImg.data[i_in*sub_nc + j_in]);
					diff = grayImg.data[rIndex*nc + cIndex] - subImg.data[i_in*sub_nc + j_in];
					totalDiff += (diff >> 31) * 2 * diff + diff;
				}
			}
			((float*)totalDiffImg.data)[i_out * tdi_nc + j_out] = (float)totalDiff;
			//printf("%.2f   ", (float)totalDiff);
			if (totalDiff < minDiff)
			{
				minDiff = totalDiff;
				*x = i_out;
				*y = j_out;
			}
		}
	}

	return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int* x, int* y)
{
	if (NULL == colorImg.data || NULL == subImg.data)
	{
		printf("Image is NULL!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}

	int nc = colorImg.cols;
	int nr = colorImg.rows;
	int sub_nc = subImg.cols;
	int sub_nr = subImg.rows;

	if (nc < sub_nc || nr < sub_nr)
	{
		printf("SubImg is larger than colorImg!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}

	int tdi_nc = nc - sub_nc + 1;
	int tdi_nr = nr - sub_nr + 1;
	int totalDiff;
	int minDiff = 255;
	int bDiff, gDiff, rDiff;
	Mat totalDiffImg(tdi_nr, tdi_nc, CV_32FC1, Scalar(255));

	for (int i_out = 0; i_out < tdi_nr; i_out++)
	{
		for (int j_out = 0; j_out < tdi_nc; j_out++)
		{
			totalDiff = 0;
			for (int i_in = 0; i_in < sub_nr; i_in++)
			{
				for (int j_in = 0; j_in < sub_nc; j_in++)
				{
					int rIndex = i_out + i_in;
					int cIndex = (j_out + j_in) * 3;
					bDiff = colorImg.data[rIndex*nc*3 + cIndex] - subImg.data[i_in*sub_nc*3 + j_in*3];
					gDiff = colorImg.data[rIndex*nc*3 + cIndex + 1] - subImg.data[i_in*sub_nc*3 + j_in*3 + 1];
					rDiff = colorImg.data[rIndex*nc*3 + cIndex + 2] - subImg.data[i_in*sub_nc*3 + j_in*3 + 2];
					
					totalDiff += ((bDiff >> 31) * 2 * bDiff + bDiff + (gDiff >> 31) * 2 * gDiff + gDiff + (rDiff >> 31) * 2 * rDiff + rDiff);
				}
			}
			((float*)totalDiffImg.data)[i_out * tdi_nc + j_out] = (float)totalDiff;
			//printf("%.2f   ", (float)totalDiff);
			if (totalDiff < minDiff)
			{
				minDiff = totalDiff;
				*x = i_out;
				*y = j_out;
			}
		}
	}

	return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_corr(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		printf("Image is NULL!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}

	int nc = grayImg.cols;
	int nr = grayImg.rows;
	int sub_nc = subImg.cols;
	int sub_nr = subImg.rows;

	if (nc < sub_nc || nr < sub_nr)
	{
		printf("SubImg is larger than grayImg!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}

	int tdi_nc = nc - sub_nc + 1;
	int tdi_nr = nr - sub_nr + 1;
	float diff,minDiff = 15555;
	int cov = 0;
	float subImgV;
	float sigma_gray = 0, sigma_sub = 0;
	float corr;

	for (int i_out = 0; i_out < tdi_nr; i_out++)
	{
		for (int j_out = 0; j_out < tdi_nc; j_out++)
		{
			cov = 0;
			sigma_gray = 0;
			sigma_sub = 0;
			for (int i_in = 0; i_in < sub_nr; i_in++)
			{
				for (int j_in = 0; j_in < sub_nc; j_in++)
				{
					int rIndex = i_out + i_in;
					int cIndex = j_out + j_in;
					subImgV = subImg.data[i_in*sub_nc + j_in];
					cov += grayImg.data[rIndex*nc + cIndex] * subImg.data[i_in*sub_nc + j_in];
					sigma_gray += grayImg.data[rIndex*nc + cIndex] * grayImg.data[rIndex*nc + cIndex];
					sigma_sub += subImgV * subImgV;
				}
			}
			float lg, ng = 1.0;
			do
			{
				lg = ng;
				ng = (lg + sigma_gray / lg) / 2.0;
			} while (lg != ng);
			sigma_gray = lg;

			ng = 1.0;
			do
			{
				lg = ng;
				ng = (lg + sigma_sub / lg) / 2.0;
			} while (lg != ng);
			sigma_sub = lg;
			//sigma_gray = sqrt(sigma_gray);
			//sigma_sub = sqrt(sigma_sub);

			corr = (float)cov / (sigma_gray*sigma_sub);
			diff = 100 * (1.0 - corr);
			if (diff < minDiff)
			{
				minDiff = diff;
				*x = i_out;
				*y = j_out;
			}
		}
	}

	return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_angle(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		printf("Image is NULL!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}
	int match_ok;
	Mat gradImg_x, gradImg_y;	
	match_ok = ustc_CalcGrad(grayImg, gradImg_x, gradImg_y);
	Mat angleImg, magImg;
	match_ok = ustc_CalcAngleMag(gradImg_x, gradImg_y, angleImg, magImg);

	Mat sub_gradImg_x, sub_gradImg_y;
	match_ok = ustc_CalcGrad(subImg, sub_gradImg_x, sub_gradImg_y);
	Mat sub_angleImg, sub_magImg;
	match_ok = ustc_CalcAngleMag(sub_gradImg_x, sub_gradImg_y, sub_angleImg, sub_magImg);

	int nc = grayImg.cols;
	int nr = grayImg.rows;
	int sub_nc = subImg.cols;
	int sub_nr = subImg.rows;

	if (nc < sub_nc || nr < sub_nr)
	{
		printf("SubImg is larger than grayImg!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}

	int tdi_nc = nc - sub_nc + 1;
	int tdi_nr = nr - sub_nr + 1;
	float totalDiff,diff;
	int minDiff = 1555555;

	for (int i_out = 0; i_out < tdi_nr; i_out++)
	{
		for (int j_out = 0; j_out < tdi_nc; j_out++)
		{
			totalDiff = 0;
			for (int i_in = 1; i_in < sub_nr - 1; i_in++)
			{
				for (int j_in = 1; j_in < sub_nc - 1; j_in++)
				{
					int rIndex = i_out + i_in;
					int cIndex = j_out + j_in;
					diff = ((float*)angleImg.data)[rIndex * nc + cIndex] - ((float*)sub_angleImg.data)[i_in * sub_nc + j_in];
					diff = (((int)(diff - 1)) >> 31) * 2 * diff + diff;
					totalDiff += (((int)(diff - 181)) >> 31)*(360 - 2 * diff) + 360 - diff;
					//totalDiff += ((diff >> 31) * 2 * diff + diff);
				}
			}
			if (totalDiff < minDiff)
			{
				minDiff = totalDiff;
				*x = i_out;
				*y = j_out;
			}
		}
	}

	return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_mag(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		printf("Image is NULL!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}
	int match_ok;
	Mat gradImg_x, gradImg_y;	
	match_ok = ustc_CalcGrad(grayImg, gradImg_x, gradImg_y);
	Mat angleImg, magImg;
	match_ok = ustc_CalcAngleMag(gradImg_x, gradImg_y, angleImg, magImg);

	Mat sub_gradImg_x, sub_gradImg_y;
	match_ok = ustc_CalcGrad(subImg, sub_gradImg_x, sub_gradImg_y);
	Mat sub_angleImg, sub_magImg;
	match_ok = ustc_CalcAngleMag(sub_gradImg_x, sub_gradImg_y, sub_angleImg, sub_magImg);

	int nc = grayImg.cols;
	int nr = grayImg.rows;
	int sub_nc = subImg.cols;
	int sub_nr = subImg.rows;

	if (nc < sub_nc || nr < sub_nr)
	{
		printf("SubImg is larger than grayImg!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}

	int tdi_nc = nc - sub_nc + 1;
	int tdi_nr = nr - sub_nr + 1;
	float totalDiff, diff;
	float minDiff = 1555555;

	for (int i_out = 0; i_out < tdi_nr; i_out++)
	{
		for (int j_out = 0; j_out < tdi_nc; j_out++)
		{
			totalDiff = 0;
			for (int i_in = 1; i_in < sub_nr - 1; i_in++)
			{
				for (int j_in = 1; j_in < sub_nc - 1; j_in++)
				{
					int rIndex = i_out + i_in;
					int cIndex = j_out + j_in;
					diff = ((float*)magImg.data)[rIndex*nc + cIndex] - ((float*)sub_magImg.data)[i_in*sub_nc + j_in];
					totalDiff += (((int)diff) >> 31) * 2 * diff + diff;
				}
			}
			if (totalDiff < minDiff)
			{
				minDiff = totalDiff;
				*x = i_out;
				*y = j_out;
			}
		}
	}

	return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_hist(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		printf("Image is NULL!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}

	int nc = grayImg.cols;
	int nr = grayImg.rows;
	int sub_nc = subImg.cols;
	int sub_nr = subImg.rows;

	if (nc < sub_nc || nr < sub_nr)
	{
		printf("SubImg is larger than grayImg!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}

	int tdi_nc = nc - sub_nc + 1;
	int tdi_nr = nr - sub_nr + 1;

	int subHist[256] = { 0 };
	int match_ok;
	int totalDiff,diff;
	int minDiff = 1555555;

	match_ok = ustc_CalcHist(subImg, subHist, 256);

	for (int i_out = 0; i_out < tdi_nr; i_out++)
	{
		for (int j_out = 0; j_out < tdi_nc; j_out++)
		{
			totalDiff = 0;
			int grayHist[256] = { 0 };
			for (int i_in = 0; i_in < sub_nr; i_in++)
			{
				for (int j_in = 0; j_in < sub_nc; j_in++)
				{
					grayHist[grayImg.data[(i_out + i_in)*nc + j_out + j_in]]++;
				}
			}
			for (int k = 0; k < 256; k++)
			{
				diff = grayHist[k] - subHist[k];
				totalDiff += ((diff >> 31) * 2 * diff + diff);
			}
			if (totalDiff < minDiff)
			{
				minDiff = totalDiff;
				*x = i_out;
				*y = j_out;
			}
		}
	}

	return SUB_IMAGE_MATCH_OK;
}

