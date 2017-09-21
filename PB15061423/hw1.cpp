#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
using namespace cv;
using namespace std;
#define IntLargeNum 2147483647
#define IMG_SHOW
//***********1.
int ustc_ConvertBgr2Gray(Mat bgrImg, Mat& grayImg)
{
	if (NULL == bgrImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int i;
	int row = bgrImg.rows;
	int col = bgrImg.cols;
	int size = row*col;
	grayImg=grayImg.Mat::zeros(bgrImg.size(), CV_8UC1);
	uchar b, g, r;
	uchar *pgray = grayImg.data;
	for ( i = 0; i < size; i++)
	{
		b = bgrImg.data[3 * i + 0];
		g = bgrImg.data[3 * i + 1];
		r = bgrImg.data[3 * i + 2];
		pgray[i] = r*0.299 + g*0.587 + b*0.114;
	}
#ifdef IMG_SHOW
	namedWindow("grayImg", 0);
	imshow("grayImg", grayImg);
	waitKey();
#endif
	if (i == size)
		return SUB_IMAGE_MATCH_OK;
	else
		return SUB_IMAGE_MATCH_FAIL;
}
//**********2.
int ustc_CalcGrad(Mat grayImg, Mat& gradImg_x, Mat& gradImg_y)
{
	if (NULL == grayImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int row = grayImg.rows;
	int col = grayImg.cols;
	gradImg_x=gradImg_x.Mat::zeros(grayImg.size(), CV_32FC1);
	gradImg_y=gradImg_y.Mat::zeros(grayImg.size(), CV_32FC1);
	int num = row*col - col * 2-row*2 + 2;
	float *px = (float*)gradImg_x.data ;
	float *py = (float*)gradImg_y.data ;
	uchar *pgray1 = grayImg.data, *pgray2 = pgray1 + 1, *pgray3 = pgray2 + 1, *pgray4 = pgray2 + col, *pgray5 = pgray4 + 1, *pgray6 = pgray5 + 1;
	uchar *pgray7 = pgray4 + col, *pgray8 = pgray7 + 1, *pgray9 = pgray8 + 1;
	int i = 0,j;
	for (i = 1; i < row-1; i++)
	{
		for (j = 1; j < col-1; j++)
		{
			px[i*col + j] = -*pgray1 - 2 * (*pgray4) - *pgray7 + *pgray3 + (*pgray6) * 2 + *pgray9;
			py[i*col + j] = -*pgray1 - (*pgray2) * 2 - *pgray3 + *pgray7 + (*pgray8) * 2 + *pgray9;

			 pgray1++; pgray2++; pgray3++; pgray4++; pgray5++; pgray6++;
			 pgray7++; pgray8++; pgray9++;
		}
		pgray1 += 2; pgray2 += 2; pgray3 += 2;  pgray4 += 2;  pgray5 += 2;
		pgray6 += 2; pgray7 += 2; pgray8 += 2;  pgray9 += 2;
	}
	
#ifdef IMG_SHOW
	Mat gradImg_x_8U(row, col, CV_8UC1);
	//为了方便观察，直接取绝对值
	for (int row_i = 0; row_i < row; row_i++)
	{
		for (int col_j = 0; col_j < col; col_j += 1)
		{
			int val = ((float*)gradImg_x.data)[row_i * col + col_j];
			gradImg_x_8U.data[row_i * col + col_j] = abs(val);
		}
	}

	namedWindow("gradImg_x_8U", 0);
	imshow("gradImg_x_8U", gradImg_x_8U);
	waitKey();
#endif
	if (i == num)
		return SUB_IMAGE_MATCH_OK;
	else
		return SUB_IMAGE_MATCH_FAIL;
}
//***********3.
int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y, Mat& angleImg, Mat& magImg)
{
	if (NULL == gradImg_x.data || NULL == gradImg_y.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int i,j;
	int row = gradImg_x.rows, col = gradImg_x.cols;
	angleImg=angleImg.Mat::zeros(gradImg_x.size(), CV_32FC1);
	magImg=magImg.Mat::zeros(gradImg_x.size(), CV_32FC1);
	float *px = (float*)gradImg_x.data;
	float *py = (float*)gradImg_y.data;
	float *pangle = (float*)angleImg.data;
	float *pmag = (float*)magImg.data;
	int sum = 0,sum0=0;
	for (i = 1; i < row-1; i++)
	{
		for (j = 1; j < col - 1; j++)
		{
			if (px[i*col + j]!=0 ||py[i*col + j]!=0)
			{
				pangle[i*col + j] = atan2(px[i*col + j], py[i*col + j]) * 180 *0.3183;//1/pi=0.3183
				sum0 = (int)pangle[i*col + j];
				sum = (sum0 >> 31) && 1;
				pangle[i*col + j] = pangle[i*col + j] + 360 * sum;
			}
			pmag[i*col+j]=sqrt( py[i*col + j] * py[i*col + j] + px[i*col + j] * px[i*col + j]);
		}
	}
	
#ifdef IMG_SHOW
	Mat angleImg_8U(row, col, CV_8UC1);
	//为了方便观察，进行些许变化
	for (int row_i = 0; row_i < row; row_i++)
	{
		for (int col_j = 0; col_j < col; col_j += 1)
		{
			float angle = ((float*)angleImg.data)[row_i * col + col_j];
			angle *= 180 / CV_PI;
			angle += 180;
			//为了能在8U上显示，缩小到0-180之间
			angle /= 2;
			angleImg_8U.data[row_i * col + col_j] = angle;
		}
	}

	namedWindow("gradImg_x_8U", 0);
	imshow("gradImg_x_8U", angleImg_8U);
	waitKey();
#endif
	if (i == row - 1 && j == col - 1)
		return SUB_IMAGE_MATCH_OK;
	else
		return SUB_IMAGE_MATCH_FAIL;
}
//************4.
int ustc_Threshold(Mat grayImg, Mat& binaryImg, int th)
{
	if (NULL == grayImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int i;
	int row = grayImg.rows;
	int col = grayImg.cols;
	int size = row*col;
	binaryImg=binaryImg.Mat::zeros(grayImg.size(), CV_8UC1);
	uchar *pgray = grayImg.data;
	uchar *pbinary = binaryImg.data;
	for (i = 0; i < size; i++)
	{
		pbinary[i] = (((th - (char)pgray[i]) >> 7) && 1) * 255;
		/*	if (pgray[i] >= th)
		pdouble[i] = 255;
		else
		pdouble[i] = 0;
		*/
	}
	
#ifdef IMG_SHOW
	namedWindow("binaryImg", 0);
	imshow("binaryImg", binaryImg);
	waitKey();
#endif
	if (i == size)
		return SUB_IMAGE_MATCH_OK;
	else
		return SUB_IMAGE_MATCH_FAIL;
}
//************5.
int ustc_CalcHist(Mat grayImg, int* hist, int hist_len)
{
	if (NULL == grayImg.data || NULL == hist)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int i;
	int row = grayImg.rows;
	int col = grayImg.cols;
	int size = row*col;
	uchar *pgray = grayImg.data;
	for ( i = 0; i < hist_len; i++)
	{
		hist[i] = 0;
	}
	for (i = 0; i < size; i++)
	{
		hist[pgray[i]]++;
	}
	if (i == size)
		return SUB_IMAGE_MATCH_OK;
	else
		return SUB_IMAGE_MATCH_FAIL;
}
//***********6.
int ustc_SubImgMatch_gray(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (grayImg.rows < subImg.rows || grayImg.cols < subImg.cols)
	{
		cout << "image is false." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (x == NULL || NULL == y)
	{
		x = new int(0);
		y = new int(0);
	}
	int i, j, sub_i, sub_j;
	uchar *pgray = grayImg.data;
	uchar *psub = subImg.data;
	int rowf = grayImg.rows;
	int colf = grayImg.cols;
	int rows = subImg.rows;
	int cols = subImg.cols;
	int diff[2] = { 0,255 * rows*cols };
	int sum = 0;
	int fx=0, fy=0;
	int xif = 0;
	int xis = 0;
	int rowlim = rowf - rows + 1;
	int collim = colf - cols + 1;
	int res=cols%8,mod5j=cols-res;
	for (i = 0; i < rowlim; i++)
	{
		for (j = 0; j < collim; j++)
		{
			diff[0] = 0;
			for (sub_i = 0; sub_i < rows; sub_i++)
			{
				 xif = (i + sub_i)*colf+j;
				 xis = sub_i*cols;
				for (sub_j = 0; sub_j < mod5j; sub_j=sub_j+8)
				{
					int numf = xif + sub_j;
					int nums = xis + sub_j;
					sum = pgray[numf+0] - psub[nums+0];
					sum = (sum >> 31)*(sum<<1) + sum;
					diff[0] = diff[0] + sum;
				
					sum = pgray[numf+1] - psub[nums+1];
					sum = (sum >> 31)*(sum << 1) + sum;
					diff[0] = diff[0] + sum;
					
					sum = pgray[numf+2] - psub[nums+2];
					sum = (sum >> 31)*(sum << 1) + sum;
					diff[0] = diff[0] + sum;
					
					sum = pgray[numf+3] - psub[nums+3];
					sum = (sum >> 31)*(sum << 1) + sum;
					diff[0] = diff[0] + sum;
					
					sum = pgray[numf+4] -psub[nums+4];
					sum = (sum >> 31)*(sum << 1) + sum;
					diff[0] = diff[0] + sum;

					sum = pgray[numf + 5] - psub[nums + 4];
					sum = (sum >> 31)*(sum << 1) + sum;
					diff[0] = diff[0] + sum;

					sum = pgray[numf + 7] - psub[nums + 4];
					sum = (sum >> 31)*(sum << 1) + sum;
					diff[0] = diff[0] + sum;

					sum = pgray[numf + 6] - psub[nums + 4];
					sum = (sum >> 31)*(sum << 1) + sum;
					diff[0] = diff[0] + sum;
				}
				for (; sub_j < res; sub_j++)
				{
					sum = (int)pgray[xif + sub_j] - (int)psub[xis + sub_j];
					sum = (sum >> 31)*(sum << 1) + sum;
					diff[0] = diff[0] + sum;
				}
			}
			if (diff[0] < diff[1])
			{
				diff[1] = diff[0];
				fx = j;
				fy = i;
			}

		}
	}
	if (diff[1] == 255 * rows*cols)
		return SUB_IMAGE_MATCH_FAIL;

	*x = fx;
	*y = fy;
	if (i == rowlim && j == collim)		
		return SUB_IMAGE_MATCH_OK;
	else
		return SUB_IMAGE_MATCH_FAIL;
	
}
//***********7
int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int* x, int* y)
{
	if (NULL == colorImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (colorImg.rows < subImg.rows || colorImg.cols < subImg.cols)
	{
		cout << "image is false." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (x == NULL || NULL == y)
	{
		x = new int(0);
		y = new int(0);
	}
	int i_f, j_f, i_sub, j_sub;
	uchar *pcolor = colorImg.data;
	uchar *psub = subImg.data;
	int row_f = colorImg.rows;
	int col_f = colorImg.cols;
	int row_s = subImg.rows;
	int col_s = subImg.cols;
	int diff[2] = { 0,255 * row_s*col_s };
	int sum = 0;
	int fx = 0, fy = 0; 
	int rowlim = row_f - row_s + 1;
	int collim = col_f - col_s + 1;
	int xif = 0, xis = 0, xij = 0;
	int res = col_s % 8, mod5j = col_s - res;
	for (i_f = 0; i_f < rowlim; i_f++)
	{

		for (j_f = 0; j_f < collim; j_f++)
		{
			diff[0] = 0;
			for (i_sub = 0; i_sub < row_s; i_sub++)
			{
				xif = ((i_f + i_sub)*col_f + j_f) * 3;
				xis = i_sub*col_s * 3;
				for (j_sub = 0; j_sub < mod5j;j_sub=j_sub+8 )
				{
					sum = 0;
					xij = j_sub * 3;
					int numf = xif + xij;
					int nums = xis + xij;
					int sum0 = pcolor[numf+0 + 0] - psub[nums + 0 + 0];
					sum0 = (sum0 >> 31) *(sum0 << 1) + sum0;
					int sum1 = pcolor[numf+0 + 1] - psub[nums + 0 + 1];
					sum1 = (sum1 >> 31) *(sum1 << 1) + sum1;
					int sum2 = pcolor[numf+0 + 2] - psub[nums + 0 + 2];
					sum2 = (sum2 >> 31) *(sum2 << 1) + sum2;
					sum = sum0 + sum1 + sum2;
					diff[0] += sum;

					sum = 0;sum0 = 0;
					sum1 = 0; sum2 = 0;

					 sum0 = pcolor[numf+3 + 0] - psub[nums + 3 + 0];
					sum0 = (sum0 >> 31) *(sum0 << 1) + sum0;
					 sum1 = pcolor[numf +3+ 1] - psub[nums + 3+ 1];
					sum1 = (sum1 >> 31) *(sum1 << 1) + sum1;
					 sum2 = pcolor[numf +3+ 2] - psub[nums + 3 + 2];
					sum2 = (sum2 >> 31) *(sum2 << 1) + sum2;
					sum = sum0 + sum1 + sum2;
					diff[0] += sum;

					sum = 0; sum0 = 0;
					sum1 = 0; sum2 = 0;
	
					 sum0 = pcolor[numf +6+ 0] - psub[nums + 6 + 0];
					sum0 = (sum0 >> 31) *(sum0 << 1) + sum0;
					 sum1 = pcolor[numf +6+ 1] - psub[nums + 6 + 1];
					sum1 = (sum1 >> 31) *(sum1 << 1) + sum1;
					 sum2 = pcolor[numf +6+ 2] - psub[nums + 6 + 2];
					sum2 = (sum2 >> 31) *(sum2 << 1) + sum2;
					sum = sum0 + sum1 + sum2;
					diff[0] += sum;
				
					sum = 0; sum0 = 0;
					sum1 = 0; sum2 = 0;
			
					 sum0 = pcolor[numf+9 + 0] - psub[nums + 9 + 0];
					sum0 = (sum0 >> 31) *(sum0 << 1) + sum0;
					 sum1 = pcolor[numf +9+ 1] - psub[nums + 9 + 1];
					sum1 = (sum1 >> 31) *(sum1 << 1) + sum1;
					 sum2 = pcolor[numf +9+ 2] - psub[nums + 9 + 2];
					sum2 = (sum2 >> 31) *(sum2 << 1) + sum2;
					sum = sum0 + sum1 + sum2;
					diff[0] += sum;
				
					sum = 0; sum0 = 0;
					sum1 = 0; sum2 = 0;
	
					 sum0 = pcolor[numf +12+ 0] - psub[nums + 12 + 0];
					sum0 = (sum0 >> 31) *(sum0 << 1) + sum0;
					 sum1 = pcolor[numf +12+ 1] - psub[nums + 12+ 1];
					sum1 = (sum1 >> 31) *(sum1 << 1) + sum1;
					 sum2 = pcolor[numf +12+ 2] - psub[nums + 12 + 2];
					sum2 = (sum2 >> 31) *(sum2 << 1) + sum2;
					sum = sum0 + sum1 + sum2;
					diff[0] += sum;
					
					sum0 = pcolor[numf + 15 + 0] - psub[nums + 15 + 0];
					sum0 = (sum0 >> 31) *(sum0 << 1) + sum0;
					sum1 = pcolor[numf + 15 + 1] - psub[nums + 15 + 1];
					sum1 = (sum1 >> 31) *(sum1 << 1) + sum1;
					sum2 = pcolor[numf + 15 + 2] - psub[nums + 15 + 2];
					sum2 = (sum2 >> 31) *(sum2 << 1) + sum2;
					sum = sum0 + sum1 + sum2;
					diff[0] += sum;

					sum0 = pcolor[numf + 18 + 0] - psub[nums + 18 + 0];
					sum0 = (sum0 >> 31) *(sum0 << 1) + sum0;
					sum1 = pcolor[numf + 18 + 1] - psub[nums + 18 + 1];
					sum1 = (sum1 >> 31) *(sum1 << 1) + sum1;
					sum2 = pcolor[numf + 18 + 2] - psub[nums + 18 + 2];
					sum2 = (sum2 >> 31) *(sum2 << 1) + sum2;
					sum = sum0 + sum1 + sum2;
					diff[0] += sum;

					sum0 = pcolor[numf + 21 + 0] - psub[nums + 21 + 0];
					sum0 = (sum0 >> 31) *(sum0 << 1) + sum0;
					sum1 = pcolor[numf + 21 + 1] - psub[nums + 21 + 1];
					sum1 = (sum1 >> 31) *(sum1 << 1) + sum1;
					sum2 = pcolor[numf + 21 + 2] - psub[nums + 21 + 2];
					sum2 = (sum2 >> 31) *(sum2 << 1) + sum2;
					sum = sum0 + sum1 + sum2;
					diff[0] += sum;
				}
				for (; j_sub < res; j_sub++)
				{
					sum = 0;
					xij = j_sub * 3;
					int sum0 = pcolor[xif + xij + 0] - psub[xis + xij + 0];
					sum0 = (sum0 >> 31) *(sum0 << 1) + sum0;
					int sum1 = pcolor[xif + xij + 1] - psub[xis + xij + 1];
					sum1 = (sum1 >> 31) *(sum1 << 1) + sum1;
					int sum2 = pcolor[xif + xij + 2] - psub[xis + xij + 2];
					sum2 = (sum2 >> 31) *(sum2 << 1) + sum2;
					sum = sum0 + sum1 + sum2;
					diff[0] += sum;
					
				}
			}
			if (diff[0] < diff[1])
			{
				diff[1] = diff[0];
				fx = j_f;
				fy = i_f;
			}
		}
	}
	if (diff[1] == 255 * row_s*col_s)
		return SUB_IMAGE_MATCH_FAIL;

	*x = fx;
	*y = fy;
	if (i_f == rowlim && j_f == collim)
		return SUB_IMAGE_MATCH_OK;
	else
		return SUB_IMAGE_MATCH_FAIL;
}
//*************8.
int ustc_SubImgMatch_corr(Mat grayImg, Mat subImg, int* x, int* y)//偏差
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (grayImg.rows < subImg.rows || grayImg.cols < subImg.cols)
	{
		cout << "image is false." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (x == NULL || NULL == y)
	{
		x = new int(0);
		y = new int(0);
	}
	int row_f = grayImg.rows;
	int col_f = grayImg.cols;
	int row_s = subImg.rows;
	int col_s = subImg.cols;
	int fx = 0, fy = 0;
	long int gray_sqsum = 0;
	long int sub_sqsum = 0;
	long int gray_sub_sum = 0;
	float corr=0;
	int diff[2] = { 0 };
	int rowlim = row_f - row_s + 1;
	int collim = col_f - col_s + 1;
	int i_f, j_f, i_sub, j_sub;
	uchar *pgray = grayImg.data;
	uchar *psub = subImg.data;
	int res = col_s % 5, mod5j = col_s - res;
	for (i_sub = 0; i_sub < row_s; i_sub++)
	{
		int xis = i_sub*col_s;
		for (j_sub = 0; j_sub < col_s; j_sub++)
		{
			sub_sqsum += psub[xis + j_sub] * psub[xis + j_sub];
		}
	}
	for (i_f = 0; i_f < rowlim; i_f++)
	{

		for (j_f = 0; j_f < collim; j_f++)
		{
			corr = 0;
			gray_sqsum = 0;
			gray_sub_sum = 0;
			for (i_sub = 0; i_sub < row_s; i_sub++)
			{
				int xif = (i_f + i_sub)*col_f + j_f;
				int xis1 = i_sub*col_s;
				for (j_sub = 0; j_sub < mod5j; j_sub+=5)
				{
					int numf = xif + j_sub;
					int nums = xis1 + j_sub;
					int pgray_ij = pgray[numf + 0];
					gray_sqsum += pgray_ij * pgray_ij;
					gray_sub_sum += pgray_ij * psub[nums +0];

					 pgray_ij = pgray[numf + 1];
					gray_sqsum += pgray_ij * pgray_ij;
					gray_sub_sum += pgray_ij * psub[nums + 1];

					 pgray_ij = pgray[numf + 2];
					gray_sqsum += pgray_ij * pgray_ij;
					gray_sub_sum += pgray_ij * psub[nums + 2];

					 pgray_ij = pgray[numf + 3];
					gray_sqsum += pgray_ij * pgray_ij;
					gray_sub_sum += pgray_ij * psub[nums +3];

					 pgray_ij = pgray[numf + 4];
					gray_sqsum += pgray_ij * pgray_ij;
					gray_sub_sum += pgray_ij * psub[nums + 4];
				}
				for (; j_sub < res; j_sub++)
				{
					gray_sqsum += pgray[xif+j_sub] * pgray[xif + j_sub];
					gray_sub_sum += pgray[xif + j_sub] * psub[xis1 + j_sub];
				}
			}
			corr = (gray_sub_sum) / (sqrt(gray_sqsum)*sqrt(sub_sqsum));
			diff[0] = corr * 1000;
			if (diff[0] > diff[1])
			{
				diff[1] = diff[0];
				fx = j_f;
				fy = i_f;
			}
		}
	}
	if (diff[1] == 0)
		return SUB_IMAGE_MATCH_FAIL;

	*x = fx;
	*y = fy;
	if (i_f == rowlim && j_f == collim)
		return SUB_IMAGE_MATCH_OK;
	else
		return SUB_IMAGE_MATCH_FAIL;
}
//************9.
int ustc_SubImgMatch_angle(Mat grayImg, Mat subImg, int* x, int* y)//偏差
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (grayImg.rows < subImg.rows || grayImg.cols < subImg.cols)
	{
		cout << "image is false." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (x == NULL || NULL == y)
	{
		x = new int(0);
		y = new int(0);
	}
	int i_f, j_f, i_sub, j_sub;

	//calculate the angle
	Mat gray_x = Mat::zeros(grayImg.size(), CV_32FC1);
	Mat gray_y = Mat::zeros(grayImg.size(), CV_32FC1);
	Mat gray_angle = Mat::zeros(grayImg.size(), CV_32FC1);
	Mat gray_mag = Mat::zeros(grayImg.size(), CV_32FC1);
	ustc_CalcGrad(grayImg, gray_x, gray_y);
	ustc_CalcAngleMag(gray_x, gray_y, gray_angle, gray_mag);

	Mat sub_x = Mat::zeros(subImg.size(), CV_32FC1);
	Mat sub_y = Mat::zeros(subImg.size(), CV_32FC1);
	Mat sub_angle = Mat::zeros(subImg.size(), CV_32FC1);
	Mat sub_mag = Mat::zeros(subImg.size(), CV_32FC1);
	ustc_CalcGrad(subImg, sub_x, sub_y);
	ustc_CalcAngleMag(sub_x, sub_y, sub_angle, sub_mag);

	int row_f = grayImg.rows;
	int col_f = grayImg.cols;
	int row_s = subImg.rows;
	int col_s = subImg.cols;
	int diff[2] = { 0,IntLargeNum };
	int sum = 0;
	int fx = 0, fy = 0;
	float *pgray =(float*) gray_angle.data;
	float *psub = (float*)sub_angle.data;
	int rowlim = row_f - row_s + 1;
	int collim = col_f - col_s + 1;
	int res = col_s % 8, mod5j = col_s - res;
	//match
	for (i_f = 0; i_f < rowlim; i_f++)
	{

		for (j_f = 0; j_f < collim; j_f++)
		{
			diff[0] = 0;
			for (i_sub = 1; i_sub < row_s-1; i_sub++)
			{
				int xif = (i_f + i_sub)*col_f + j_f;
				int xis = i_sub*col_s;
				for (j_sub = 1; j_sub < mod5j; j_sub=8+j_sub)
				{
					int numf = xif + j_sub;
					int nums = xis + j_sub;
					sum = pgray[numf+0] - psub[nums+0];
					sum = (sum >> 31) *(sum<<1) + sum;
					int sum0 = 180 - sum;
					sum = (sum0 >> 31) *((sum << 1) - 360) + sum;
					diff[0] += sum;

					sum = pgray[numf + 1] - psub[nums + 1];
					sum = (sum >> 31) *(sum << 1) + sum;
					 sum0 = 180 - sum;
					sum = (sum0 >> 31) *((sum << 1) - 360) + sum;
					diff[0] += sum;

					sum = pgray[numf + 2] - psub[nums + 2];
					sum = (sum >> 31) *(sum << 1) + sum;
					 sum0 = 180 - sum;
					sum = (sum0 >> 31) *((sum << 1) - 360) + sum;
					diff[0] += sum;

					sum = pgray[numf + 3] - psub[nums + 3];
					sum = (sum >> 31) *(sum << 1) + sum;
					 sum0 = 180 - sum;
					sum = (sum0 >> 31) *((sum << 1) - 360) + sum;
					diff[0] += sum;

					sum = pgray[numf + 4] - psub[nums + 4];
					sum = (sum >> 31) *(sum << 1) + sum;
					 sum0 = 180 - sum;
					sum = (sum0 >> 31) *((sum << 1) - 360) + sum;
					diff[0] += sum;

					sum = (int)pgray[numf + 5] - (int)psub[nums + 5];
					sum = (sum >> 31) *(sum << 1) + sum;
					sum0 = 180 - sum;
					sum = (sum0 >> 31) *((sum << 1) - 360) + sum;
					diff[0] += sum;

					sum = pgray[numf + 6] - psub[nums + 6];
					sum = (sum >> 31) *(sum << 1) + sum;
					sum0 = 180 - sum;
					sum = (sum0 >> 31) *((sum << 1) - 360) + sum;
					diff[0] += sum;

					sum = pgray[numf + 7] - psub[nums + 7];
					sum = (sum >> 31) *(sum << 1) + sum;
					sum0 = 180 - sum;
					sum = (sum0 >> 31) *((sum << 1) - 360) + sum;
					diff[0] += sum;
				}
				for (; j_sub < res; j_sub++)
				{
					sum = (int)pgray[xif + j_sub] - (int)psub[xis + j_sub];
					sum = (sum >> 31) *(sum << 1) + sum;
					int sum0 = 180 - sum;
					sum = (sum0 >> 31) *((sum << 1) - 360) + sum;
					diff[0] += sum;
				}
			}
			if (diff[0] < diff[1])
			{
				diff[1] = diff[0];
				fx = j_f;
				fy = i_f;
			}
		}
	}
	if (diff[1] == IntLargeNum)
		return SUB_IMAGE_MATCH_FAIL;

	*x = fx;
	*y = fy;
	if (i_f == rowlim && j_f == collim)
		return SUB_IMAGE_MATCH_OK;
	else
		return SUB_IMAGE_MATCH_FAIL;
}
//***********10.
int ustc_SubImgMatch_mag(Mat grayImg, Mat subImg, int* x, int* y)//偏差
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (grayImg.rows < subImg.rows || grayImg.cols < subImg.cols)
	{
		cout << "image is false." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (x == NULL || NULL == y)
	{
		x = new int(0);
		y = new int(0);
	}
	int i_f, j_f, i_sub, j_sub;

	//calculate the mag
	Mat gray_x = Mat::zeros(grayImg.size(), CV_32FC1);
	Mat gray_y = Mat::zeros(grayImg.size(), CV_32FC1);
	Mat gray_angle = Mat::zeros(grayImg.size(), CV_32FC1);
	Mat gray_mag = Mat::zeros(grayImg.size(), CV_32FC1);
	ustc_CalcGrad(grayImg, gray_x, gray_y);
	ustc_CalcAngleMag(gray_x, gray_y, gray_angle, gray_mag);

	Mat sub_x = Mat::zeros(subImg.size(), CV_32FC1);
	Mat sub_y = Mat::zeros(subImg.size(), CV_32FC1);
	Mat sub_angle = Mat::zeros(subImg.size(), CV_32FC1);
	Mat sub_mag = Mat::zeros(subImg.size(), CV_32FC1);
	ustc_CalcGrad(subImg, sub_x, sub_y);
	ustc_CalcAngleMag(sub_x, sub_y, sub_angle, sub_mag);

	int row_f = grayImg.rows;
	int col_f = grayImg.cols;
	int row_s = subImg.rows;
	int col_s = subImg.cols;
	int diff[2] = { 0,IntLargeNum };
	int sum = 0;
	int fx = 0, fy = 0;
	float *pgray = (float*)gray_mag.data;
	float *psub = (float*)sub_mag.data;
	int rowlim = row_f - row_s + 1;
	int collim = col_f - col_s + 1;
	int res = col_s % 5, mod5j = col_s - res;
	//match
	for (i_f = 0; i_f < rowlim; i_f++)
	{

		for (j_f = 0; j_f < collim; j_f++)
		{
			diff[0] = 0;
			for (i_sub = 0; i_sub < row_s-1; i_sub++)
			{
				int xif = (i_f + i_sub)*col_f + j_f;
				int xis = i_sub*col_s;
				for (j_sub = 0; j_sub < mod5j; j_sub+=5)
				{
					int numf = xif + j_sub;
					int nums = xis + j_sub;
					sum = (int)pgray[numf+0] - psub[nums+0];
					sum = (sum >> 31)*(sum<<1) + sum;
					diff[0] += sum;

					sum = (int)pgray[numf + 1] - psub[nums + 1];
					sum = (sum >> 31)*(sum << 1) + sum;
					diff[0] += sum;

					sum = (int)pgray[numf + 2] - psub[nums + 2];
					sum = (sum >> 31)*(sum << 1) + sum;
					diff[0] += sum;

					sum = (int)pgray[numf + 3] - psub[nums + 3];
					sum = (sum >> 31)*(sum << 1) + sum;
					diff[0] += sum;

					sum = (int)pgray[numf + 4] - psub[nums + 4];
					sum = (sum >> 31)*(sum << 1) + sum;
					diff[0] += sum;
				}
				for (; j_sub < res; j_sub++)
				{
					sum = (int)pgray[xif + j_sub] - psub[xis + j_sub];
					sum = (sum >> 31)*(sum << 1) + sum;
					diff[0] += sum;
				}
			}
			if (diff[0] < diff[1])
			{
				diff[1] = diff[0];
				fx = j_f;
				fy = i_f;
			}
		}
	}
	if (diff[1] == IntLargeNum)
		return SUB_IMAGE_MATCH_FAIL;

	*x = fx;
	*y = fy;
	if (i_f == rowlim && j_f == collim)
		return SUB_IMAGE_MATCH_OK;
	else
		return SUB_IMAGE_MATCH_FAIL;
}
//***********11.
int ustc_SubImgMatch_hist(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (grayImg.rows < subImg.rows || grayImg.cols < subImg.cols)
	{
		cout << "image is false." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (x == NULL || NULL == y)
	{
		x = new int(0);
		y = new int(0);
	}
	uchar *pgray = grayImg.data;
	int row_f = grayImg.rows;
	int col_f = grayImg.cols;
	int row_s = subImg.rows;
	int col_s = subImg.cols;
	int size_sub = row_s*col_s;
	int rowlim = row_f - row_s + 1;
	int collim = col_f - col_s + 1;
	int subHist[256] = { 0 };
	int diff[2] = { 0,IntLargeNum };
	int sum,fx=0,fy=0;
	ustc_CalcHist(subImg, subHist, 256);
	int i_f, j_f, i_sub,j_sub;
	for (i_f = 0; i_f < rowlim; i_f++)
	{
		for (j_f = 0; j_f < collim; j_f++)
		{
			diff[0] = 0;
			int grayHist[256] = { 0 };
			for (i_sub = 0; i_sub < row_s; i_sub++)
			{
				int xif = (i_f + i_sub)*col_f + j_f;
				int xis = i_sub*col_s;
				for (j_sub = 0; j_sub < col_s; j_sub++)
				{
					grayHist[pgray[xif+ j_sub]]++;
				}
			}
			for (i_sub = 0; i_sub < 256; i_sub++)
			{
				sum = grayHist[i_sub] - subHist[i_sub];
				sum = (sum >> 31)*( sum<<1) + sum;
				diff[0] += sum;
			}
			if (diff[0] < diff[1])
			{
				diff[1] = diff[0];
				fx = j_f;
				fy = i_f;
			}
		}
	}
	if (diff[1] == IntLargeNum)
		return SUB_IMAGE_MATCH_FAIL;
	*x = fx;
	*y = fy;
	if (i_f == rowlim&& j_f == collim)
		return SUB_IMAGE_MATCH_OK;
	else
		return SUB_IMAGE_MATCH_FAIL;
}
