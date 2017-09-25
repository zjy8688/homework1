#include "SubImageMatch.h"
int ustc_ConvertBgr2Gray(Mat bgrImg, Mat& grayImg)
{
	if (bgrImg.data == nullptr)  return SUB_IMAGE_MATCH_FAIL;
	if (bgrImg.type() != CV_8UC3) return SUB_IMAGE_MATCH_FAIL;
	int R = int(0.299f*(1 << 16));
	int G = int(0.587f*(1 << 16));
	int B = int(0.114f*(1 << 16));
	int i, j,col,row;
	row = bgrImg.rows;
	col = bgrImg.cols;
	grayImg=Mat(row, col, CV_8UC1);
	if (bgrImg.isContinuous())
	{
		col *= row;
		row = 1;
	}	
	unsigned char *p, *q;
	for (i = 0; i < row; i++)
	{
		q = bgrImg.ptr<uchar>(i);
		p = grayImg.ptr<uchar>(i);
		for (j = col - 1; j >= 0; j--)
		{
			*(p++) = (unsigned char)((*(q++)*B + *(q++)*G + *(q++)*R) >> 16);
		}
	}
	return SUB_IMAGE_MATCH_OK;
}

int ustc_CalcGrad(Mat grayImg, Mat& gradImg_x, Mat& gradImg_y)
{
	int i, j, col, row;	
	if (grayImg.data == nullptr)  return SUB_IMAGE_MATCH_FAIL;
	if (grayImg.type() != CV_8UC1) return SUB_IMAGE_MATCH_FAIL;
	row = grayImg.rows;
	col = grayImg.cols;
	gradImg_x=Mat(row, col, CV_32FC1);
	gradImg_y=Mat(row, col, CV_32FC1);	
	unsigned char *q, *q1, *q2;
	float *p, *p1;
	p= gradImg_x.ptr<float>(0);
	p1= gradImg_y.ptr<float>(0);
	p++;p1++;
	
	for (i = 1;i < col - 1;i++)
	{
		*(p) = 0;
		*(p1) = 0;
			p++;p1++;
	}
	p = gradImg_x.ptr<float>(row-1);
	p1 = gradImg_y.ptr<float>(row-1);
	p++;p1++;
	for (i = 1;i < col - 1;i++)
	{
		*(p++) =0;
		*(p1++) =0;
	}
	for (i = 1;i < row - 1;i++)
	{
		p = gradImg_x.ptr<float>(i);
		p1 = gradImg_y.ptr<float>(i);
		*p = 0;
		*p1 = 0;
		p += col - 1;
		p1 += col - 1;
		*p = 0;
		*p1 = 0;
	}
	p = gradImg_x.ptr<float>(0);
	p1 = gradImg_y.ptr<float>(0);
	*p = 0;
	*p1= 0;
	p += col - 1;
	p1 += col - 1;
	*p = 0;
	*p1 =0;
	p = gradImg_x.ptr<float>(row-1);
	p1 = gradImg_y.ptr<float>(row-1);
	*p = 0;
	*p1 =0;
	p += col - 1;
	p1 += col - 1;
	*p =0;
	*p1 = 0;

	for (i = 1;i < row - 1;i++)	
	{
		q = grayImg.ptr<uchar>(i - 1);
		q1 = grayImg.ptr<uchar>(i);
		q2 = grayImg.ptr<uchar>(i + 1);
		p = gradImg_x.ptr<float>(i);
		p1 = gradImg_y.ptr<float>(i);
		q++;q1++;q2++;p++;p1++;
		for (j = 1;j < col - 1;j++)
		{
			*p = (float)((*(q + 1) + *(q1 + 1) * 2 + *(q2 + 1)) - (*(q - 1) + *(q1 - 1) * 2 + *(q2 - 1)));
			*p1 = (float)((*(q2 + 1) + (*q2) * 2 + *(q2 - 1)) - (*(q - 1) + (*q) * 2 + *(q + 1)));
			q++;q1++;q2++;p++;p1++;
		}
	}

	return SUB_IMAGE_MATCH_OK;

}

int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y, Mat& angleImg, Mat& magImg)
{
	if (gradImg_x.data == nullptr)  return SUB_IMAGE_MATCH_FAIL;
	if (gradImg_x.type() != CV_32FC1) return SUB_IMAGE_MATCH_FAIL;
	if (gradImg_y.data == nullptr)  return SUB_IMAGE_MATCH_FAIL;
	if (gradImg_y.type() != CV_32FC1) return SUB_IMAGE_MATCH_FAIL;
	int i, j, col, row;
	row = gradImg_x.rows;
	col = gradImg_x.cols;
	angleImg=Mat(row, col, CV_32FC1);
	magImg=Mat(row, col, CV_32FC1);
	float *q, *q1, k, k1, x, a, s, r, o1, o2;
	float *p, *p1;
	long ip;
	float x2, y,number;
	const float threehalfs = 1.5F;
	for (i = row - 1;i >= 0;i--)
	{
		q = gradImg_x.ptr<float>(i);
		q1 = gradImg_y.ptr<float>(i);
		p = angleImg.ptr<float>(i);
		p1 = magImg.ptr<float>(i);
		for (j = col - 1;j >= 0;j--)
		{
			k = *(q++);
			k1 = *(q1++);
			number = (float)(k*k + k1*k1);
			x2 = number * 0.5F;
			y = number;
			ip = *(long *)&y;            
			ip = 0x5f3759df - (ip >> 1); 
			y = *(float *)&ip;
			*(p1++) = y * (threehalfs - (x2 * y * y));
			x= k;
			y = k1;
			if (x < 0)x = -x;
			if (y < 0)y = -y;
			o1 = y;
			if (x < y)o1 = x;
			o2 = x + y - o1;
			a = o1 / (o2 + (float)DBL_EPSILON);
			s = a*a;
			r = ((-0.0464964749 * s + 0.15931422) * s - 0.327622764) * s * a + a;
			if (y > x) r = 1.57079637 - r;
			if (k< 0) r = 3.14159274f - r;
			if (k1 < 0) r = 6.28318548f - r;
			*(p++) = 180.0f*r / 3.1415926f;
		}
	}	
	return SUB_IMAGE_MATCH_OK;
}
int ustc_Threshold(Mat grayImg, Mat& binaryImg, int th)
{
	if (grayImg.data == nullptr)  return SUB_IMAGE_MATCH_FAIL;
	if (grayImg.type() != CV_8UC1) return SUB_IMAGE_MATCH_FAIL;
	int i, j, col, row;
	row = grayImg.rows;
	col = grayImg.cols;
	binaryImg=Mat(row, col, CV_8UC1);
	unsigned char *q, *p;
	for (i = row - 1;i >= 0;i--)
	{
		q = grayImg.ptr<uchar>(i);
		p = binaryImg.ptr<uchar>(i);
		for (j = col - 1;j >= 0;j--)
		{
			*(p++) = (((th - *(q++)) >> 31) & 1) * 255;
		}
	}
	return SUB_IMAGE_MATCH_OK;
	
}
int ustc_CalcHist(Mat grayImg, int* hist, int hist_len)
{
	if (grayImg.data == nullptr)  return SUB_IMAGE_MATCH_FAIL;
	if (grayImg.type() != CV_8UC1) return SUB_IMAGE_MATCH_FAIL;
	if (hist == nullptr)  return SUB_IMAGE_MATCH_FAIL;
	int i, j, col, row;
	row = grayImg.rows;
	col = grayImg.cols;
	unsigned char *q;
	memset(hist, 0, sizeof(int) * hist_len);	
	for (i = 0;i < row;i++)
	{
		q = grayImg.ptr<uchar>(i);
		for (j = 0;j < col;j++)
		{
			*(hist + *(q++)) += 1;
		}
	}
	return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_gray(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (grayImg.data == nullptr)  return SUB_IMAGE_MATCH_FAIL;
	if (grayImg.type() != CV_8UC1) return SUB_IMAGE_MATCH_FAIL;
	if (x == nullptr)  return SUB_IMAGE_MATCH_FAIL;
	if (y == nullptr)  return SUB_IMAGE_MATCH_FAIL;
	int i, j, ii, jj, col, row, col1, row1;
	long min, now;
	int op;
	unsigned char *q, *p;
	row = grayImg.rows;
	col = grayImg.cols;
	row1 = subImg.rows;
	col1 = subImg.cols;
	if(row<row1||col<col1) return SUB_IMAGE_MATCH_FAIL;
	min = row1*col1 * 256; 
	for (i = 0;i < row - row1 + 1;i++)			
		for (j = 0;j < col - col1 + 1;j++)
		{
			now = 0;
			for (ii = 0;ii < row1;ii++)
			{
				p = subImg.ptr<uchar>(ii);
				q = grayImg.ptr<uchar>(i+ii);
				q += j;
				for (jj = 0;jj < col1;jj++)
				{
					op=*(p++) - *(q++);
					if (op < 0) op = -op;
					now += op;
				}
			}
			if (now < min)
			{
				min = now;
				*x = j;
				*y = i;
			}
		}
	return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int* x, int* y)
{
	if (colorImg.data == nullptr)  return SUB_IMAGE_MATCH_FAIL;
	if (colorImg.type() != CV_8UC3) return SUB_IMAGE_MATCH_FAIL;
	if (subImg.data == nullptr)  return SUB_IMAGE_MATCH_FAIL;
	if (subImg.type() != CV_8UC3) return SUB_IMAGE_MATCH_FAIL;
	if (x == nullptr)  return SUB_IMAGE_MATCH_FAIL;
	if (y == nullptr)  return SUB_IMAGE_MATCH_FAIL;
	int i, j, ii, jj, col, row, col1, row1, op1, op2, op3;
	long min, now;
	unsigned char *q, *p;
	row = colorImg.rows;
	col = colorImg.cols;
	row1 = subImg.rows;
	col1 = subImg.cols;
	if (row<row1 || col<col1) return SUB_IMAGE_MATCH_FAIL;
	min = row1*col1 * 256 * 3;
	for (i = 0;i < row - row1 + 1;i++)
		for (j = 0;j < col - col1 + 1;j++)
		{
			now = 0;
			for (ii = 0;ii < row1;ii++)
			{
				p = subImg.ptr<uchar>(ii);
				q = colorImg.ptr<uchar>(i + ii);
				q += j*3;
				for (jj = 0;jj < col1;jj++)
				{
					op1= *(p++) - *(q++);
					op2= *(p++) - *(q++);
					op3= *(p++) - *(q++);
					if (op1 < 0) op1 = -op1;
					if (op2 < 0) op2 = -op2;
					if (op3 < 0) op3 = -op3;
					now += op1 + op2 + op3;
				}
			}
			if (now < min)
			{
				min = now;
				*x = j;
				*y = i;
			}
		}
	return SUB_IMAGE_MATCH_OK;

}
int ustc_SubImgMatch_corr(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (grayImg.data == nullptr)  return SUB_IMAGE_MATCH_FAIL;
	if (grayImg.type() != CV_8UC1) return SUB_IMAGE_MATCH_FAIL;
	if (subImg.data == nullptr)  return SUB_IMAGE_MATCH_FAIL;
	if (subImg.type() != CV_8UC1) return SUB_IMAGE_MATCH_FAIL;
	if (x == nullptr)  return SUB_IMAGE_MATCH_FAIL;
	if (y == nullptr)  return SUB_IMAGE_MATCH_FAIL;
	int i, j, ii, jj, col, row, col1, row1;
	double min, now;
	long s, r, t;
	unsigned char *q, *p;
	row = grayImg.rows;
	col = grayImg.cols;
	row1 = subImg.rows;
	col1 = subImg.cols;
	if (row < row1 || col < col1) return SUB_IMAGE_MATCH_FAIL;
	min = 10000000000.0;
	for (i = 0;i < row - row1 + 1;i++)
		for (j = 0;j < col - col1 + 1;j++)
		{
			now = 0;
			s = 0;
			r = 0;
			t = 0;
			for (ii = 0;ii < row1;ii++)
			{
				p = subImg.ptr<uchar>(ii);
				q = grayImg.ptr<uchar>(i + ii);
				q += j;
				for (jj = 0;jj < col1;jj++)
				{
					s += (*q)*(*p);
					r += (*q)*(*q);
					t += (*p)*(*p);
					q++;
					p++;
				}
			}
			s = s*s;
			now = (double)s / (double)(r*t);
			now = now - 1.0;
			if (now < 0)now = -now;
			if (now < min)
			{
				min = now;
				*x = j;
				*y = i;
			}
		}
	return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_angle(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (grayImg.data == nullptr)  return SUB_IMAGE_MATCH_FAIL;
	if (grayImg.type() != CV_8UC1) return SUB_IMAGE_MATCH_FAIL;
	if (subImg.data == nullptr)  return SUB_IMAGE_MATCH_FAIL;
	if (subImg.type() != CV_8UC1) return SUB_IMAGE_MATCH_FAIL;
	if (x == nullptr)  return SUB_IMAGE_MATCH_FAIL;
	if (y == nullptr)  return SUB_IMAGE_MATCH_FAIL;
	int i, j, ii, jj, col, row, col1, row1;
	double min, now;
	float *q, *p, angl;
	row = grayImg.rows;
	col = grayImg.cols;
	row1 = subImg.rows;
	col1 = subImg.cols;
	if (row<row1 || col<col1) return SUB_IMAGE_MATCH_FAIL;
	Mat gradImg_x, gradImg_y, angleImg, magImg, subgradImg_x, subgradImg_y, subangleImg, submagImg;
	ustc_CalcGrad(grayImg, gradImg_x, gradImg_y);
	ustc_CalcGrad(subImg, subgradImg_x, subgradImg_y);
	ustc_CalcAngleMag(gradImg_x, gradImg_y, angleImg, magImg);
	ustc_CalcAngleMag(subgradImg_x, subgradImg_y, subangleImg, submagImg);
	min = (float)row1*(float)col1 * 180.0f;
	for (i = 0;i < row - row1 + 1;i++)
		for (j = 0;j < col - col1 + 1;j++)
		{
			now = 0.0;
			for (ii = 0;ii < row1;ii++)
			{
				p = angleImg.ptr<float>(i + ii);
				q = subangleImg.ptr<float>(ii);
				p += j;
				for (jj = 0;jj < col1;jj++)
				{
					angl=*(p++) - *(q++);
					if (angl < 0)angl = -angl;
					if (angl > 180.0f) angl = 360.0f-angl;
					now += angl;
				}
			}
			
			if (now < min)
			{
				min = now;
				*x = j;
				*y = i;
			}
		}
	return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_mag(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (grayImg.data == nullptr)  return SUB_IMAGE_MATCH_FAIL;
	if (grayImg.type() != CV_8UC1) return SUB_IMAGE_MATCH_FAIL;
	if (subImg.data == nullptr)  return SUB_IMAGE_MATCH_FAIL;
	if (subImg.type() != CV_8UC1) return SUB_IMAGE_MATCH_FAIL;
	if (x == nullptr)  return SUB_IMAGE_MATCH_FAIL;
	if (y == nullptr)  return SUB_IMAGE_MATCH_FAIL;
	int i, j, ii, jj, col, row, col1, row1;
	double min, now;
	float *q, *p, op;
	row = grayImg.rows;
	col = grayImg.cols;
	row1 = subImg.rows;
	col1 = subImg.cols;
	if (row < row1 || col < col1) return SUB_IMAGE_MATCH_FAIL;
	Mat gradImg_x, gradImg_y, angleImg, magImg, subgradImg_x, subgradImg_y, subangleImg, submagImg;
	ustc_CalcGrad(grayImg, gradImg_x, gradImg_y);
	ustc_CalcGrad(subImg, subgradImg_x, subgradImg_y);
	ustc_CalcAngleMag(gradImg_x, gradImg_y, angleImg, magImg);
	ustc_CalcAngleMag(subgradImg_x, subgradImg_y, subangleImg, submagImg);
	//return SUB_IMAGE_MATCH_FAIL;
	min = (float)row1*(float)col1 * 2000.0f;
	for (i = 0;i < row - row1 + 1;i++)
		for (j = 0;j < col - col1 + 1;j++)
		{
			now = 0.0;
			for (ii = 1;ii < row1-1;ii++)
			{
				p = submagImg.ptr<float>(ii);
				q = magImg.ptr<float>(i + ii);
				q += j;
				p++;
				q++;
				for (jj = 1;jj < col1-1;jj++)
				{
					op= *(p++) - *(q++);
					if (op < 0) op = -op;
					now += op;
				}
			}
			if (now < min)
			{
				min = now;
				*x = j;
				*y = i;
			}
		}
	return SUB_IMAGE_MATCH_OK;
}
int ustc_SubImgMatch_hist(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (grayImg.data == nullptr)  return SUB_IMAGE_MATCH_FAIL;
	if (grayImg.type() != CV_8UC1) return SUB_IMAGE_MATCH_FAIL;
	if (subImg.data == nullptr)  return SUB_IMAGE_MATCH_FAIL;
	if (subImg.type() != CV_8UC1) return SUB_IMAGE_MATCH_FAIL;
	if (x == nullptr)  return SUB_IMAGE_MATCH_FAIL;
	if (y == nullptr)  return SUB_IMAGE_MATCH_FAIL;
	int i, j, ii, jj, col, row, col1, row1, hist_len = 256, op;
	long min, now;
	unsigned char *q;
	row = grayImg.rows;
	col = grayImg.cols;
	row1 = subImg.rows;
	col1 = subImg.cols;
	if (row<row1 || col<col1) return SUB_IMAGE_MATCH_FAIL;
	min = row1*col1 * hist_len;
	int subhist[256];
	if (ustc_CalcHist(subImg, subhist, hist_len) < 0)
		return SUB_IMAGE_MATCH_FAIL;
	int* hist = new int[hist_len];
	for (i = 0;i < row - row1 + 1;i++)
		for (j = 0;j < col - col1 + 1;j++)
		{
			now = 0;
			memset(hist, 0, sizeof(int) * hist_len);
			for (ii = 0;ii < row1;ii++)
			{
				q = grayImg.ptr<uchar>(i + ii);
				q += j;
				for (jj = 0;jj < col1;jj++)
				{
					hist[*(q++)] += 1;
				}
			}
			for (ii = 0;ii < hist_len-1;ii++)
			{
				op= hist[ii] - subhist[ii];
				if (op < 0) op = -op;
				now += op;
			}
			if (now < min)
			{
				min = now;
				*x = j;
				*y = i;
			}
		}
	return SUB_IMAGE_MATCH_OK;
}
