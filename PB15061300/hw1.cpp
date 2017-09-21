#include "SubImageMatch.h"

//函数功能：将bgr图像转化成灰度图像
//bgrImg：彩色图，像素排列顺序是bgr
//grayImg：灰度图，单通道
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL
int ustc_ConvertBgr2Gray(Mat bgrImg, Mat & grayImg)
{
	if (NULL == grayImg.data || NULL == bgrImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int nRows = bgrImg.rows;
	int nCols = bgrImg.cols;
	uchar *bgr = bgrImg.data, *gray = grayImg.data;
	//int* grayValue = new int[nRows*nCols];
	int length = nRows*nCols;
	for (int i=0; i < length; i++)
	{
		*gray++ =( *bgr++ * 7472 + *bgr++ * 38469 + *bgr++ * 19595) >> 16;
	}
	return SUB_IMAGE_MATCH_OK;
}

//函数功能：根据灰度图像计算梯度图像
//grayImg：灰度图，单通道
//gradImg_x：水平方向梯度，浮点类型图像，CV32FC1
//gradImg_y：垂直方向梯度，浮点类型图像，CV32FC1
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL
int ustc_CalcGrad(Mat grayImg, Mat & gradImg_x, Mat & gradImg_y)
{
	if (NULL == grayImg.data || NULL == gradImg_x.data || NULL == gradImg_y.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int nRows = grayImg.rows;
	int nCols = grayImg.cols;
	uchar *a0 = grayImg.data, *a1 = a0 + nCols, *a2 = a1 + nCols;
	int i, j;
	float *data_x = (float*)gradImg_x.data, *data_y = (float*)gradImg_y.data;
	for (i = nCols; i; --i)
	{
		*data_x++ = 0;
		*data_y++ = 0;
	}
	for (i = nRows-2; i; --i)
	{
		for (j = nCols - 2, *data_x++ = 0, *data_y++ = 0; j; --j)
		{
			*data_x++ = *a2 + 2 * *(a2 + 1) + *(a2 + 2) - *a0 - 2 * *(a0 + 1) - *(a0 + 2);
			*data_y++ = *a0 + 2 * *a1 + *a2 - *(a0 + 2) - 2 * *(a1 + 2) - *(a2 + 2);
			a0++;
			a1++; 
			a2++;
		}
		*data_x++ = 0;
		*data_y++ = 0;
		a0 += 2;
		a1 += 2;
		a2 += 2;
	}
	for (i = nCols; i; --i)
	{
		*data_x++ = 0;
		*data_y++ = 0;
	}
	return SUB_IMAGE_MATCH_OK;
}

//函数功能：根据水平和垂直梯度，计算角度和幅值图
//gradImg_x：水平方向梯度，浮点类型图像，CV32FC1
//gradImg_y：垂直方向梯度，浮点类型图像，CV32FC1
//angleImg：角度图，浮点类型图像，CV32FC1
//magImg：幅值图，浮点类型图像，CV32FC1
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL
int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y, Mat & angleImg, Mat & magImg)
{
	if (NULL == gradImg_x.data || NULL == gradImg_y.data || NULL == angleImg.data || NULL == magImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int nRows = gradImg_x.rows;
	int nCols = gradImg_y.cols;
	int length = nRows*nCols;
	float *angle = (float*)angleImg.data, *mag = (float*)magImg.data;
	float *dx = (float*)gradImg_x.data, *dy = (float*)gradImg_y.data;
	float ang;
	for (int i = 0; i < length; i++)
	{
		ang = atan2(*dy, *dx)*57.3;
		if (ang < 0)
		{
			ang += 360;
		}
		*angle++ = ang;
		*mag++ = sqrt(*dy*(*dy++) + *dx*(*dx++));
	}
	return SUB_IMAGE_MATCH_OK;
}

//函数功能：对灰度图像进行二值化
//grayImg：灰度图，单通道
//binaryImg：二值图，单通道
//th：二值化阈值，高于此值，255，低于此值0
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL
int ustc_Threshold(Mat grayImg, Mat & binaryImg, int th)
{
	if (NULL == grayImg.data || NULL == binaryImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int nRows = grayImg.rows;
	int nCols = grayImg.cols;
	int length = nRows*nCols;
	uchar *gray = grayImg.data, *bin = binaryImg.data;
	for (int i = 0; i < length; i++)
	{
		*bin++ = (((*gray++) + 256 - th) >> 8) * 255;
	}
	return SUB_IMAGE_MATCH_OK;
}

//函数功能：对灰度图像计算直方图
//grayImg：灰度图，单通道
//hist：直方图
//hist_len：直方图的亮度等级，直方图数组的长度
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL
int ustc_CalcHist(Mat grayImg, int * hist, int hist_len)
{
	if (NULL == grayImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int nRows = grayImg.rows;
	int nCols = grayImg.cols;
	int length = nRows*nCols;
	uchar *gray = grayImg.data;
	for (int i = length; i; --i)
	{
		hist[(*gray)-1]++;
		++gray;
	}
	return SUB_IMAGE_MATCH_OK;
}

//函数功能：利用亮度进行子图匹配
//grayImg：灰度图，单通道
//subImg：模板子图，单通道
//x：最佳匹配子图左上角x坐标
//y：最佳匹配子图左上角y坐标
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL
int ustc_SubImgMatch_gray(Mat grayImg, Mat subImg, int * x, int * y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int subRows = subImg.rows;
	int subCols = subImg.cols;
	int nRows = grayImg.rows - subRows + 1;
	int nCols = grayImg.cols - subCols + 1;
	int dCols = grayImg.cols - subCols;
	int finex = 0, finey = 0;
	uchar *sub, *gray = grayImg.data, *gray0 = gray;
	int i, j, k, l;
	int min= 0x7FFFFFFF;
	int sum;
	int diff;
	for (i = 0; i < nRows; ++i)
	{
		for (j = 0; j < nCols; ++j)
		{
			for (sub = subImg.data, gray = gray0, k = subRows, sum = 0; k; --k)
			{
				for (l = subCols; l; --l, ++sub, ++gray)
				{
					diff = *sub - *gray;
					sum+= ((diff >> 31) & 1)*(~diff + 1) + !(diff >> 31)*diff;
				}
				gray += dCols;
				if ((min - sum) >> 31)break;
			}
			min = !((sum - min) >> 31)*min + !((min - sum) >> 31)*sum;
			finex = !(sum^min)*j + ((sum^min) && 1)*finex;
			finey = !(sum^min)*i + ((sum^min)&&1)*finey;
			++gray0;
		}
		gray0 =gray0+ (subCols-1);
	}
	*x = finex;
	*y = finey;
	return SUB_IMAGE_MATCH_OK;
}

//函数功能：利用色彩进行子图匹配
//colorImg：彩色图，三通单
//subImg：模板子图，三通道
//x：最佳匹配子图左上角x坐标
//y：最佳匹配子图左上角y坐标
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL
int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int * x, int * y)
{
	if (NULL == colorImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int subRows = subImg.rows;
	int subCols = subImg.cols * 3;
	int nRows = colorImg.rows - subRows + 1;
	int nCols = colorImg.cols - subImg.cols + 1;
	int dCols = colorImg.cols * 3 - subCols;
	int finex = 0, finey = 0;
	uchar *sub, *gbr = colorImg.data, *gbr0 = gbr;
	int i, j, k, l;
	int min = 0x7FFFFFFF;
	int sum;
	int diff;
	for (i = 0; i < nRows; ++i)
	{
		for (j = 0; j < nCols; ++j)
		{
			for (sub = subImg.data, gbr = gbr0, k = subRows, sum = 0; k; --k)
			{
				for (l = subCols; l; --l, ++sub, ++gbr)
				{
					diff = *sub - *gbr;
					sum += ((diff >> 31) & 1)*(~diff + 1) + !(diff >> 31)*diff;
				}
				gbr += dCols;
				if ((min - sum) >> 31)break;
			}

			min = !((sum - min) >> 31)*min + !((min - sum) >> 31)*sum;
			finex = !(sum^min)*j + ((sum^min) && 1)*finex;
			finey = !(sum^min)*i + ((sum^min) && 1)*finey;
			gbr0 += 3;
		}
		gbr0 = gbr0 + (subCols - 3);
	}
	*x = finex;
	*y = finey;
	return SUB_IMAGE_MATCH_OK;
}

//函数功能：利用亮度相关性进行子图匹配
//grayImg：灰度图，单通道
//subImg：模板子图，单通道
//x：最佳匹配子图左上角x坐标
//y：最佳匹配子图左上角y坐标
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL
int ustc_SubImgMatch_corr(Mat grayImg, Mat subImg, int * x, int * y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int subRows = subImg.rows;
	int subCols = subImg.cols;
	int nRows = grayImg.rows - subRows + 1;
	int nCols = grayImg.cols - subCols + 1;
	int dCols = grayImg.cols - subCols;
	int finex = 0, finey = 0;
	uchar *sub = subImg.data, *gray = grayImg.data, *gray0 = gray;
	int i, j, k, l;
	int max = 0;
	int subD = 0, grayD = 0, D = 0;
	_int64 Div;
	for (i = subRows*subCols; i; --i, sub++)
	{
		subD += *sub**sub;
	}
	for (i = 0; i < nRows; ++i)
	{
		for (j = 0; j < nCols; ++j)
		{
			for (sub = subImg.data, gray = gray0, k = subRows, grayD = 0, D = 0; k; --k)
			{
				for (l = subCols; l; --l, ++sub, ++gray)
				{
					grayD += *gray**gray;
					D += *gray**sub;
				}
				gray += dCols;
			}
			Div = D;
			Div = Div << 8;
			Div = Div / subD*Div / grayD;
			max = !((max - Div) >> 31)*max + !((Div - max) >> 31)*Div;
			finex = !(Div^max)*j + ((Div^max) && 1)*finex;
			finey = !(Div^max)*i + ((Div^max) && 1)*finey;
			++gray0;
		}
		gray0 = gray0 + (subCols - 1);
	}
	*x = finex;
	*y = finey;
	return SUB_IMAGE_MATCH_OK;
}

//函数功能：利用角度值进行子图匹配
//grayImg：灰度图，单通道
//subImg：模板子图，单通道
//x：最佳匹配子图左上角x坐标
//y：最佳匹配子图左上角y坐标
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL
int ustc_SubImgMatch_angle(Mat grayImg, Mat subImg, int * x, int * y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int angRows = subImg.rows - 2;
	int angCols = subImg.cols - 2;
	int subRows = subImg.rows;
	int subCols = subImg.cols;
	int grayRows = grayImg.rows;
	int grayCols = grayImg.cols;
	int *sub_ang, *gray_ang;
	sub_ang = new int[angRows*angCols];
	gray_ang = new int[(grayRows - 2)*(grayCols - 2)];
	int *sub_p = sub_ang, *gray_p = gray_ang;
	uchar *a0 = subImg.data, *a1 = a0 + subCols, *a2 = a1 + subCols;
	uchar *b0 = grayImg.data, *b1 = b0 + grayCols, *b2 = b1 + grayCols;
	int i, j;
	int temp;
	int dx, dy;
	for (i = angRows; i; --i)
	{
		for (j = angCols; j; --j, sub_p++, gray_p++)
		{
			dx = *a2 + 2 * *(a2 + 1) + *(a2 + 2) - *a0 - 2 * *(a0 + 1) - *(a0 + 2);
			dy = *a0 + 2 * *a1 + *a2 - *(a0 + 2) - 2 * *(a1 + 2) - *(a2 + 2);
			temp = atan2(dx, dy)*57.3;
			*sub_p = temp + (!(dx >> 31))*((temp >> 31)) * 360;
			dx = *b2 + 2 * *(b2 + 1) + *(b2 + 2) - *b0 - 2 * *(b0 + 1) - *(b0 + 2);
			dy = *b0 + 2 * *b1 + *b2 - *(b0 + 2) - 2 * *(b1 + 2) - *(b2 + 2);
			temp = atan2(dx, dy)*57.3;
			*gray_p = temp + (!(dx >> 31))*((temp >> 31)) * 360;
			++a0;
			++a1;
			++a2;
			++b0;
			++b1;
			++b2;
		}
		a0 += 2;
		a1 += 2;
		a2 += 2;
		for (j = grayCols - subCols; j; --j, gray_p++)
		{
			dx = *b2 + 2 * *(b2 + 1) + *(b2 + 2) - *b0 - 2 * *(b0 + 1) - *(b0 + 2);
			dy = *b0 + 2 * *b1 + *b2 - *(b0 + 2) - 2 * *(b1 + 2) - *(b2 + 2);
			temp = atan2(dx, dy)*57.3;
			*gray_p = temp + (!(dx >> 31))*((temp >> 31)) * 360;
			++b0;
			++b1;
			++b2;
		}
		b0 += 2;
		b1 += 2;
		b2 += 2;
	}
	for (i = grayRows - subRows; i; --i)
	{
		for (j = grayCols - 2; j; --j, gray_p++)
		{
			dx = *b2 + 2 * *(b2 + 1) + *(b2 + 2) - *b0 - 2 * *(b0 + 1) - *(b0 + 2);
			dy = *b0 + 2 * *b1 + *b2 - *(b0 + 2) - 2 * *(b1 + 2) - *(b2 + 2);
			temp = atan2(dx, dy)*57.3;
			*gray_p = temp + (!(dx >> 31))*((temp >> 31)) * 360;
			++b0;
			++b1;
			++b2;
		}
		b0 += 2;
		b1 += 2;
		b2 += 2;
	}
	int nRows = grayImg.rows - angRows - 1;
	int nCols = grayImg.cols - angCols - 1;
	int dCols = grayImg.cols - subCols;
	int finex = 0, finey = 0;
	int *gray0 = gray_ang;
	int  k, l;
	int min = 0x7FFFFFFF;
	int sum;
	int diff;
	for (i = 0; i < nRows; ++i)
	{
		for (j = 0; j < nCols; ++j)
		{
			for (gray_p = gray0, sub_p = sub_ang, k = angRows, sum = 0; k; --k)
			{
				for (l = angCols; l; --l, ++sub_p, ++gray_p)
				{
					diff = *sub_p - *gray_p;
					diff = ((diff >> 31) & 1)*(~diff + 1) + !(diff >> 31)*diff;
					diff = !((diff - 180) >> 31)*(360 - diff) + (((diff - 180) >> 31) & 1)* diff;
					sum += diff;
				}
				if ((min - sum) >> 31)break;
				gray_p += dCols;
			}
			min = !((sum - min) >> 31)*min + !((min - sum) >> 31)*sum;
			finex = !(sum^min)*j + ((sum^min) && 1)*finex;
			finey = !(sum^min)*i + ((sum^min) && 1)*finey;
			++gray0;
		}
		gray0 = gray0 + (subCols - 3);
	}
	*x = finex;
	*y = finey;
	return SUB_IMAGE_MATCH_OK;
}

//函数功能：利用幅值进行子图匹配
//grayImg：灰度图，单通道
//subImg：模板子图，单通道
//x：最佳匹配子图左上角x坐标
//y：最佳匹配子图左上角y坐标
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL
int ustc_SubImgMatch_mag(Mat grayImg, Mat subImg, int * x, int * y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int angRows = subImg.rows - 2;
	int angCols = subImg.cols - 2;
	int subRows = subImg.rows;
	int subCols = subImg.cols;
	int grayRows = grayImg.rows;
	int grayCols = grayImg.cols;
	int *sub_ang, *gray_ang;
	sub_ang = new int[angRows*angCols];
	gray_ang = new int[(grayRows - 2)*(grayCols - 2)];
	int *sub_p = sub_ang, *gray_p = gray_ang;
	uchar *a0 = subImg.data, *a1 = a0 + subCols, *a2 = a1 + subCols;
	uchar *b0 = grayImg.data, *b1 = b0 + grayCols, *b2 = b1 + grayCols;
	int i, j;
	int dx, dy;
	for (i = angRows; i; --i)
	{
		for (j = angCols; j; --j, sub_p++, gray_p++)
		{
			dx = *a2 + 2 * *(a2 + 1) + *(a2 + 2) - *a0 - 2 * *(a0 + 1) - *(a0 + 2);
			dy = *a0 + 2 * *a1 + *a2 - *(a0 + 2) - 2 * *(a1 + 2) - *(a2 + 2);
			*sub_p = dx * dx + dy*dy;
			dx = *b2 + 2 * *(b2 + 1) + *(b2 + 2) - *b0 - 2 * *(b0 + 1) - *(b0 + 2);
			dy = *b0 + 2 * *b1 + *b2 - *(b0 + 2) - 2 * *(b1 + 2) - *(b2 + 2);
			*gray_p = dx*dx + dy*dy;
			++a0;
			++a1;
			++a2;
			++b0;
			++b1;
			++b2;
		}
		a0 += 2;
		a1 += 2;
		a2 += 2;
		for (j = grayCols - subCols; j; --j, gray_p++)
		{
			dx = *b2 + 2 * *(b2 + 1) + *(b2 + 2) - *b0 - 2 * *(b0 + 1) - *(b0 + 2);
			dy = *b0 + 2 * *b1 + *b2 - *(b0 + 2) - 2 * *(b1 + 2) - *(b2 + 2);
			*gray_p = dx*dx + dy*dy;
			++b0;
			++b1;
			++b2;
		}
		b0 += 2;
		b1 += 2;
		b2 += 2;
	}
	for (i = grayRows - subRows; i; --i)
	{
		for (j = grayCols - 2; j; --j, gray_p++)
		{
			dx = *b2 + 2 * *(b2 + 1) + *(b2 + 2) - *b0 - 2 * *(b0 + 1) - *(b0 + 2);
			dy = *b0 + 2 * *b1 + *b2 - *(b0 + 2) - 2 * *(b1 + 2) - *(b2 + 2);
			*gray_p = dx*dx + dy*dy;
			++b0;
			++b1;
			++b2;
		}
		b0 += 2;
		b1 += 2;
		b2 += 2;
	}
	int nRows = grayImg.rows - angRows - 1;
	int nCols = grayImg.cols - angCols - 1;
	int dCols = grayImg.cols - subCols;
	int finex = 0, finey = 0;
	int *gray0 = gray_ang;
	int  k, l;
	int min = 0x7FFFFFFF;
	int sum;
	int diff;
	for (i = 0; i < nRows; ++i)
	{
		for (j = 0; j < nCols; ++j)
		{
			for (gray_p = gray0, sub_p = sub_ang, k = angRows, sum = 0; k; --k)
			{
				for (l = angCols; l; --l, ++sub_p, ++gray_p)
				{
					diff = *sub_p - *gray_p;
					diff = ((diff >> 31) & 1)*(~diff + 1) + !(diff >> 31)*diff;
					sum += diff;
				}
				if ((min - sum) >> 31)break;
				gray_p += dCols;
			}
			min = !((sum - min) >> 31)*min + !((min - sum) >> 31)*sum;
			finex = !(sum^min)*j + ((sum^min) && 1)*finex;
			finey = !(sum^min)*i + ((sum^min) && 1)*finey;
			++gray0;
		}
		gray0 = gray0 + (subCols - 3);
	}
	*x = finex;
	*y = finey;
	return SUB_IMAGE_MATCH_OK;
}

//函数功能：利用直方图进行子图匹配
//grayImg：灰度图，单通道
//subImg：模板子图，单通道
//x：最佳匹配子图左上角x坐标
//y：最佳匹配子图左上角y坐标
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL
int ustc_SubImgMatch_hist(Mat grayImg, Mat subImg, int * x, int * y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int hist[256] = { 0 };
	int *p, *q;
	int subRows = subImg.rows;
	int subCols = subImg.cols;
	int length = subRows*subCols;
	int nRows = grayImg.rows - subRows + 1;
	int nCols = grayImg.cols - subCols + 1;
	int dCols = grayImg.cols - subCols;
	int finex = 0, finey = 0;
	uchar *sub = subImg.data, *gray = grayImg.data, *gray0 = gray;
	int i, j, k, l, m;
	int min = 0x7FFFFFFF;
	int sum;
	int diff;
	for (i = length; i; --i)
	{
		++hist[(*sub) - 1];
		++sub;
	}
	for (i = 0; i < nRows; ++i)
	{
		for (j = 0; j < nCols; ++j)
		{
			int grayhist[256] = { 0 };
			for (gray = gray0, k = subRows, sum = 0; k; --k)
			{
				for (l = subCols; l; --l, ++gray)
				{
					grayhist[(*gray) - 1]--;
				}
				gray += dCols;
			}
			for (m = 256, p = hist, q = grayhist; m; --m)
			{
				diff = *(p++) + *(q++);
				sum += ((diff >> 31) & 1)*(~diff + 1) + !(diff >> 31)*diff;
			}
			min = !((sum - min) >> 31)*min + !((min - sum) >> 31)*sum;
			finex = !(sum^min)*j + ((sum^min) && 1)*finex;
			finey = !(sum^min)*i + ((sum^min) && 1)*finey;
			++gray0;
		}
		gray0 = gray0 + (subCols - 1);
	}
	*x = finex;
	*y = finey;
	return SUB_IMAGE_MATCH_OK;
}
