#include "SubImageMatch.h"

#define PI 3.1415926

//函数功能：将bgr图像转化成灰度图像
//bgrImg：彩色图，像素排列顺序是bgr
//grayImg：灰度图，单通道
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL
int ustc_ConvertBgr2Gray(Mat bgrImg, Mat& grayImg) {
	if (NULL == bgrImg.data || NULL == grayImg.data)
	{
		cout << "图片为空！" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (bgrImg.type() != 16 || grayImg.type() != 0) {
		cout << "图片格式错误！" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int nRows = bgrImg.rows;
	int nCols = bgrImg.cols;
	int step0 = bgrImg.step[0];
	int i, j;
	uchar* temp;
	uchar* p,* pg;
	p = bgrImg.data - step0;
	pg = grayImg.data - nCols;
	//for (i = j = 0; j < nCols; ++j)
	//{
	//	temp = p + j * 3;
	//	*(pg + j) = (*(temp + 0)*7472 + *(temp + 1)*38469 + *(temp + 2)*19595)>>16;
	//}
	for (i = nRows; i > 0; --i)
	{
		p += step0;
		pg += nCols;
		for (j = 0; j < nCols; ++j)
		{
			temp = p + j * 3;
			*(pg + j) = (*(temp + 0) * 7472 + *(temp + 1) * 38469 + *(temp + 2) * 19595) >> 16;
		}
	}
	return SUB_IMAGE_MATCH_OK;
}

//函数功能：根据灰度图像计算梯度图像
//grayImg：灰度图，单通道
//gradImg_x：水平方向梯度，浮点类型图像，CV32FC1
//gradImg_y：垂直方向梯度，浮点类型图像，CV32FC1
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL
int ustc_CalcGrad(Mat grayImg, Mat& gradImg_x, Mat& gradImg_y) {
	if (NULL == grayImg.data || NULL == gradImg_x.data || NULL == gradImg_y.data)
	{
		cout << "图片为空！" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (gradImg_x.type() != 5 || gradImg_y.type() != 5 || grayImg.type() != 0) {
		cout << "图片格式错误！" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int nRows = grayImg.rows;
	int nCols = grayImg.cols;
	int i, j;
	uchar* p,* ptemp;
	float* px,* py;
	//int temp;
	uchar p11, p12, p13, p21, p23, p31, p32, p33;
	if (nRows <= 2) {
		return SUB_IMAGE_MATCH_OK;
	}
	gradImg_x.setTo(0);
	gradImg_y.setTo(0);
	p = grayImg.data;
	px= (float*)gradImg_x.data;
	py= (float*)gradImg_y.data;
	for (i = 1; i < nRows - 1; ++i)
	{
		p += nCols;
		px += nCols;
		py += nCols;
		for (j = 1; j < nCols - 1; ++j)
		{
			ptemp = p + j;
			p11 = *(ptemp - nCols - 1);
			p12 = *(ptemp - nCols);
			p13 = *(ptemp - nCols + 1);
			p21 = *(ptemp - 1);
			//p22 = *(ptemp);
			p23 = *(ptemp + 1);
			p31 = *(ptemp + nCols - 1);
			p32 = *(ptemp + nCols);
			p33 = *(ptemp + nCols + 1);
			*(px + j) = p11 + (p21 + p21) + p31 - p13 - (p23 + p23) - p33;
			*(py + j) = p11 + (p12 + p12) + p13 - p31 - (p32 + p32) - p33;
			//cout << *(px + j) << ", " << *(py + j) << endl;
		}
	}
	return SUB_IMAGE_MATCH_OK;
}

//函数功能：根据水平和垂直梯度，计算角度和幅值图
//gradImg_x：水平方向梯度，浮点类型图像，CV32FC1
//gradImg_y：垂直方向梯度，浮点类型图像，CV32FC1
//angleImg：角度图，浮点类型图像，CV32FC1
//magImg：幅值图，浮点类型图像，CV32FC1
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL
int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y, Mat& angleImg, Mat& magImg) {
	if (NULL == gradImg_x.data || NULL == gradImg_y.data || NULL == angleImg.data || NULL == magImg.data)
	{
		cout << "图片为空！" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (angleImg.type() != 5 || magImg.type() != 5 || gradImg_x.type() != 5 || gradImg_y.type() != 5) {
		cout << "图片格式错误！" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int nRows = gradImg_x.rows;
	int nCols = gradImg_x.cols;
	int i, j;
	float *pgx, *pgy, *pa, *pm;
	angleImg.setTo(0);
	magImg.setTo(0);
	pgx = (float*)gradImg_x.data - nCols;
	pgy = (float*)gradImg_y.data - nCols;
	pa = (float*)angleImg.data - nCols;
	pm = (float*)magImg.data - nCols;
	float x, y;
	float angle;
	for (i = nRows; i > 0; --i)
	{
		pgx += nCols;
		pgy += nCols;
		pa += nCols;
		pm += nCols;
		for (j = 0; j < nCols; ++j)
		{
			x = *(pgx + j);
			y = *(pgy + j);
			angle = atan2(y, x);
			*(pa + j) = angle * 180 / PI + 360 * (!!((int)(angle) >> 31));
			//cout << angle << endl;

			//*(pa + j) = Table_tan[(int)*(pgx + j) + 255][(int)*(pgy + j) + 255];
			*(pm + j) = sqrt(y*y + x*x);
			//f = (*(pgy + j))*(*(pgy + j)) + (*(pgx + j))*(*(pgx + j));
			//mag = *(int*)&f; 
			//mag -= 0x3f800000; 
			//mag >>= 1; 
			//mag += 0x3f800000;
			//*(pm + j) = f;
		}
	}
	return SUB_IMAGE_MATCH_OK;
}

//函数功能：对灰度图像进行二值化
//grayImg：灰度图，单通道
//binaryImg：二值图，单通道
//th：二值化阈值，高于此值，255，低于此值0
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL
int ustc_Threshold(Mat grayImg, Mat& binaryImg, int th) {
	if (NULL == grayImg.data || NULL == binaryImg.data)
	{
		cout << "图片为空！" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (grayImg.type() != 0 || binaryImg.type() != 0) {
		cout << "图片格式错误！" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int nRows = grayImg.rows;
	int nCols = grayImg.cols;
	int i, j;
	uchar* pg,* pb;
	pg = grayImg.data - nCols;
	pb = binaryImg.data - nCols;
	//if (grayImg.isContinuous()&& binaryImg.isContinuous())
	//{
	//	nCols *= nRows;
	//	nRows = 1;
	//}
	//for (i = j = 0; j < nCols; ++j)
	//{
	//	//if (*(pg + j) > th) *(pb + j) = 255;
	//	//else *(pb + j) = 0;
	//	*(pb + j) = ~((*(pg + j) - th) >> 7);
	//}
	for (i = nRows; i > 0; --i)
	{
		pg += nCols;
		pb += nCols;
		for (j = 0; j < nCols; ++j)
		{
			*(pb + j) = ~((*(pg + j) - th) >> 7);
		}
	}
	return SUB_IMAGE_MATCH_OK;
}

//函数功能：对灰度图像计算直方图
//grayImg：灰度图，单通道
//hist：直方图
//hist_len：直方图的亮度等级，直方图数组的长度
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL
int ustc_CalcHist(Mat grayImg, int* hist, int hist_len) {
	if (NULL == grayImg.data)
	{
		cout << "图片为空！" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (grayImg.type() != 0) {
		cout << "图片格式错误！" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int nRows = grayImg.rows;
	int nCols = grayImg.cols;
	int i, j;
	uchar* p;
	memset(hist, 0, hist_len * sizeof(int));
	//if (grayImg.isContinuous())
	//{
	//	nCols *= nRows;
	//	nRows = 1;
	//}
	p = grayImg.data - nCols;
	//for (i = j = 0; j < nCols; ++j)
	//{
	//	*(hist + *(p + j)) += 1;
	//}
	for (i = nRows; i > 0; --i)
	{
		p += nCols;
		for (j = 0; j < nCols; ++j)
		{
			*(hist + *(p + j)) += 1;
		}
	}
	return SUB_IMAGE_MATCH_OK;
}

//函数功能：利用亮度进行子图匹配
//grayImg：灰度图，单通道
//subImg：模板子图，单通道
//x：最佳匹配子图左上角x坐标
//y：最佳匹配子图左上角y坐标
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL
int ustc_SubImgMatch_gray(Mat grayImg, Mat subImg, int* x, int* y) 
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "图片为空！" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (grayImg.type() != 0 || subImg.type() != 0) {
		cout << "图片格式错误！" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int nCols = grayImg.cols;
	int nRows = grayImg.rows;
	int subCols = subImg.cols;
	int subRows = subImg.rows;

	int i, j, ii, jj;
	int row, subrow;
	int temp0;
	int min_diff = 0x7FFFFFFF;
	int total_diff = 0;
	*x = *y = 0;
	uchar* p,* ps;
	p = grayImg.data - nCols;
	ps = subImg.data;
	for (i = 0; i < nRows - subRows; i++)
	{
		p += nCols;
		for (j = 0; j < nCols - subCols; j++)
		{
			total_diff = 0;
			row = -nCols + j;
			subrow = -subCols;
			for (ii = 0; ii < subRows; ii++)
			{
				row += nCols;
				subrow += subCols;
				for (jj = 0; jj < subCols; ++jj)
				{
					temp0 = p[row + jj] - ps[subrow + jj];
					total_diff += abs(temp0);
				}
			}
			if (total_diff < min_diff)
			{
				*x = j;
				*y = i;
				min_diff = total_diff;
			}
			//temp0 = !((total_diff - min_diff)>>31);
			//*x = (*x)*temp0 + j * !temp0;
			//*y = (*y)*temp0 + i * !temp0;
			//min_diff = min_diff*temp0 + total_diff * !temp0;
		}
	}
	return SUB_IMAGE_MATCH_OK;
}

//函数功能：利用色彩进行子图匹配
//colorImg：彩色图，三通单
//subImg：模板子图，三通道
//x：最佳匹配子图左上角x坐标
//y：最佳匹配子图左上角y坐标
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL
int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int* x, int* y) 
{
	if (NULL == colorImg.data || NULL == subImg.data)
	{
		cout << "图片为空！" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (colorImg.type() != 16 || subImg.type() != 16) {
		cout << "图片格式错误！" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int nCols = colorImg.cols;
	int nRows = colorImg.rows;
	int subCols = subImg.cols;
	int subRows = subImg.rows;
	int step = nCols * 3;
	int substep = subCols * 3;

	int i, j, ii, jj;
	int row, subrow,col,subcol;
	char temp0, temp1, temp2, temp3;
	int min_diff = 0x7FFFFFFF;
	int total_diff = 0;
	*x = *y = 0;

	uchar* p,* ps;
	p = colorImg.data - step;
	ps = subImg.data;
	for (i = 0; i < nRows - subRows; i++)
	{
		p += step;
		for (j = 0; j < nCols - subCols; j++)
		{
			total_diff = 0;
			row = -step + j * 3;
			subrow = -substep;
			for (ii = 0; ii < subRows; ii++)
			{
				row += step;
				subrow += substep;
				temp1 = temp2 = temp3 = 0;
				for (jj = 0; jj < subCols; ++jj)
				{
					col = row + jj * 3;
					subcol = subrow + jj * 3;
					temp0 = p[col] - ps[subcol];
					temp1 = abs(temp0);//1
					temp0 = p[col + 1] - ps[subcol + 1];
					temp2 = abs(temp0);//1
					temp0 = p[col + 2] - ps[subcol + 2];
					temp3 = abs(temp0);//1
				}
				total_diff += temp1 + temp2 + temp3;
			}
			//存储当前像素位置的匹配误差
			//((float*)searchImg.data)[i * nCols + j] = total_diff;
			if (total_diff < min_diff)
			{
					*x = j;
					*y = i;
					min_diff = total_diff;
			}
			
		}
	}
	return SUB_IMAGE_MATCH_OK;
}

//函数功能：利用亮度相关性进行子图匹配
//grayImg：灰度图，单通道
//subImg：模板子图，单通道
//x：最佳匹配子图左上角x坐标
//y：最佳匹配子图左上角y坐标
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL
int ustc_SubImgMatch_corr(Mat grayImg, Mat subImg, int* x, int* y) 
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "图片为空！" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (grayImg.type() != 0 || subImg.type() != 0) {
		cout << "图片格式错误！" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int nCols = grayImg.cols;
	int nRows = grayImg.rows;
	int subCols = subImg.cols;
	int subRows = subImg.rows;


	int i, j, ii, jj;
	int row, subrow;
	//int temp0;
	float max_diff = 0;
	float total_diff = 0;
	int ST, S2, T2;
	int s, t;
	*x = *y = 0;
	uchar* p, *ps;
	p = grayImg.data - nCols;
	ps = subImg.data;
	for (i = 0; i < nRows - subRows; i++)
	{
		p += nCols;
		for (j = 0; j < nCols - subCols; j++)
		{
			//total_diff = 0;
			row = -nCols + j;
			subrow = -subCols;
			ST = 0;
			S2 = 0;
			T2 = 0;
			for (ii = 0; ii < subRows; ii++)
			{
				row += nCols;
				subrow += subCols;
				for (jj = 0; jj < subCols; jj++)
				{
					s = p[row + jj];
					t = ps[subrow + jj];
					ST += s*t;
					S2 += s*s;
					T2 += t*t;
				}
			}
			//存储当前像素位置的匹配误差
			//((float*)searchImg.data)[i * nCols + j] = total_diff;
			total_diff = ST/sqrt(S2)*sqrt(T2);
			if (total_diff > max_diff)
			{
				*x = j;
				*y = i;
				max_diff = total_diff;
			}
		}
	}
	return SUB_IMAGE_MATCH_OK;
}

//函数功能：利用角度值进行子图匹配
//grayImg：灰度图，单通道
//subImg：模板子图，单通道
//x：最佳匹配子图左上角x坐标
//y：最佳匹配子图左上角y坐标
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL
int ustc_SubImgMatch_angle(Mat grayImg, Mat subImg, int* x, int* y) 
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "图片为空！" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (grayImg.type() != 0 || subImg.type() != 0) {
		cout << "图片格式错误！" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int nCols = grayImg.cols;
	int nRows = grayImg.rows;
	int subCols = subImg.cols;
	int subRows = subImg.rows;

	Mat angleImg(nRows, nCols, CV_8UC1);
	Mat subangleImg(subRows, subCols, CV_8UC1);

	int i, j, ii, jj;
	int row, subrow;
	int temp0;
	float angle;
	int min_diff = 0x7FFFFFFF;
	int total_diff = 0;
	*x = *y = 0;
	int Gx, Gy;
	uchar p11, p12, p13, p21, p23, p31, p32, p33;
	uchar* p,* ps;
	uchar* p0;
	p = subangleImg.data;
	p0 = subImg.data;
	row = 0;
	for (i = 1; i < subRows - 1; ++i)
	{
		row += subCols;
		for (j = 1; j < subCols - 1; ++j)
		{
			temp0 = row + j;
			p11 = p0[temp0 - subCols - 1];
			p12 = p0[temp0 - subCols];
			p13 = p0[temp0 - subCols + 1];
			p21 = p0[temp0 - 1];
			//p22 = *(ptemp);
			p23 = p0[temp0 + 1];
			p31 = p0[temp0 + subCols - 1];
			p32 = p0[temp0 + subCols];
			p33 = p0[temp0 + subCols + 1];
			Gx = p11 + (p21 + p21) + p31 - p13 - (p23 + p23) - p33;
			Gy = p11 + (p12 + p12) + p13 - p31 - (p32 + p32) - p33;
			angle = atan2(Gy, Gx);
			p[temp0] = angle * 180 / PI + 360 * (!!((int)(angle - 1) >> 31));
			//printf("%d, %d, %f\n", Gx, Gy, p[temp0 + j]);
		}
	}
	p = angleImg.data;
	p0 = grayImg.data;
	row = 0;
	for (i = 1; i < nRows - 1; ++i)
	{
		row += nCols;
		for (j = 1; j < nCols - 1; ++j)
		{
			temp0 = row + j;
			p11 = p0[temp0 - nCols - 1];
			p12 = p0[temp0 - nCols];
			p13 = p0[temp0 - nCols + 1];
			p21 = p0[temp0 - 1];
			//p22 = *(ptemp);
			p23 = p0[temp0 + 1];
			p31 = p0[temp0 + nCols - 1];
			p32 = p0[temp0 + nCols];
			p33 = p0[temp0 + nCols + 1];
			Gx = p11 + (p21 + p21) + p31 - p13 - (p23 + p23) - p33;
			Gy = p11 + (p12 + p12) + p13 - p31 - (p32 + p32) - p33;
			angle = atan2(Gy, Gx);
			p[temp0] = angle * 180 / PI + 360 * (!!((int)angle >> 31));
			//printf("%d, %d, %f\n", Gx, Gy, p[temp0 + j]);
		}
	}

	int temp1;
	uchar temp2;
	int result;

	p = angleImg.data - nCols;
	ps = subangleImg.data;
	for (i = 0; i < nRows - subRows; i++)
	{
		p += nCols;
		for (j = 0; j < nCols - subCols; j++)
		{
			total_diff = 0;
			row = j;
			subrow = 0;
			for (ii = 1; ii < subRows - 1; ii++)
			{
				row += nCols;
				subrow += subCols;
				for (jj = 1; jj < subCols - 1; jj++)
				{
					temp0 = p[row + jj] - ps[subrow + jj];
					temp1 = abs(temp0);
					temp2 = !!((temp1 - 180) >> 31);
					result = 360 * !temp2 + (-1 + temp2 + temp2)*temp1;
					total_diff += result;
				}
			}
			if (total_diff < min_diff)
			{
				*x = j;
				*y = i;
				min_diff = total_diff;
			}
		}
	}
	return SUB_IMAGE_MATCH_OK;
}

//函数功能：利用幅值进行子图匹配
//grayImg：灰度图，单通道
//subImg：模板子图，单通道
//x：最佳匹配子图左上角x坐标
//y：最佳匹配子图左上角y坐标
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL
int ustc_SubImgMatch_mag(Mat grayImg, Mat subImg, int* x, int* y) 
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "图片为空！" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (grayImg.type() != 0 || subImg.type() != 0) {
		cout << "图片格式错误！" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int nCols = grayImg.cols;
	int nRows = grayImg.rows;
	int subCols = subImg.cols;
	int subRows = subImg.rows;

	Mat magImg(nRows, nCols, CV_8UC1);
	Mat submagImg(subRows, subCols, CV_8UC1);

	int i, j, ii, jj;
	int row, subrow;
	int temp0;
	//float mag;
	int min_diff = 0x7FFFFFFF;
	int total_diff = 0;
	*x = *y = 0;
	int Gx, Gy;
	uchar p11, p12, p13, p21, p23, p31, p32, p33;
	uchar* p, *ps;
	uchar* p0;
	p = submagImg.data;
	p0 = subImg.data;
	row = 0;
	for (i = 1; i < subRows - 1; ++i)
	{
		row += subCols;
		for (j = 1; j < subCols - 1; ++j)
		{
			temp0 = row + j;
			p11 = p0[temp0 - subCols - 1];
			p12 = p0[temp0 - subCols];
			p13 = p0[temp0 - subCols + 1];
			p21 = p0[temp0 - 1];
			//p22 = *(ptemp);
			p23 = p0[temp0 + 1];
			p31 = p0[temp0 + subCols - 1];
			p32 = p0[temp0 + subCols];
			p33 = p0[temp0 + subCols + 1];
			Gx = p11 + (p21 + p21) + p31 - p13 - (p23 + p23) - p33;
			Gy = p11 + (p12 + p12) + p13 - p31 - (p32 + p32) - p33;
			p[temp0] = sqrt(Gx*Gx+Gy*Gy);
			//printf("%d, %d, %f\n", Gx, Gy, p[temp0 + j]);
		}
	}
	p = magImg.data;
	p0 = grayImg.data;
	row = 0;
	for (i = 1; i < nRows - 1; ++i)
	{
		row += nCols;
		for (j = 1; j < nCols - 1; ++j)
		{
			temp0 = row + j;
			p11 = p0[temp0 - nCols - 1];
			p12 = p0[temp0 - nCols];
			p13 = p0[temp0 - nCols + 1];
			p21 = p0[temp0 - 1];
			//p22 = *(ptemp);
			p23 = p0[temp0 + 1];
			p31 = p0[temp0 + nCols - 1];
			p32 = p0[temp0 + nCols];
			p33 = p0[temp0 + nCols + 1];
			Gx = p11 + (p21 + p21) + p31 - p13 - (p23 + p23) - p33;
			Gy = p11 + (p12 + p12) + p13 - p31 - (p32 + p32) - p33;
			p[temp0] = sqrt(Gx*Gx + Gy*Gy);
			//printf("%d, %d, %f\n", Gx, Gy, p[temp0 + j]);
		}
	}
	p = magImg.data - nCols;
	ps = submagImg.data;
	for (i = 0; i < nRows - subRows; i++)
	{
		p += nCols;
		for (j = 0; j < nCols - subCols; j++)
		{
			total_diff = 0;
			row = j;
			subrow = 0;
			for (ii = 1; ii < subRows - 1; ii++)
			{
				row += nCols;
				subrow += subCols;
				for (jj = 1; jj < subCols - 1; jj++)
				{
					temp0 = p[row + jj] - ps[subrow + jj];

					total_diff += abs(temp0);//1
					total_diff += (!(temp0 >> 31))*temp0 - (!!(temp0 >> 31))*temp0;//2
				}
			}
			if (total_diff < min_diff)
			{
				*x = j;
				*y = i;
				min_diff = total_diff;
			}
		}
	}
	return SUB_IMAGE_MATCH_OK;
}

//函数功能：利用直方图进行子图匹配
//grayImg：灰度图，单通道
//subImg：模板子图，单通道
//x：最佳匹配子图左上角x坐标
//y：最佳匹配子图左上角y坐标
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL
int ustc_SubImgMatch_hist(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "图片为空！" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (grayImg.type() != 0 || subImg.type() != 0) {
		cout << "图片格式错误！" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int nCols = grayImg.cols;
	int nRows = grayImg.rows;
	int subCols = subImg.cols;
	int subRows = subImg.rows;

	int i, j, ii, jj, n;
	int hist_len = 256;
	int* hist_temp = new int[hist_len];
	int* subhist = new int[hist_len];
	memset(subhist, 0, sizeof(int) * hist_len);
	uchar* p;
	//计算子图直方图
	p = subImg.data - subCols;
	for (i = 0; i < subRows; ++i)
	{
		p += subCols;
		for (j = 0; j < subCols; ++j)
		{
			*(subhist + *(p + j)) += 1;
		}
	}
	int temp, temp0, temp00;
	int bigImg_pix;
	int total_diff;
	int min_diff = 0x7fffffff;
	//匹配
	temp = -nCols;
	p = grayImg.data;

	for (i = 0; i < nRows - subRows; i++)
	{
		temp += nCols;
		j = 0;
		memset(hist_temp, 0, sizeof(int) * hist_len);
		temp0 = temp + j;
		for (ii = 0; ii < subRows; ii++)
		{
			temp00 = temp0 + ii * nCols;
			for (jj = 0; jj < subCols; jj++)
			{
				bigImg_pix = p[temp00 + jj];
				hist_temp[bigImg_pix]++;
			}
		}
		total_diff = 0;
		for (n = 0; n < hist_len; n++)
		{
			total_diff += abs(hist_temp[n] - subhist[n]);
		}
		if (total_diff < min_diff)
		{
			*x = j;
			*y = i;
			min_diff = total_diff;
		}
		for (j = 1; j < nCols - subCols; j++)
		{
			temp0 = temp + j + subCols - 1;
			for (ii = 0; ii < subRows; ii++)
			{
				temp00 = temp0 + ii * nCols;
				bigImg_pix = p[temp00];
				hist_temp[bigImg_pix]++;
			}
			temp0 = temp + j - 1;
			for (ii = 0; ii < subRows; ii++)
			{
				temp00 = temp0 + ii * nCols;
				bigImg_pix = p[temp00];
				hist_temp[bigImg_pix]--;
			}
			total_diff = 0;
			for (n = 0; n < hist_len; n++)
			{
				total_diff += abs(hist_temp[n] - subhist[n]);
			}
			if (total_diff < min_diff)
			{
				*x = j;
				*y = i;
				min_diff = total_diff;
			}
		}
	}
	delete[] hist_temp;
	delete[] subhist;
	return SUB_IMAGE_MATCH_OK;
}
