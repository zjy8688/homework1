#include"SubImageMatch.h"

//函数功能：将bgr图像转化成灰度图像
//bgrImg：彩色图，像素排列顺序是bgr
//grayImg：灰度图，单通道
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL
int ustc_ConvertBgr2Gray(Mat bgrImg, Mat& grayImg) {
	if (NULL == bgrImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}


	if (bgrImg.channels() != 3 || grayImg.channels() != 1)
	{
		cout << "channel not fit" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}


	int height = bgrImg.rows;
	int width = bgrImg.cols;
	grayImg.create(height, width, CV_8UC1);

	uchar *p = bgrImg.data, *end = bgrImg.data + 3 * height * width, *q = grayImg.data;

	while (p < end)
	{
		(*q) = (uchar)(((*p) * 114) >> 10) + (uchar)(((*(p + 1)) * 587) >> 10) + (uchar)(((*(p + 2)) * 229 >> 10));
		q += 1;
		p += 3;
	}
	return SUB_IMAGE_MATCH_OK;
}

//函数功能：根据灰度图像计算梯度图像
//grayImg：灰度图，单通道
//gradImg_x：水平方向梯度，浮点类型图像，CV32FC1
//gradImg_y：垂直方向梯度，浮点类型图像，CV32FC1
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL
int ustc_CalcGrad(Mat grayImg, Mat& gradImg_x, Mat& gradImg_y)
{
	if (NULL == grayImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	//计算X方向梯度

	if (grayImg.channels() != 1)
	{
		cout << "channel not fit" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int height = grayImg.rows;
	int width = grayImg.cols;


	gradImg_x.create(height, width, CV_32FC1);
	gradImg_x.setTo(0);

	for (int row_i = 1; row_i < height - 1; row_i++)
	{
		int temp0 = row_i*width;
		for (int col_j = 1; col_j < width - 1; col_j += 1)
		{
			int temp1 = temp0 + col_j + 1;
			int temp2 = temp0 + col_j - 1;
			((float *)(gradImg_x.data))[temp0 + col_j] = *(grayImg.data + temp1 - width)
				+ ((*(grayImg.data + temp1)) << 1)
				+ *(grayImg.data + temp1 + width)
				- *(grayImg.data + temp2 - width)
				- (*(grayImg.data + temp2) << 1)
				- *(grayImg.data + temp2 + width);
		}
	}

	//计算y方向梯度

	gradImg_y.create(height, width, CV_32FC1);
	gradImg_y.setTo(0);

	for (int row_i = 1; row_i < height - 1; row_i++)
	{
		int temp0 = row_i*width;
		for (int col_j = 1; col_j < width - 1; col_j++)
		{
			int temp1 = temp0 + col_j - width;
			int temp2 = temp0 + col_j + width;
			((float*)gradImg_y.data)[temp0 + col_j] = -*(grayImg.data + temp1 - 1)
				- (*(grayImg.data + temp1) << 1)
				- *(grayImg.data + temp1 + 1)
				+ *(grayImg.data + temp2 - 1)
				+ (*(grayImg.data + temp2) << 1)
				+ *(grayImg.data + temp2 + 1);
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
	if (NULL == gradImg_x.data || NULL == gradImg_y.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = gradImg_x.cols;
	int height = gradImg_x.rows;


	if (gradImg_x.channels() != 1 || gradImg_y.channels() != 1)
	{
		cout << "channel not fit" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	angleImg.create(height, width, CV_32FC1);
	angleImg.setTo(0);


	for (int row_i = 1; row_i < height - 1; row_i++)
	{
		int temp0 = row_i*width;
		for (int col_j = 1; col_j < width - 1; col_j += 1)
		{
			int temp1 = temp0 + col_j;
			float grad_x = ((float*)(gradImg_x.data))[temp1];
			float grad_y = ((float*)(gradImg_y.data))[temp1];
			float angle = atan2(grad_y, grad_x);
			angle = angle * 180 / CV_PI;
			angle = angle > 0 ? angle : 360 + angle;

			((float*)angleImg.data)[temp1] = angle;
		}
	}


	magImg.create(height, width, CV_32FC1);
	magImg.setTo(0);
	for (int row_i = 1; row_i < height - 1; row_i++)
	{
		int temp0 = row_i*width;
		for (int col_j = 1; col_j < width - 1; col_j++)
		{
			int temp1 = temp0 + col_j;
			float grad_x = ((float*)(gradImg_y.data))[temp1];
			float grad_y = ((float*)(gradImg_y.data))[temp1];
			float mag = sqrt(grad_x*grad_x + grad_y*grad_y);
			((float*)magImg.data)[temp1] = mag;
		}
	}

	return SUB_IMAGE_MATCH_OK;
}

//函数功能：对灰度图像进行二值化
//grayImg：灰度图，单通道
//binaryImg：二值图，单通道
//th：二值化阈值，高于此值，255，低于此值0
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL
int ustc_Threshold(Mat grayImg, Mat& binaryImg, int th)
{
	if (NULL == grayImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	if (grayImg.channels() != 1 || binaryImg.channels() != 1)
	{
		cout << "channel not fit" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int height = grayImg.rows;
	int width = grayImg.cols;
	binaryImg.create(height, width, CV_8UC1);

	uchar* end = grayImg.data + height*width, *q = binaryImg.data, *p = grayImg.data;

	while (p < end)
	{
		*(q) = *(p) < th ? 0 : 255;
		q += 1;
		p += 1;
	}

	return SUB_IMAGE_MATCH_OK;

}


//函数功能：对灰度图像计算直方图
//grayImg：灰度图，单通道
//hist：直方图
//hist_len：直方图的亮度等级，直方图数组的长度
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL
int ustc_CalcHist(Mat grayImg, int* hist, int hist_len) {
	if (NULL == grayImg.data || NULL == hist)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	if (grayImg.channels() != 1)
	{
		cout << "channel not fit" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;

	if (hist_len != 256) 
	{
		hist = new int[256];
	}
	//清零
	for (int i = 0; i < hist_len; i++) hist[i] = 0;

	uchar *p = grayImg.data, *end = p + width*height;

	while (p < end)
	{
		hist[*p] += 1;
		p += 1;
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
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	if (grayImg.channels() != subImg.channels() || grayImg.channels() != 1)
	{
		cout << "channels not fit" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	if (width < sub_width || height < sub_height)
	{
		cout << "the subimage is larger than original one" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}


	int num = width*height;
	float minimum = FLT_MAX;
	int gray_row, gray_col, diff_height = height - sub_height, diff_width = width - sub_width;


	for (int row_x = 0; row_x <= diff_height; row_x += 1)
	{
		for (int col_y = 0; col_y <= diff_width; col_y += 1)
		{
			float different = 0;

			for (int row_i = 0; row_i < sub_height; row_i += 1)
			{
				gray_row = row_x + row_i;
				int temp0 = gray_row*width;
				int temp1 = row_i*sub_width;

				for (int col_j = 0; col_j < sub_width; col_j += 1)
				{
					gray_col = col_y + col_j;
					int new_elem = *(grayImg.data + temp0 + gray_col) - *(subImg.data + temp1 + col_j);
					new_elem = (new_elem > 0 ? new_elem : -new_elem);
					different += new_elem;
				}
			}

			if (minimum > different)
			{
				minimum = different;
				*x = row_x;
				*y = col_y;
			}
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
		cout << "image is Null." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	if (colorImg.channels() != subImg.channels() || colorImg.channels() != 3)
	{
		cout << "channels not fit" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}


	int width = colorImg.cols;
	int height = colorImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	if (width < sub_width || height < sub_height)
	{
		cout << "the subimage is larger than original one" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}



	float minimum = FLT_MAX;
	int diff_height = height - sub_height;
	int diff_width = width - sub_width;
	int row_x, col_y, row_i, col_j;
	int tempelem1, tempelem2, tempelem3, new_elem;
	for (row_x = 0; row_x <= diff_height; row_x++)
	{
		for (col_y = 0; col_y <= diff_width; col_y++)
		{

			int different = 0;
			for (row_i = 0; row_i < sub_height; row_i++)
			{
				int color_row = row_x + row_i;
				int temp0 = color_row*width;
				int temp1 = row_i*sub_width;

				for (col_j = 0; col_j < sub_width; col_j++)
				{
					int temp3 = 3 * (temp0 + col_y + col_j);
					int temp4 = 3 * (temp1 + col_j);


					tempelem1 = colorImg.data[temp3] - subImg.data[temp4];
					tempelem1 = tempelem1 > 0 ? tempelem1 : -tempelem1;

					tempelem2 = colorImg.data[temp3 + 1] - subImg.data[temp4 + 1];
					tempelem2 = tempelem2 > 0 ? tempelem2 : -tempelem2;

					tempelem3 = colorImg.data[temp3 + 2] - subImg.data[temp4 + 2];
					tempelem3 = tempelem3 > 0 ? tempelem3 : -tempelem3;

					new_elem = tempelem1 + tempelem2 + tempelem3;
					different += new_elem;
				}
			}

			if (minimum > different)
			{
				minimum = different;
				(*x) = row_x;
				(*y) = col_y;
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
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_OK;
	}

	if (grayImg.channels() != subImg.channels() || grayImg.channels() != 1)
	{
		cout << "images have different channels" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	if (width < sub_width || height < sub_height)
	{
		cout << "the subimage is larger than original one" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}


	uchar *p = subImg.data, *end = subImg.data + sub_width*sub_height;
	int sum3 = 0;
	while (p < end)
	{
		sum3 += (*p)*(*p);
		p += 1;
	}
	double answer3 = sqrt(sum3);


	float maximum = 0;
	int diff_height = height - sub_height;
	int diff_width = width - sub_width;


	for (int row_x = 0; row_x <= diff_height; row_x++)
	{
		//每一行第一个的big的sum2
		int sum2 = 0;
		for (int ii = 0; ii < sub_height; ii += 1) {
			int temp_row = (ii + row_x)*width;
			for (int jj = 0; jj < sub_width; jj += 1) {
				sum2 += grayImg.data[temp_row + jj] * grayImg.data[temp_row + jj];
			}
		}

		//每一列循环
		for (int col_y = 0; col_y <= diff_width; col_y++)
		{
			int sum1 = 0;
			for (int row_i = 0; row_i < sub_height; row_i++)
			{
				int gray_row = row_x + row_i;
				int tempbig1 = gray_row*width;
				int tempsma1 = row_i*sub_width;

				if (col_y)
				{
					sum2 -= (grayImg.data[tempbig1 + col_y - 1] * grayImg.data[tempbig1 + col_y - 1]);
					sum2 += (grayImg.data[tempbig1 + col_y + sub_width - 1] * grayImg.data[tempbig1 + col_y + sub_width - 1]);
				}

				double answer = answer3*sqrt(sum2);


				for (int col_j = 0; col_j < sub_width; col_j++)
				{
					sum1 += grayImg.data[tempbig1 + col_y + col_j] * subImg.data[tempsma1 + col_j];
				}

				float Relation = sum1 / answer;
				if (maximum < Relation)
				{
					maximum = Relation;
					*x = row_x;
					*y = col_y;
				}
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
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_OK;
	}

	if (grayImg.channels() != subImg.channels() || grayImg.channels() != 1)
	{
		cout << "images have different channels" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	if (width < sub_width || height < sub_height)
	{
		cout << "the subimage is larger than original one" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}


	Mat graygrad_x, graygrad_y, grayangle, subgrad_x, subgrad_y, subangle;


	ustc_CalcGrad(grayImg, graygrad_x, graygrad_y);
	grayangle.create(height, width, CV_32FC1);
	grayangle.setTo(0);
	for (int row_i = 1; row_i < height - 1; row_i++)
	{
		int temp0 = row_i*width;
		for (int col_j = 1; col_j < width - 1; col_j += 1)
		{
			int temp1 = temp0 + col_j;
			float grad_x = ((float*)(graygrad_x.data))[temp1];
			float grad_y = ((float*)(graygrad_y.data))[temp1];
			float angle = atan2(grad_y, grad_x);
			angle = angle * 180 / CV_PI;
			angle = angle > 0 ? angle : 360 + angle;
			((float*)grayangle.data)[temp1] = angle;
		}
	}

	ustc_CalcGrad(subImg, subgrad_x, subgrad_y);
	subangle.create(sub_height, sub_width, CV_32FC1);
	subangle.setTo(0);
	for (int row_i = 1; row_i < sub_height - 1; row_i++)
	{
		int temp0 = row_i*sub_width;
		for (int col_j = 1; col_j < sub_width - 1; col_j += 1)
		{
			int temp1 = temp0 + col_j;
			float grad_x = ((float *)(subgrad_x.data))[temp1];
			float grad_y = ((float *)(subgrad_y.data))[temp1];
			float angle = atan2(grad_y, grad_x);
			angle = angle * 180 / CV_PI;
			angle = angle > 0 ? angle : 360 + angle;
			((float*)subangle.data)[temp1] = angle;
		}
	}



	float minimum = FLT_MAX;
	for (int row_x = 0; row_x <= height - sub_height; row_x++)
	{
		for (int col_y = 0; col_y <= width - sub_width; col_y++)
		{
			float different = 0;

			for (int row_i = 1; row_i < sub_height - 1; row_i++)
			{
				int gray_row = row_x + row_i;
				int temp0 = gray_row*width;
				int temp1 = row_i*sub_width;
				for (int col_j = 1; col_j < sub_width - 1; col_j++)
				{
					int gray_col = col_y + col_j;
					float new_elem = ((float*)grayangle.data)[temp0 + gray_col] - ((float*)subangle.data)[temp1 + col_j];
					new_elem = new_elem > 0 ? new_elem : -new_elem;
					new_elem = new_elem > 180 ? 360 - new_elem : new_elem;
					different += new_elem;
				}
			}
			if (minimum > different)
			{
				minimum = different;
				*x = row_x;
				*y = col_y;
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
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_OK;
	}

	if (grayImg.channels() != subImg.channels() || grayImg.channels() != 1)
	{
		cout << "images have different channels" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	if (width < sub_width || height < sub_height)
	{
		cout << "the subimage is larger than original one" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}


	Mat graygrad_x, graygrad_y, graymag, subgrad_x, subgrad_y, submag;


	ustc_CalcGrad(grayImg, graygrad_x, graygrad_y);
	graymag.create(height, width, CV_32FC1);
	graymag.setTo(0);
	for (int row_i = 1; row_i < height - 1; row_i++)
	{
		int temp0 = row_i*width;
		for (int col_j = 1; col_j < width - 1; col_j++)
		{
			int temp1 = temp0 + col_j;
			float grad_x = ((float *)(graygrad_x.data))[temp1];
			float grad_y = ((float*)(graygrad_y.data))[temp1];
			float mag = sqrt(grad_x*grad_x + grad_y*grad_y);
			((float*)graymag.data)[temp1] = mag;
		}
	}

	ustc_CalcGrad(subImg, subgrad_x, subgrad_y);
	submag.create(sub_height, sub_width, CV_32FC1);
	submag.setTo(0);
	for (int row_i = 1; row_i < sub_height - 1; row_i++)
	{
		int temp0 = row_i*sub_width;
		for (int col_j = 1; col_j < sub_width - 1; col_j++)
		{
			int temp1 = temp0 + col_j;
			float grad_x = ((float*)(subgrad_x.data))[temp1];
			float grad_y = ((float*)(subgrad_y.data))[temp1];
			float mag = sqrt(grad_x*grad_x + grad_y*grad_y);
			((float*)submag.data)[temp1] = mag;
		}
	}



	float minimum = FLT_MAX;
	for (int row_x = 0; row_x <= height - sub_height; row_x++)
	{
		for (int col_y = 0; col_y <= width - sub_width; col_y++)
		{
			float different = 0;

			for (int row_i = 1; row_i < sub_height - 1; row_i++)
			{
				int gray_row = row_x + row_i;
				int temp0 = gray_row*width;
				int temp1 = row_i*sub_width;

				for (int col_j = 1; col_j < sub_width - 1; col_j++)
				{
					int gray_col = col_y + col_j;
					float new_elem = ((float*)graymag.data)[temp0 + gray_col] - ((float*)submag.data)[temp1 + col_j];
					new_elem = new_elem > 0 ? new_elem : -new_elem;
					different += new_elem;
				}
			}
			if (minimum > different)
			{
				minimum = different;
				*x = row_x;
				*y = col_y;
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
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_OK;
	}

	if (grayImg.channels() != subImg.channels() || grayImg.channels() != 1)
	{
		cout << "images have different channels" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	if (width < sub_width || height < sub_height)
	{
		cout << "the subimage is larger than original one" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}


	int sub_hist[256] = { 0 };

	//temp_hist 置零

	//计算 sub_hist;
	for (int ii = 0; ii < sub_height; ++ii)
	{
		int temp0 = ii*sub_width;
		for (int jj = 0; jj < sub_width; ++jj)
		{
			++sub_hist[subImg.data[temp0 + jj]];
		}
	}

	int diff_width = width - sub_width;
	int diff_height = height - sub_height;
	int minimum = 999999999;
	for (int row_x = 0; row_x <= diff_height; ++row_x)
	{
		int	temp_hist[256] = { 0 };


		//每一行row_x算第一个temp_hist
		for (int row_i = 0; row_i < sub_height; ++row_i)
		{
			int temp0 = (row_i + row_x)*width;
			for (int col_j = 0; col_j < sub_width; ++col_j)
			{
				++temp_hist[grayImg.data[temp0 + col_j]];
			}
		}


		for (int col_y = 0; col_y <= diff_width; ++col_y)
		{
			//计算hist变化值
			for (int row_i = 0; row_i < sub_height; ++row_i)
			{
				int row_big = row_x + row_i;
				int temp0 = row_big*width;
				if (col_y)
				{
					temp_hist[grayImg.data[temp0 + sub_width + col_y - 1]]++;
					temp_hist[grayImg.data[temp0 + col_y - 1]]--;  //important
				}
			}

			int different = 0;
			for (int ii = 0; ii < 256; ++ii)
			{
				int diff = temp_hist[ii] - sub_hist[ii];
				different += diff > 0 ? diff : -diff;
			}

			if (minimum > different)
			{
				minimum = different;
				(*x) = row_x;
				(*y) = col_y;
			}
		}
	}
	return SUB_IMAGE_MATCH_OK;

}
