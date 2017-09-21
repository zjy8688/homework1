//#include <stdafx.h>
#include"SubImageMatch.h"


//#define IMG_SHOW
#define MY_OK 1
#define MY_FAIL -1

//函数功能：将bgr图像转化成灰度图像
//bgrImg：彩色图，像素排列顺序是bgr
//grayImg：灰度图，单通道
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_ConvertBgr2Gray(Mat bgrImg, Mat& grayImg)
{
	if (!bgrImg.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}
	if (bgrImg.channels() != 3) {
		cout << "bgrImg channels wrong!" << endl;
		return MY_FAIL;
	}
	if (grayImg.channels() != 1) {
		cout << "grayImg channels wrong!" << endl;
		return MY_FAIL;
	}
	if (bgrImg.depth() != 0) {
		cout << "bgrImg depth wrong!" << endl;
		return MY_FAIL;
	}
	if (grayImg.depth() != 0) {
		cout << "grayImg depth wrong!" << endl;
		return MY_FAIL;
	}
	if (bgrImg.rows != grayImg.rows || bgrImg.cols != grayImg.cols)
	{
		cout << "Image size wrong!" << endl;
		return MY_FAIL;
	}
	
	//int width = bgrImg.cols;
	//int height = bgrImg.rows;
	//int b, g, r, grayVal;
	//int row_i,col_j;

	//for (int row_i = height-1; row_i >= 0; row_i--)
	//{
	//	int row_width = row_i*width;   //重复计算
	//	for (int col_j = width-1; col_j >= 0; col_j--)
	//	{
	//		//int _3_row_width_col = 3 * (row_width + col_j);
	//		//b = bgrImg.data[3 * (row_i * width + col_j) + 0];
	//		//g = bgrImg.data[3 * (row_i * width + col_j) + 1];
	//		//r = bgrImg.data[3 * (row_i * width + col_j) + 2];

	//		//grayVal = bgrImg.data[3 * (row_width + col_j) + 0] * 0.114f
	//		//	+ bgrImg.data[3 * (row_width + col_j) + 1] * 0.587f
	//		//	+ bgrImg.data[3 * (row_width + col_j) + 2] * 0.229f;
	//		grayImg.data[row_width + col_j] =
	//			bgrImg.data[3 * (row_width + col_j) + 0] * 0.114f 
	//			+ bgrImg.data[3 * (row_width + col_j) + 1] * 0.587f 
	//			+ bgrImg.data[3 * (row_width + col_j) + 2] * 0.229f;
	//	}
	//}
	//__m128 a;
	int i = bgrImg.rows*bgrImg.cols-1;
	int _3i = 3 * i;
	for (i; i >= 0; i--)
	{
		grayImg.data[i] =
			bgrImg.data[_3i + 0] * 0.114f
			+ bgrImg.data[_3i + 1] * 0.587f
			+ bgrImg.data[_3i + 2] * 0.229f;
		_3i -= 3;
	}

//#ifdef IMG_SHOW
//	namedWindow("grayImg", 0);
//	imshow("grayImg", grayImg);
//	waitKey(1);
//#endif
	return MY_OK;
}

//函数功能：根据灰度图像计算梯度图像
//grayImg：灰度图，单通道
//gradImg_x：水平方向梯度，浮点类型图像，CV32FC1
//gradImg_y：垂直方向梯度，浮点类型图像，CV32FC1
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_CalcGrad(Mat grayImg, Mat& gradImg_x, Mat& gradImg_y)
{
	if (!grayImg.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}
	if (grayImg.channels() != 1) {
		cout << "grayImg channels wrong!" << endl;
		return MY_FAIL;
	}
	if (gradImg_x.channels() != 1) {
		cout << "gradImg_x channels wrong!" << endl;
		return MY_FAIL;
	}
	if (gradImg_y.channels() != 1) {
		cout << "gradImg_y channels wrong!" << endl;
		return MY_FAIL;
	}
	if (grayImg.depth() != 0) {
		cout << "grayImg depth wrong!" << endl;
		return MY_FAIL;
	}
	if (gradImg_x.depth() != 5) {
		cout << "gradImg_x depth wrong!" << endl;
		return MY_FAIL;
	}
	if (gradImg_y.depth() != 5) {
		cout << "gradImg_y depth wrong!" << endl;
		return MY_FAIL;
	}
	if (grayImg.rows != gradImg_x.rows || grayImg.rows != gradImg_y.rows || grayImg.cols != gradImg_x.cols || grayImg.cols != gradImg_y.cols)
	{
		cout << "Image size wrong!" << endl;
		return MY_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	//int col_j;
	//int row_i = grayImg.rows - 2;
	gradImg_x.setTo(0);
	gradImg_y.setTo(0);
	int i = height*width - width - 2;
	//计算x,y方向梯度图
	for (i; i > width; i--)
	{
		((float*)gradImg_x.data)[i] =
			-grayImg.data[i - width - 1] - 2 *grayImg.data[i - 1] - grayImg.data[i + width - 1]
			+ grayImg.data[i - width + 1] + 2 * grayImg.data[i + 1] + grayImg.data[i + width + 1];
		((float*)gradImg_y.data)[i] =
			-grayImg.data[i - width - 1] - 2 * grayImg.data[i - width] - grayImg.data[i - width + 1]
			+ grayImg.data[i + width - 1] + 2 * grayImg.data[i + width] + grayImg.data[i + width + 1];
	}
	i = height*width - 2 * width;
	for (i; i > width; i -= width)
	{
		((float*)gradImg_x.data)[i] = ((float*)gradImg_y.data)[i] = 
			((float*)gradImg_x.data)[i-1] = ((float*)gradImg_y.data)[i-1]=0;
	}
	//for (row_i = grayImg.rows - 2; row_i >0; row_i--)
	//{
	//	for (col_j = grayImg.cols - 2; col_j >0; col_j--)
	//	{

	//		((float*)gradImg_x.data)[row_i * width + col_j] = 
	//			grayImg.data[(row_i - 1) * width + col_j + 1]
	//			+ 2 * grayImg.data[(row_i)* width + col_j + 1]
	//			+ grayImg.data[(row_i + 1)* width + col_j + 1]
	//			- grayImg.data[(row_i - 1) * width + col_j - 1]
	//			- 2 * grayImg.data[(row_i)* width + col_j - 1]
	//			- grayImg.data[(row_i + 1)* width + col_j - 1];

	//		((float*)gradImg_y.data)[row_i * width + col_j] =
	//			grayImg.data[(row_i + 1) * width + col_j - 1]
	//			+ 2 * grayImg.data[(row_i + 1)* width + col_j]
	//			+ grayImg.data[(row_i + 1)* width + col_j + 1]
	//			- grayImg.data[(row_i - 1) * width + col_j - 1]
	//			- 2 * grayImg.data[(row_i - 1)* width + col_j]
	//			- grayImg.data[(row_i - 1)* width + col_j + 1];
	//	}
	//}

//#ifdef IMG_SHOW
//	Mat gradImg_x_8U(height, width, CV_8UC1);
//	Mat gradImg_y_8U(height, width, CV_8UC1);
//	//为了方便观察，直接取绝对值
//	for (int row_i = 0; row_i < height; row_i++)
//	{
//		for (int col_j = 0; col_j < width; col_j += 1)
//		{
//			int val = ((float*)gradImg_x.data)[row_i * width + col_j];
//			gradImg_x_8U.data[row_i * width + col_j] = abs(val);
//		}
//	}
//	//为了方便观察，直接取绝对值
//	for (int row_i = 0; row_i < height; row_i++)
//	{
//		for (int col_j = 0; col_j < width; col_j += 1)
//		{
//			int val = ((float*)gradImg_y.data)[row_i * width + col_j];
//			gradImg_y_8U.data[row_i * width + col_j] = abs(val);
//		}
//	}
//
//	namedWindow("gradImg_x_8U", 0);
//	imshow("gradImg_x_8U", gradImg_x_8U);
//	waitKey(1);
//	namedWindow("gradImg_y_8U", 0);
//	imshow("gradImg_y_8U", gradImg_y_8U);
//	waitKey(1);
//#endif
	return MY_OK;
}

//函数功能：根据水平和垂直梯度，计算角度和幅值图
//gradImg_x：水平方向梯度，浮点类型图像，CV32FC1
//gradImg_y：垂直方向梯度，浮点类型图像，CV32FC1
//angleImg：角度图，浮点类型图像，CV32FC1
//magImg：幅值图，浮点类型图像，CV32FC1
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y, Mat& angleImg, Mat& magImg)
{
	if (NULL == gradImg_x.data || NULL == gradImg_y.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}
	if (gradImg_x.channels() != 1) {
		cout << "gradImg_x channels wrong!" << endl;
		return MY_FAIL;
	}
	if (gradImg_y.channels() != 1) {
		cout << "gradImg_y channels wrong!" << endl;
		return MY_FAIL;
	}
	if (angleImg.channels() != 1) {
		cout << "angleImg channels wrong!" << endl;
		return MY_FAIL;
	}
	if (magImg.channels() != 1) {
		cout << "magImg channels wrong!" << endl;
		return MY_FAIL;
	}
	if (gradImg_x.depth() != 5) {
		cout << "gradImg_x depth wrong!" << endl;
		return MY_FAIL;
	}
	if (gradImg_y.depth() != 5) {
		cout << "gradImg_y depth wrong!" << endl;
		return MY_FAIL;
	}
	if (angleImg.depth() != 5) {
		cout << "angleImg depth wrong!" << endl;
		return MY_FAIL;
	}
	if (magImg.depth() != 5) {
		cout << "magImg depth wrong!" << endl;
		return MY_FAIL;
	}
	if (gradImg_x.rows != gradImg_y.rows || gradImg_y.rows != angleImg.rows || magImg.rows != angleImg.rows)
	{
		cout << "Image size wrong!" << endl;
		return MY_FAIL;
	}
	if (gradImg_x.cols != gradImg_y.cols || gradImg_y.cols != angleImg.cols || magImg.cols != angleImg.cols)
	{
		cout << "Image size wrong!" << endl;
		return MY_FAIL;
	}
	angleImg.setTo(0);
	magImg.setTo(0);
	int i = gradImg_x.rows*gradImg_x.cols - 1;
	float convert = 180 / CV_PI;
	//计算x,y方向梯度图
	for (i; i > 0; i--)
	{
		((float*)angleImg.data)[i] = atan2(((float*)gradImg_y.data)[i], ((float*)gradImg_x.data)[i]) * convert;
		if (((float*)angleImg.data)[i]<0)((float*)angleImg.data)[i] += 360;

		((float*)magImg.data)[i] = sqrt(((float*)gradImg_x.data)[i] * ((float*)gradImg_x.data)[i] + ((float*)gradImg_y.data)[i] * ((float*)gradImg_y.data)[i]);
	}
	////计算角度图
	//for (int row_i = 1; row_i < height - 1; row_i++)
	//{
	//	for (int col_j = 1; col_j < width - 1; col_j += 1)
	//	{
	//		float grad_x = ((float*)gradImg_x.data)[row_i * width + col_j];
	//		float grad_y = ((float*)gradImg_y.data)[row_i * width + col_j];
	//		float angle = atan2(grad_y, grad_x);
	//		angle *= 180 / CV_PI;
	//		if (angle < 0)angle += 360;
	//		//自己找办法优化三角函数速度，并且转化为角度制，规范化到0-360
	//		((float*)angleImg.data)[row_i * width + col_j] = angle;
	//	}
	//}
	////计算幅值图
	//for (int row_i = 1; row_i < height - 1; row_i++)
	//{
	//	for (int col_j = 1; col_j < width - 1; col_j += 1)
	//	{
	//		float grad_x = ((float*)gradImg_x.data)[row_i * width + col_j];
	//		float grad_y = ((float*)gradImg_y.data)[row_i * width + col_j];
	//		float mag = sqrt(grad_x * grad_x + grad_y * grad_y);
	//		((float*)magImg.data)[row_i * width + col_j] = mag;
	//	}
	//}

//#ifdef IMG_SHOW
//	Mat angleImg_8U(gradImg_x.rows, gradImg_x.cols, CV_8UC1);
//	Mat magImg_8U(gradImg_x.rows, gradImg_x.cols, CV_8UC1);
//	//为了方便观察，进行些许变化
//	for (int row_i = 0; row_i < gradImg_x.rows; row_i++)
//	{
//		for (int col_j = 0; col_j < gradImg_x.cols; col_j += 1)
//		{
//			float angle = ((float*)angleImg.data)[row_i * gradImg_x.cols + col_j];
//			//为了能在8U上显示，缩小到0-180之间
//			angle /= 2;
//			angleImg_8U.data[row_i * gradImg_x.cols + col_j] = angle;
//			magImg_8U.data[row_i * gradImg_x.cols + col_j] = ((float*)magImg.data)[row_i * gradImg_x.cols + col_j];
//		}
//	}
//
//	namedWindow("angleImg_8U", 0);
//	imshow("angleImg_8U", angleImg_8U);
//	waitKey(1);
//	namedWindow("magImg_8U", 0);
//	imshow("magImg_8U", magImg);
//	waitKey(1);
//#endif
	return MY_OK;
}

//函数功能：对灰度图像进行二值化
//grayImg：灰度图，单通道
//binaryImg：二值图，单通道
//th：二值化阈值，高于此值，255，低于此值0
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_Threshold(Mat grayImg, Mat& binaryImg, int th)
{
	if (!grayImg.data|| !binaryImg.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}
	if (grayImg.channels() != 1) {
		cout << "grayImg channels wrong!" << endl;
		return MY_FAIL;
	}
	if (binaryImg.channels() != 1) {
		cout << "binaryImg channels wrong!" << endl;
		return MY_FAIL;
	}
	if (binaryImg.depth() != 0) {
		cout << "binaryImg depth wrong!" << endl;
		return MY_FAIL;
	}
	if (grayImg.depth() != 0) {
		cout << "grayImg depth wrong!" << endl;
		return MY_FAIL;
	}
	if (binaryImg.rows != grayImg.rows || binaryImg.cols != grayImg.cols)
	{
		cout << "Image size wrong!" << endl;
		return MY_FAIL;
	}
	if (th < 0 || th>255)
	{
		cout << "threshold wrong!" << endl;
		return MY_FAIL;
	}
	//int width = grayImg.cols;
	//int height = grayImg.rows;	
	for (int i = grayImg.cols*grayImg.rows - 1; i >= 0; i--)
	{
		binaryImg.data[i] = ((th-1- grayImg.data[i]) >> 31);
		//binaryImg.data[i] = (((grayImg.data[i] - th) >> 31) ) & 255+1;
		//if (grayImg.data[i] > th)binaryImg.data[i] = 255;

	}
	//int binary_th = 50;
	//for (int row_i = 0; row_i < height; row_i++)
	//{
	//	int temp0 = row_i * width;
	//	for (int col_j = 0; col_j < width; col_j += 1)
	//	{
	//		//int pixVal = grayImg.at<uchar>(row_i, col_j);
	//		int temp1 = temp0 + col_j;
	//		int pixVal = grayImg.data[temp1];
	//		int dstVal = 0;
	//		if (pixVal > binary_th)
	//		{
	//			dstVal = 255;
	//		}
	//		else if (pixVal <= binary_th)
	//		{
	//			dstVal = 0;
	//		}
	//		//binaryImg.at<uchar>(row_i, col_j) = dstVal;
	//		binaryImg.data[temp1] = dstVal;
	//	}
	//}

//#ifdef IMG_SHOW
//	namedWindow("binaryImg", 0);
//	imshow("binaryImg", binaryImg);
//	waitKey(1);
//#endif

	return MY_OK;
}

//函数功能：对灰度图像计算直方图
//grayImg：灰度图，单通道
//hist：直方图
//hist_len：直方图的亮度等级，直方图数组的长度
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_CalcHist(Mat grayImg, int* hist, int hist_len)
{
	if (!grayImg.data || !hist)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}
	if (grayImg.channels() != 1) {
		cout << "grayImg channels wrong!" << endl;
		return MY_FAIL;
	}
	if (grayImg.depth() != 0) {
		cout << "grayImg depth wrong!" << endl;
		return MY_FAIL;
	}
	//int width = grayImg.cols;
	//int height = grayImg.rows;
	int i;
	//直方图清零
	for (i = hist_len-1; i >=0; i--)
	{
		hist[i] = 0;
	}
	for (i = grayImg.cols*grayImg.rows - 1; i >= 0; i--)
	{
		hist[grayImg.data[i]]++;
	}
	//计算直方图
	//for (int row_i = 0; row_i < grayImg.rows; row_i++)
	//{
	//	for (int col_j = 0; col_j < grayImg.cols; col_j += 1)
	//	{
	//		int pixVal = grayImg.data[row_i * grayImg.cols + col_j];
	//		hist[pixVal]++;
	//	}
	//}
	return MY_OK;
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
		return MY_FAIL;
	}
	if (subImg.channels() != 1) {
		cout << "subImg channels wrong!" << endl;
		return MY_FAIL;
	}
	if (grayImg.channels() != 1) {
		cout << "grayImg channels wrong!" << endl;
		return MY_FAIL;
	}
	if (subImg.depth() != 0) {
		cout << "subImg depth wrong!" << endl;
		return MY_FAIL;
	}
	if (grayImg.depth() != 0) {
		cout << "grayImg depth wrong!" << endl;
		return MY_FAIL;
	}
	if (subImg.rows>grayImg.rows || subImg.cols>grayImg.cols)
	{
		cout << "Image size wrong!" << endl;
		return MY_FAIL;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int diff_min = INT_MAX;
	int m, n;
	int i_m_width=0;
	int sub_height_m_width = sub_height*width;
	int i_max = height - sub_height + 1;
	int j_max = width - sub_width + 1;
	register int total_diff = 0;
	int sub = 0;
	register short diff;
	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i < i_max; i++)
	{
		for (int j = 0; j < j_max; j++)
		{
			total_diff = 0;
			sub = 0;
			//遍历模板图上的每一个像素
			for (m = 0; m < sub_height; m++)
			{
				for (n = 0; n < sub_width; n++)
				{
					//大图上的像素位置
					//int row_index = i + m;
					//int col_index = j + n;
					//int bigImg_pix = grayImg.data[i_m_width + j + n];
					////模板图上的像素
					//int template_pix = subImg.data[sub++];
					diff = grayImg.data[i_m_width + j + n] - subImg.data[sub++];
					if (diff < 0)diff = 0 - diff;
					total_diff += diff;
				}
				i_m_width += width;
			}
			i_m_width -= sub_height_m_width;
			//i_m_width++;
			if (total_diff < diff_min)
			{
				diff_min = total_diff;
				*x = j;
				*y = i;
			}
		}
		i_m_width += width;
	}
	return MY_OK;
}

//函数功能：利用色彩进行子图匹配
//colorImg：彩色图，三通道
//subImg：模板子图，三通道
//x：最佳匹配子图左上角x坐标
//y：最佳匹配子图左上角y坐标
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int* x, int* y)
{
	if (NULL == colorImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}
	if (subImg.channels() != 3) {
		cout << "subImg channels wrong!" << endl;
		return MY_FAIL;
	}
	if (colorImg.channels() != 3) {
		cout << "colorImg channels wrong!" << endl;
		return MY_FAIL;
	}
	if (subImg.depth() != 0) {
		cout << "subImg depth wrong!" << endl;
		return MY_FAIL;
	}
	if (colorImg.depth() != 0) {
		cout << "colorImg depth wrong!" << endl;
		return MY_FAIL;
	}
	if (subImg.rows>colorImg.rows || subImg.cols>colorImg.cols)
	{
		cout << "Image size wrong!" << endl;
		return MY_FAIL;
	}
	int width = colorImg.cols;
	int height = colorImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int diff_min = INT_MAX;
	int m, n;
	int i_m_width = 0;
	int sub_height_m_width = sub_height*width;
	int i_max = height - sub_height + 1;
	int j_max = width - sub_width + 1;
	register int total_diff = 0;
	int sub = 0;
	register short diff;
	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i < i_max; i++)
	{
		for (int j = 0; j < j_max; j++)
		{
			total_diff = 0;
			sub = 0;
			//遍历模板图上的每一个像素
			for (m = 0; m < sub_height; m++)
			{
				for (n = 0; n < sub_width; n++)
				{
					//大图上的像素位置
					//int row_index = i + m;
					//int col_index = j + n;
					//int bigImg_pix = grayImg.data[i_m_width + j + n];
					////模板图上的像素
					//int template_pix = subImg.data[sub++];
					diff = colorImg.data[3 * (i_m_width + j + n) + 0] - subImg.data[sub++];
					if (diff < 0)diff = 0 - diff;
					total_diff += diff;
					 diff = colorImg.data[3 * (i_m_width + j + n) + 1] - subImg.data[sub++];
					if (diff < 0)diff = 0 - diff;
					 diff = colorImg.data[3 * (i_m_width + j + n) + 2] - subImg.data[sub++];
					if (diff < 0)diff = 0 - diff;
					total_diff += diff;
					//total_diff += abs(colorImg.data[3 * (i_m_width + j + n) + 0] - subImg.data[sub++]);
					//total_diff += abs(colorImg.data[3 * (i_m_width + j + n) + 1] - subImg.data[sub++]);
					//total_diff += abs(colorImg.data[3 * (i_m_width + j + n) + 2] - subImg.data[sub++]);
				}
				i_m_width += width;
			}
			i_m_width -= sub_height_m_width;
			//i_m_width++;
			if (total_diff < diff_min)
			{
				diff_min = total_diff;
				*x = j;
				*y = i;
			}
		}
		i_m_width += width;
	}
	return MY_OK;

	//if (NULL == colorImg.data || NULL == subImg.data)
	//{
	//	cout << "image is NULL." << endl;
	//	return MY_FAIL;
	//}

	//int width = colorImg.cols;
	//int height = colorImg.rows;
	//int sub_width = subImg.cols;
	//int sub_height = subImg.rows;
	//int diff_min = INT_MAX;


	////遍历大图每一个像素，注意行列的起始、终止坐标
	//for (int i = 0; i < height - sub_height + 1; i++)
	//{
	//	for (int j = 0; j < width - sub_width + 1; j++)
	//	{
	//		int total_diff = 0;
	//		//遍历模板图上的每一个像素
	//		for (int m = 0; m < sub_height; m++)
	//		{
	//			for (int n = 0; n < sub_width; n++)
	//			{
	//				//大图上的像素位置
	//				int row_index = i + m;
	//				int col_index = j + n;
	//				int bigImg_pix_b = colorImg.data[3 * (row_index * width + col_index) + 0];
	//				int bigImg_pix_g = colorImg.data[3 * (row_index * width + col_index) + 1];
	//				int bigImg_pix_r = colorImg.data[3 * (row_index * width + col_index) + 2];
	//				//模板图上的像素
	//				int template_pix_b = subImg.data[3 * (m * sub_width + n) + 0];
	//				int template_pix_g = subImg.data[3 * (m * sub_width + n) + 1];
	//				int template_pix_r = subImg.data[3 * (m * sub_width + n) + 2];

	//				total_diff += abs(bigImg_pix_b - template_pix_b);
	//				total_diff += abs(bigImg_pix_g - template_pix_g);
	//				total_diff += abs(bigImg_pix_r - template_pix_r);
	//			}
	//		}
	//		if (total_diff < diff_min)
	//		{
	//			diff_min = total_diff;
	//			*x = j;
	//			*y = i;
	//		}
	//	}
	//}
	//return MY_OK;
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
		return MY_FAIL;
	}
	if (subImg.channels() != 1) {
		cout << "subImg channels wrong!" << endl;
		return MY_FAIL;
	}
	if (grayImg.channels() != 1) {
		cout << "grayImg channels wrong!" << endl;
		return MY_FAIL;
	}
	if (subImg.depth() != 0) {
		cout << "subImg depth wrong!" << endl;
		return MY_FAIL;
	}
	if (grayImg.depth() != 0) {
		cout << "grayImg depth wrong!" << endl;
		return MY_FAIL;
	}
	if (subImg.rows>grayImg.rows || subImg.cols>grayImg.cols)
	{
		cout << "Image size wrong!" << endl;
		return MY_FAIL;
	}


	int width = grayImg.cols;
	int height = grayImg.rows;
	//uchar gray_array[1000*1000];
	//for (int i = 0; i < width*height; i++)
	//{
	//	gray_array[i] = grayImg.data[i];
	//}
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	//uchar sub_array[1000*1000];
	//for (int i = 0; i < sub_width*sub_height; i++)
	//{
	//	sub_array[i] = grayImg.data[i];
	//}
	int diff_min = INT_MAX;
	int m, n;
	int i_m_width = 0;
	int sub_height_m_width = sub_height*width;
	int i_max = height - sub_height + 1;
	int j_max = width - sub_width + 1;
	float corr_max=0;
	int sum_st = 0;
	int sum_s = 0;
	int sum_t = 0;
	float corr = 0;
	uchar bigImg_pix;
	uchar template_pix;
	int sub = 0;
	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i < i_max; i++)
	{
		for (int j = 0; j < j_max; j++)
		{
			sub = 0;
			sum_s = sum_t = sum_st = 0;
			//遍历模板图上的每一个像素
			for (m = 0; m < sub_height; m++)
			{
				for (n = 0; n < sub_width; n++)
				{
					//大图上的像素位置
					bigImg_pix = grayImg.data[i_m_width + j + n];
					//模板图上的像素
					template_pix = subImg.data[sub++];
					sum_st += bigImg_pix*template_pix;
					sum_s += bigImg_pix*bigImg_pix;
					sum_t += template_pix*template_pix;
				}
				i_m_width += width;
			}
			i_m_width -= sub_height_m_width;
			corr = sum_st / sqrt(sum_s) / sqrt(sum_t);
			if (corr > corr_max)
			{
				corr_max = corr;
				*x = j;
				*y = i;
			}
		}
		i_m_width += width;
	}
	////遍历大图每一个像素，注意行列的起始、终止坐标
	//for (int i = 0; i < i_max; i++)
	//{
	//	for (int j = 0; j < j_max; j++)
	//	{
	//		int sum_st = 0;
	//		int sum_s = 0;
	//		int sum_t = 0;
	//		float corr = 0;
	//		//遍历模板图上的每一个像素
	//		for (int m = 0; m < sub_height; m++)
	//		{
	//			for (int n = 0; n < sub_width; n++)
	//			{
	//				//大图上的像素位置
	//				int row_index = i + m;
	//				int col_index = j + n;
	//				int bigImg_pix = grayImg.data[(i + m) * width + j + n];
	//				//模板图上的像素
	//				int template_pix = subImg.data[m * sub_width + n];

	//				sum_st += bigImg_pix*template_pix;
	//				sum_s += bigImg_pix*bigImg_pix;
	//				sum_t += template_pix*template_pix;
	//			}
	//		}
	//		//存储当前像素位置的匹配误差
	//		corr = sum_st / sqrt(sum_s) / sqrt(sum_t);
	//		if (corr > corr_max)
	//		{
	//			corr_max = corr;
	//			*x = j;
	//			*y = i;
	//		}
	//	}
	//}
	//return MY_OK;
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
		return MY_FAIL;
	}
	if (subImg.channels() != 1) {
		cout << "subImg channels wrong!" << endl;
		return MY_FAIL;
	}
	if (grayImg.channels() != 1) {
		cout << "grayImg channels wrong!" << endl;
		return MY_FAIL;
	}
	if (subImg.depth() != 5) {
		cout << "subImg depth wrong!" << endl;
		return MY_FAIL;
	}
	if (grayImg.depth() != 5) {
		cout << "grayImg depth wrong!" << endl;
		return MY_FAIL;
	}
	if (subImg.rows>grayImg.rows || subImg.cols>grayImg.cols)
	{
		cout << "Image size wrong!" << endl;
		return MY_FAIL;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int diff_min = INT_MAX;
	int m, n;
	int i_m_width = 0;
	int sub_height_m_width = sub_height*width;
	int i_max = height - sub_height + 1;
	int j_max = width - sub_width + 1;
	register short diff;
	register int total_diff = 0;
	int sub = 0;
	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i < i_max; i++)
	{
		for (int j = 0; j < j_max; j++)
		{
			total_diff = 0;
			sub = 0;
			//遍历模板图上的每一个像素
			for (m = 0; m < sub_height; m++)
			{
				for (n = 0; n < sub_width; n++)
				{
					//大图上的像素位置
					//int row_index = i + m;
					//int col_index = j + n;
					//int bigImg_pix = grayImg.data[i_m_width + j + n];
					////模板图上的像素
					//int template_pix = subImg.data[sub++];
					diff =((float*)grayImg.data)[i_m_width + j + n] - ((float*)subImg.data)[sub++];
					if (diff < 0)diff = 0 - diff;
					if (diff > 180)diff = 360 - diff;
					total_diff += diff;
				}
				i_m_width += width;
			}
			i_m_width -= sub_height_m_width;
			//i_m_width++;
			if (total_diff < diff_min)
			{
				diff_min = total_diff;
				*x = j;
				*y = i;
			}
		}
		i_m_width += width;
	}
	return MY_OK;
	//if (NULL == grayImg.data || NULL == subImg.data)
	//{
	//	cout << "image is NULL." << endl;
	//	return MY_FAIL;
	//}

	//int width = grayImg.cols;
	//int height = grayImg.rows;
	//int sub_width = subImg.cols;
	//int sub_height = subImg.rows;
	//int diff_min = INT_MAX;
	//int flag;

	////该图用于记录每一个像素位置的匹配误差
	//Mat searchImg(height, width, CV_32FC1);
	////匹配误差初始化
	//searchImg.setTo(FLT_MAX);
	////两个图的xy方向梯度
	//Mat graygradImg_x(grayImg.rows, grayImg.cols, CV_32FC1);
	//Mat graygradImg_y(grayImg.rows, grayImg.cols, CV_32FC1);
	//Mat subgradImg_x(subImg.rows, subImg.cols, CV_32FC1);
	//Mat subgradImg_y(subImg.rows, subImg.cols, CV_32FC1);

	//Mat grayangleImg(grayImg.rows, grayImg.cols, CV_32FC1);
	//Mat subangleImg(subImg.rows, subImg.cols, CV_32FC1);
	//grayangleImg.setTo(0);
	//subangleImg.setTo(0);
	//flag = ustc_CalcGrad(grayImg, graygradImg_x, graygradImg_y);  //计算大图梯度
	//flag = ustc_CalcGrad(subImg, subgradImg_x, subgradImg_y);  //计算小图梯度

	////计算大图角度图
	//for (int row_i = 1; row_i < height - 1; row_i++)
	//{
	//	for (int col_j = 1; col_j < width - 1; col_j += 1)
	//	{
	//		float grad_x = ((float*)graygradImg_x.data)[row_i * width + col_j];
	//		float grad_y = ((float*)graygradImg_y.data)[row_i * width + col_j];
	//		float angle = atan2(grad_y, grad_x);
	//		angle *= 180 / CV_PI;
	//		if (angle < 0)angle += 360;
	//		//自己找办法优化三角函数速度，并且转化为角度制，规范化到0-360
	//		((float*)grayangleImg.data)[row_i * width + col_j] = angle;
	//	}
	//}
	////计算小图角度图
	//for (int row_i = 1; row_i < sub_height - 1; row_i++)
	//{
	//	for (int col_j = 1; col_j < sub_width - 1; col_j += 1)
	//	{
	//		float grad_x = ((float*)subgradImg_x.data)[row_i * sub_width + col_j];
	//		float grad_y = ((float*)subgradImg_y.data)[row_i * sub_width + col_j];
	//		float angle = atan2(grad_y, grad_x);
	//		angle *= 180 / CV_PI;
	//		if (angle < 0)angle += 360;
	//		//自己找办法优化三角函数速度，并且转化为角度制，规范化到0-360
	//		((float*)subangleImg.data)[row_i * sub_width + col_j] = angle;
	//	}
	//}

	////遍历大图每一个像素，注意行列的起始、终止坐标
	//for (int i = 0; i < height - sub_height + 1; i++)
	//{
	//	for (int j = 0; j < width - sub_width + 1; j++)
	//	{
	//		int total_diff = 0;
	//		//遍历模板图上的每一个像素
	//		for (int m = 0; m < sub_height; m++)
	//		{
	//			for (int n = 0; n < sub_width; n++)
	//			{
	//				//大图上的像素位置
	//				int row_index = i + m;
	//				int col_index = j + n;
	//				int bigImg_pix = ((float*)grayangleImg.data)[row_index * width + col_index];
	//				//模板图上的像素
	//				int template_pix = ((float*)subangleImg.data)[m * sub_width + n];

	//				total_diff += abs(bigImg_pix - template_pix);
	//			}
	//		}
	//		//存储当前像素位置的匹配误差
	//		((float*)searchImg.data)[i * width + j] = total_diff;
	//		if (total_diff < diff_min)
	//		{
	//			diff_min = total_diff;
	//			*x = j;
	//			*y = i;
	//		}
	//	}
	//}
	//return MY_OK;
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
		return MY_FAIL;
	}
	if (subImg.channels() != 1) {
		cout << "subImg channels wrong!" << endl;
		return MY_FAIL;
	}
	if (grayImg.channels() != 1) {
		cout << "grayImg channels wrong!" << endl;
		return MY_FAIL;
	}
	if (subImg.depth() != 5) {
		cout << "subImg depth wrong!" << endl;
		return MY_FAIL;
	}
	if (grayImg.depth() != 5) {
		cout << "grayImg depth wrong!" << endl;
		return MY_FAIL;
	}
	if (subImg.rows>grayImg.rows || subImg.cols>grayImg.cols)
	{
		cout << "Image size wrong!" << endl;
		return MY_FAIL;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int diff_min = INT_MAX;
	int m, n;
	int i_m_width = 0;
	int sub_height_m_width = sub_height*width;
	int i_max = height - sub_height + 1;
	int j_max = width - sub_width + 1;
	short diff;
	int total_diff = 0;
	int sub = 0;
	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i < i_max; i++)
	{
		for (int j = 0; j < j_max; j++)
		{
			total_diff = 0;
			sub = 0;
			//遍历模板图上的每一个像素
			for (m = 0; m < sub_height; m++)
			{
				for (n = 0; n < sub_width; n++)
				{
					//大图上的像素位置
					//int row_index = i + m;
					//int col_index = j + n;
					//int bigImg_pix = grayImg.data[i_m_width + j + n];
					////模板图上的像素
					//int template_pix = subImg.data[sub++];
					diff = ((float*)grayImg.data)[i_m_width + j + n] - ((float*)subImg.data)[sub++];
					if (diff < 0)diff = 0 - diff;
					total_diff += diff;
				}
				i_m_width += width;
			}
			i_m_width -= sub_height_m_width;
			//i_m_width++;
			if (total_diff < diff_min)
			{
				diff_min = total_diff;
				*x = j;
				*y = i;
			}
		}
		i_m_width += width;
	}
	return MY_OK;
	//if (NULL == grayImg.data || NULL == subImg.data)
	//{
	//	cout << "image is NULL." << endl;
	//	return MY_FAIL;
	//}

	//int width = grayImg.cols;
	//int height = grayImg.rows;
	//int sub_width = subImg.cols;
	//int sub_height = subImg.rows;
	//int diff_min = INT_MAX;
	//int flag;

	////该图用于记录每一个像素位置的匹配误差
	//Mat searchImg(height, width, CV_32FC1);
	////匹配误差初始化
	//searchImg.setTo(FLT_MAX);
	////两个图的xy方向梯度
	//Mat graymagImg(grayImg.rows, grayImg.cols, CV_32FC1);
	//Mat submagImg(subImg.rows, subImg.cols, CV_32FC1);
	//graymagImg.setTo(0);
	//submagImg.setTo(0);
	//Mat graygradImg_x(grayImg.rows, grayImg.cols, CV_32FC1);
	//Mat graygradImg_y(grayImg.rows, grayImg.cols, CV_32FC1);
	//Mat subgradImg_x(subImg.rows, subImg.cols, CV_32FC1);
	//Mat subgradImg_y(subImg.rows, subImg.cols, CV_32FC1);
	//flag = ustc_CalcGrad(grayImg, graygradImg_x, graygradImg_y);  //计算大图梯度
	//flag = ustc_CalcGrad(subImg, subgradImg_x, subgradImg_y);  //计算小图梯度

	////计算大图幅度图
	//for (int row_i = 1; row_i < height - 1; row_i++)
	//{
	//	for (int col_j = 1; col_j < width - 1; col_j += 1)
	//	{
	//		float grad_x = ((float*)graygradImg_x.data)[row_i * width + col_j];
	//		float grad_y = ((float*)graygradImg_y.data)[row_i * width + col_j];
	//		float mag = sqrt(grad_x*grad_x + grad_y *grad_y);
	//		//自己找办法优化三角函数速度，并且转化为角度制，规范化到0-360
	//		((float*)graymagImg.data)[row_i * width + col_j] = mag;
	//	}
	//}
	////计算小图幅度图
	//for (int row_i = 1; row_i < sub_height - 1; row_i++)
	//{
	//	for (int col_j = 1; col_j < sub_width - 1; col_j += 1)
	//	{
	//		float grad_x = ((float*)subgradImg_x.data)[row_i * sub_width + col_j];
	//		float grad_y = ((float*)subgradImg_y.data)[row_i * sub_width + col_j];
	//		float mag = sqrt(grad_x*grad_x + grad_y *grad_y);
	//		//自己找办法优化三角函数速度，并且转化为角度制，规范化到0-360
	//		((float*)submagImg.data)[row_i * sub_width + col_j] = mag;
	//	}
	//}

	////遍历大图每一个像素，注意行列的起始、终止坐标
	//for (int i = 0; i < height - sub_height + 1; i++)
	//{
	//	for (int j = 0; j < width - sub_width + 1; j++)
	//	{
	//		int total_diff = 0;
	//		//遍历模板图上的每一个像素
	//		for (int m = 0; m < sub_height; m++)
	//		{
	//			for (int n = 0; n < sub_width; n++)
	//			{
	//				//大图上的像素位置
	//				int row_index = i + m;
	//				int col_index = j + n;
	//				int bigImg_pix = ((float*)graymagImg.data)[row_index * width + col_index];
	//				//模板图上的像素
	//				int template_pix = ((float*)submagImg.data)[m * sub_width + n];

	//				total_diff += abs(bigImg_pix - template_pix);
	//			}
	//		}
	//		//存储当前像素位置的匹配误差
	//		((float*)searchImg.data)[i * width + j] = total_diff;
	//		if (total_diff < diff_min)
	//		{
	//			diff_min = total_diff;
	//			*x = j;
	//			*y = i;
	//		}
	//	}
	//}
	//return MY_OK;
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
		return MY_FAIL;
	}
	if (subImg.channels() != 1) {
		cout << "subImg channels wrong!" << endl;
		return MY_FAIL;
	}
	if (grayImg.channels() != 1) {
		cout << "grayImg channels wrong!" << endl;
		return MY_FAIL;
	}
	if (subImg.depth() != 0) {
		cout << "subImg depth wrong!" << endl;
		return MY_FAIL;
	}
	if (grayImg.depth() != 0) {
		cout << "grayImg depth wrong!" << endl;
		return MY_FAIL;
	}
	if (subImg.rows>grayImg.rows || subImg.cols>grayImg.cols)
	{
		cout << "Image size wrong!" << endl;
		return MY_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int diff_min = INT_MAX;
	int flag;
	int minorhist[256];    //子图的直方图
	int subhist[256];      //目标框的直方图
	Mat minorImg(subImg.rows, subImg.cols, CV_8UC1);   //子图
	int i_max = height - sub_height + 1;
	int j_max = width - sub_width + 1;
	int total_diff;
	flag = ustc_CalcHist(subImg, subhist, 256);     //目标图直方图


	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i < i_max; i++)
	{
		for (int j = 0; j < j_max; j++)
		{
			total_diff = 0;
			//遍历模板图上的每一个像素
			for (int m = 0; m < sub_height; m++)
			{
				for (int n = 0; n < sub_width; n++)
				{
					//大图上的像素位置
					minorImg.data[m * sub_width + n] = grayImg.data[(i + m) * width + j + n];
				}
			}
			for (int k = 255; k >= 0; k--)
			{
				minorhist[k] = 0;
			}
			for (int k = minorImg.cols*minorImg.rows - 1; k >= 0; k--)
			{
				minorhist[minorImg.data[k]]++;
			}
			//ustc_CalcHist(minorImg, minorhist, 256);    //计算子图的直方图
			for (int m = 0; m < 256; m++)
			{
				total_diff += abs(minorhist[m] - subhist[m]);
			}
			//存储当前像素位置的匹配误差
			if (total_diff < diff_min)
			{
				diff_min = total_diff;
				*x = j;
				*y = i;
			}
		}
	}
	return MY_OK;
}
