
#include "SubImageMatch.h"
#define SUB_IMAGE_MATCH_OK 1
#define SUB_IMAGE_MATCH_FAIL -1

float SqrtByRSQRTSS(float a);
float InvSqrt(float x);
//函数功能：将bgr图像转化成灰度图像
//bgrImg：彩色图，像素排列顺序是bgr
//grayImg：灰度图，单通道
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_ConvertBgr2Gray(Mat bgrImg, Mat& grayImg)
{
	if (bgrImg.data == NULL)
	{
		printf("Image is NULL!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = bgrImg.cols;
	int height = bgrImg.rows;
	for (int row_i = height - 1; row_i >= 0; row_i--)
	{
		for (int col_j = width - 1; col_j >= 0; col_j--)
		{
			int b = bgrImg.data[3 * (row_i * width + col_j) + 0];
			int g = bgrImg.data[3 * (row_i * width + col_j) + 1];
			int r = bgrImg.data[3 * (row_i * width + col_j) + 2];

			int grayVal = (b * 30 + 150 * g + 76 * r) >> 8;
			grayImg.data[row_i * width + col_j] = grayVal;
		}
	}

#ifdef IMG_SHOW
	namedWindow("grayImg", WINDOW_NORMAL);
	imshow("grayImag", grayImg);
	waitKey(0);
#endif

	return SUB_IMAGE_MATCH_OK;
}

//函数功能：根据灰度图像计算梯度图像
//grayImg：灰度图，单通道
//gradImg_x：水平方向梯度，浮点类型图像，CV32FC1
//gradImg_y：垂直方向梯度，浮点类型图像，CV32FC1
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_CalcGrad(Mat grayImg, Mat& gradImg_x, Mat& gradImg_y)
{
	if (grayImg.data == NULL)
	{
		printf("image is NULL!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	//x方向的梯度
	gradImg_x.setTo(0);
	gradImg_y.setTo(0);

	for (int row_i = 1; row_i < height - 1; row_i++)
	{
		for (int col_j = 1; col_j < width - 1; col_j++)
		{
			int grad_x = 
				grayImg.data[(row_i - 1) * width + col_j + 1]
				+ 2 * grayImg.data[(row_i)* width + col_j + 1]
				+ grayImg.data[(row_i + 1)* width + col_j + 1]
				- grayImg.data[(row_i - 1) * width + col_j - 1]
				- 2 * grayImg.data[(row_i)* width + col_j - 1]
				- grayImg.data[(row_i + 1)* width + col_j - 1];

			((float*)gradImg_x.data)[row_i * width + col_j] = grad_x;

			int grad_y =
				grayImg.data[(row_i + 1) * width + col_j - 1]
				+ 2 * grayImg.data[(row_i + 1) * width + col_j]
				+ grayImg.data[(row_i + 1) * width + col_j + 1]
				- grayImg.data[(row_i - 1) * width + col_j - 1]
				- 2 * grayImg.data[(row_i + 1) * width + col_j]
				- grayImg.data[(row_i + 1) * width + col_j + 1];

			((float*)gradImg_y.data)[row_i * width + col_j] = grad_y;
		}
	}

#ifdef IMG_SHOW
	Mat gradImg_x_8U(height, width, CV_8UC1);
	Mat gradImg_y_8U(height, width, CV_8UC1);
	for (int row_i = 0; row_i < height; row_i++)
	{
		for (int col_j = 0; col_j < width; col_j++)
		{
			int val_x = ((float*)gradImg_x.data)[row_i * width + col_j];
			int val_y = ((float*)gradImg_y.data)[row_i * width + col_j];
			gradImg_x_8U.data[row_i * width + col_j] = abs(val_x);
			gradImg_y_8U.data[row_i * width + col_j] = abs(val_y);
		}
	}

	namedWindow("gradImg_x_8U", 0);
	imshow("gradImg_x_8U", gradImg_x_8U);
	waitKey(100);

	namedWindow("gradImg_y_8U", 0);
	imshow("gradImg_y_8U", gradImg_y_8U);
	waitKey(100);
#endif

	return SUB_IMAGE_MATCH_OK;
}

//函数功能：根据水平和垂直梯度，计算角度和幅值图
//gradImg_x：水平方向梯度，浮点类型图像，CV32FC1
//gradImg_y：垂直方向梯度，浮点类型图像，CV32FC1
//angleImg：角度图，浮点类型图像，CV32FC1
//magImg：幅值图，浮点类型图像，CV32FC1
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y, Mat& angleImg, Mat& magImg)
{
	if (gradImg_x.data == NULL || gradImg_y.data == NULL)
	{
		printf("data is NULL!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = gradImg_x.cols;
	int height = gradImg_x.rows;

	//计算角度图
	angleImg.setTo(0);

	for (int row_i = 1; row_i < height - 1; row_i++)
	{
		for (int col_j = 1; col_j < width - 1; col_j++)
		{
			float grad_x = ((float*)gradImg_x.data)[row_i * width + col_j];
			float grad_y = ((float*)gradImg_y.data)[row_i * width + col_j];
			float angle = atan2(grad_y, grad_x);
			if (angle > 0.0)
			{
				angle = 180 * angle / CV_PI;
			}
			else
			{
				angle = 360 + 180 * angle / CV_PI;
			}
			((float*)angleImg.data)[row_i * width + col_j] = angle;

			float mag = SqrtByRSQRTSS(grad_y * grad_y + grad_x * grad_x);
			((float*)magImg.data)[row_i * width + col_j] = mag;
		}
	}

#ifdef IMG_SHOW
	Mat angleImg_8U(height, width, CV_8UC1);
	for (int row_i = 0; row_i < height; row_i++)
	{
		for (int col_j = 0; col_j < width; col_j++)
		{
			//angleImg_8U.data[row_i * width + col_j] = 0.5 * angleImg.data[row_i * width + col_j];
			float angle = ((float*)angleImg.data)[row_i * width + col_j];
			angle *= 180 / CV_PI;
			angle += 180;
			//为了能在8U上显示，缩小到0-180之间
			angle /= 2;
			angleImg_8U.data[row_i * width + col_j] = angle;
		}
	}

	namedWindow("angleImg_8U",0);
	imshow("angleImg_8U", angleImg_8U);
	waitKey(100);

	namedWindow("magImg", 0);
	imshow("magImg", magImg);
	waitKey();
#endif

	return SUB_IMAGE_MATCH_OK;
}

//函数功能：对灰度图像进行二值化
//grayImg：灰度图，单通道
//binaryImg：二值图，单通道
//th：二值化阈值，高于此值，255，低于此值0
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_Threshold(Mat grayImg, Mat& binaryImg, int th)
{
	if (grayImg.data == NULL)
	{
		printf("image is NULL\n");
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int heigth = grayImg.rows;

	for (int row_i = heigth - 1; row_i >= 0; row_i--)
	{
		int temp0 = row_i * width;
		for (int col_j = width - 1; col_j >= 0; col_j--)
		{
			int temp1 = temp0 + col_j;
			int pixVal = grayImg.data[temp1];
			int dstVal = 0;
			if (pixVal > th)
			{
				dstVal = 255;
			}
			else
			{
				dstVal = 0;
			}
			binaryImg.data[temp1] = dstVal;
		}
	}
#ifdef IMG_SHOW
	namedWindow("binaryImg", 0);
	imshow("binaryImg", binaryImg);
	waitKey();
#endif

	return SUB_IMAGE_MATCH_OK;
}

//函数功能：对灰度图像计算直方图
//grayImg：灰度图，单通道
//hist：直方图
//hist_len：直方图的亮度等级，直方图数组的长度
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_CalcHist(Mat grayImg, int* hist, int hist_len)
{
	if (grayImg.data == NULL||hist == NULL)
	{
		printf("image is NULL!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}	
	int width = grayImg.cols;
	int height = grayImg.rows;

	for (int i = hist_len - 1; i >= 0; i--)
	{
		hist[i] = 0;
	}

	for (int row_i = height - 1; row_i >= 0; row_i--)
	{
		for (int col_j = width - 1; col_j >= 0; col_j--)
		{
			int pixVal = grayImg.data[row_i * width + col_j];
			hist[pixVal]++;
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
	if (grayImg.data == NULL || subImg.data == NULL)
	{
		printf("image is NULL!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	int sum = INT_MAX;
	if (width <= sub_width || height <= sub_height)
	{
		printf("subgraph is bigger than original picture!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}

	//Mat searchImg(height, width, CV_32FC1);
	//searchImg.setTo(FLT_MAX);

	for (int row_i = height - sub_height - 1; row_i >= 0; row_i--)
	{
		for (int col_j = width - sub_width - 1; col_j >= 0; col_j--)
		{
			int total_diff = 0;
			for (int sub_row_i = sub_height - 1; sub_row_i >= 0; sub_row_i--)
			{
				for (int sub_col_j = sub_width - 1; sub_col_j >= 0; sub_col_j--)
				{
					int row_index = row_i + sub_row_i;
					int col_index = col_j + sub_col_j;
					int bigImg_pix = grayImg.data[row_index * width + col_index];
					int template_pix = subImg.data[sub_row_i * sub_width + sub_col_j];
					total_diff += abs(bigImg_pix - template_pix);
				}
			}
			//((float*)searchImg.data)[row_i * width + col_j] = total_diff;
			if (total_diff < sum)
			{
				*x = col_j+1;
				*y = row_i+1;
				sum = total_diff;
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
	if (colorImg.data == NULL || subImg.data == NULL)
	{
		printf("image is NULL!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = colorImg.cols;
	int height = colorImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	int sum = INT_MAX;
	if (width <= sub_width || height <= sub_height)
	{
		printf("subgraph is bigger than original picture!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}

	//Mat searchImg(height, width, CV_32FC1);
	//searchImg.setTo(FLT_MAX);

	for (int row_i = height - sub_height - 1; row_i >= 0; row_i--)
	{
		for (int col_j = width - sub_width - 1; col_j >= 0; col_j--)
		{
			int total_diff = 0;
			for (int sub_row_i = sub_height - 1; sub_row_i >= 0; sub_row_i--)
			{
				for (int sub_col_j = sub_width - 1; sub_col_j >= 0; sub_col_j--)
				{
					int row_index = row_i + sub_row_i;
					int col_index = col_j + sub_col_j;
					int bigImg_pix_b = colorImg.data[3 * (row_index * width + col_index) + 0];
					int bigImg_pix_g = colorImg.data[3 * (row_index * width + col_index) + 1];
					int bigImg_pix_r = colorImg.data[3 * (row_index * width + col_index) + 2];
					int template_pix_b = subImg.data[3 * (sub_row_i * sub_width + sub_col_j) + 0];
					int template_pix_g = subImg.data[3 * (sub_row_i * sub_width + sub_col_j) + 1];
					int template_pix_r = subImg.data[3 * (sub_row_i * sub_width + sub_col_j) + 2];
					total_diff += (abs(bigImg_pix_b - template_pix_b) + abs(bigImg_pix_g - template_pix_g) + abs(bigImg_pix_r - template_pix_r));
				}
			}
			//((float*)searchImg.data)[row_i * width + col_j] = total_diff;
			if (total_diff < sum)
			{
				*x = col_j+1;
				*y = row_i+1;
				sum = total_diff;
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
	if (grayImg.data == NULL || subImg.data == NULL)
	{
		printf("image is NULL!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	float rtag = 0;
	if (width <= sub_width || height <= sub_height)
	{
		printf("subgraph is bigger than original picture!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}

	//Mat searchImg(height, width, CV_32FC1);
	//searchImg.setTo(FLT_MAX);

	for (int row_i = height - sub_height - 1; row_i >= 0; row_i--)
	{
		for (int col_j = width - sub_width - 1; col_j >= 0; col_j--)
		{
			float rtemp = 0;
			int product_sum = 0;//乘积和
			int square_sum_bigImg = 0;//大图灰度值平方和
			int square_sum_subImg = 0;//模板子图灰度值平方和
			for (int sub_row_i = sub_height - 1; sub_row_i >= 0; sub_row_i--)
			{
				for (int sub_col_j = sub_width - 1; sub_col_j >= 0; sub_col_j--)
				{
					int row_index = row_i + sub_row_i;
					int col_index = col_j + sub_col_j;
					int bigImg_pix = grayImg.data[row_index * width + col_index];//大图灰度值
					int template_pix = subImg.data[sub_row_i * sub_width + sub_col_j];//模板子图灰度值
					product_sum += bigImg_pix*template_pix;
					square_sum_bigImg += bigImg_pix*bigImg_pix;
					square_sum_subImg += template_pix*template_pix;
					//total_diff += abs(bigImg_pix - template_pix);
				}
			}
			//((float*)searchImg.data)[row_i * width + col_j] = total_diff;
			rtemp = product_sum * InvSqrt(square_sum_bigImg * square_sum_subImg);
			if (rtemp > rtag)
			{
				*x = col_j+1;
				*y = row_i+1;
				rtag = rtemp;
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
	if (grayImg.data == NULL || subImg.data == NULL)
	{
		printf("image is NULL!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	if (width <= sub_width || height <= sub_height)
	{
		printf("subgraph is bigger than original picture!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}

	//计算大图的梯度和角度
	Mat gradImg_x(height, width, CV_32FC1);
	Mat gradImg_y(height, width, CV_32FC1);
	Mat angleImg(height, width, CV_32FC1);
	Mat magImg(height, width, CV_32FC1);

	int grad = ustc_CalcGrad(grayImg, gradImg_x, gradImg_y);
	if (grad == SUB_IMAGE_MATCH_FAIL)
	{
		printf("something is wrong with the gradImg!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}

	int angle = ustc_CalcAngleMag(gradImg_x, gradImg_y, angleImg, magImg);
	if (angle == SUB_IMAGE_MATCH_FAIL)
	{
		printf("something is wrong with the angleImg!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}
	
	//计算子图的梯度和角度
	Mat sub_gradImg_x(sub_height, sub_width, CV_32FC1);
	Mat sub_gradImg_y(sub_height, sub_width, CV_32FC1);
	Mat sub_angleImg(sub_height, sub_width, CV_32FC1);
	Mat sub_magImg(sub_height, sub_width, CV_32FC1);

	int sub_grad = ustc_CalcGrad(subImg, sub_gradImg_x, sub_gradImg_y);
	if (sub_grad == SUB_IMAGE_MATCH_FAIL)
	{
		printf("something is wrong with the sub_gradImg!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}
	int sub_angle = ustc_CalcAngleMag(sub_gradImg_x, sub_gradImg_y, sub_angleImg, sub_magImg);
	if (sub_angle == SUB_IMAGE_MATCH_FAIL)
	{
		printf("something is wrong with the sub_angleImg!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}

	/*
	int flag = ustc_SubImgMatch_gray(angleImg, sub_angleImg, x, y);
	if (flag == SUB_IMAGE_MATCH_FAIL)
	{
		printf("somgthing is wrong with the matching!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}
	*/

	int sum = INT_MAX;

	for (int row_i = height - sub_height - 1; row_i >= 0; row_i--)
	{
		for (int col_j = width - sub_width - 1; col_j >= 0; col_j--)
		{
			int total_diff = 0;
			for (int sub_row_i = sub_height - 1; sub_row_i >= 0; sub_row_i--)
			{
				for (int sub_col_j = sub_width - 1; sub_col_j >= 0; sub_col_j--)
				{
					int row_index = row_i + sub_row_i;
					int col_index = col_j + sub_col_j;
					int bigImg_pix = grayImg.data[row_index * width + col_index];
					int template_pix = subImg.data[sub_row_i * sub_width + sub_col_j];
					
					total_diff += abs(bigImg_pix - template_pix);
					
				}
			}
			//((float*)searchImg.data)[row_i * width + col_j] = total_diff;
			if (total_diff < sum)
			{ 
				*x = col_j+1;
				*y = row_i+1;
				sum = total_diff;
			}
		}
	}
	//printf("total_diff:%d\n", sum);
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
	if (grayImg.data == NULL || subImg.data == NULL)
	{
		printf("image is NULL!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	if (width <= sub_width || height <= sub_height)
	{
		printf("subgraph is bigger than original picture!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}

	//计算大图的梯度和角度
	Mat gradImg_x(height, width, CV_32FC1);
	Mat gradImg_y(height, width, CV_32FC1);
	Mat angleImg(height, width, CV_32FC1);
	Mat magImg(height, width, CV_32FC1);

	int grad = ustc_CalcGrad(grayImg, gradImg_x, gradImg_y);
	if (grad == SUB_IMAGE_MATCH_FAIL)
	{
		printf("something is wrong with the gradImg!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}

	int angle = ustc_CalcAngleMag(gradImg_x, gradImg_y, angleImg, magImg);
	if (angle == SUB_IMAGE_MATCH_FAIL)
	{
		printf("something is wrong with the angleImg!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}

	//计算子图的梯度和角度
	Mat sub_gradImg_x(sub_height, sub_width, CV_32FC1);
	Mat sub_gradImg_y(sub_height, sub_width, CV_32FC1);
	Mat sub_angleImg(sub_height, sub_width, CV_32FC1);
	Mat sub_magImg(sub_height, sub_width, CV_32FC1);

	int sub_grad = ustc_CalcGrad(subImg, sub_gradImg_x, sub_gradImg_y);
	if (sub_grad == SUB_IMAGE_MATCH_FAIL)
	{
		printf("something is wrong with the sub_gradImg!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}
	int sub_angle = ustc_CalcAngleMag(sub_gradImg_x, sub_gradImg_y, sub_angleImg, sub_magImg);
	if (sub_angle == SUB_IMAGE_MATCH_FAIL)
	{
		printf("something is wrong with the sub_angleImg!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}

	/*
	int flag = ustc_SubImgMatch_gray(angleImg, sub_angleImg, x, y);
	if (flag == SUB_IMAGE_MATCH_FAIL)
	{
	printf("somgthing is wrong with the matching!\n");
	return SUB_IMAGE_MATCH_FAIL;
	}
	*/

	int sum = INT_MAX;

	for (int row_i = height - sub_height - 1; row_i >= 0; row_i--)
	{
		for (int col_j = width - sub_width - 1; col_j >= 0; col_j--)
		{
			int total_diff = 0;
			for (int sub_row_i = sub_height - 1; sub_row_i >= 0; sub_row_i--)
			{
				for (int sub_col_j = sub_width -0; sub_col_j >= 0; sub_col_j--)
				{
					int row_index = row_i + sub_row_i;
					int col_index = col_j + sub_col_j;
					int bigImg_pix = grayImg.data[row_index * width + col_index];
					int template_pix = subImg.data[sub_row_i * sub_width + sub_col_j];
					total_diff += abs(bigImg_pix - template_pix);
				}
			}
			//((float*)searchImg.data)[row_i * width + col_j] = total_diff;
			if (total_diff < sum)
			{
				*x = col_j + 1;
				*y = row_i + 1;
				sum = total_diff;
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
	if (grayImg.data == NULL || subImg.data == NULL)
	{
		printf("image is NULL!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	if (width <= sub_width || height <= sub_height)
	{
		printf("subgraph is bigger than original picture!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}

	int sum = INT_MAX;
	int* hist_temp = new int[255];
	memset(hist_temp, 0, sizeof(int)* 255);

	int* sub_hist = new int[255];
	memset(sub_hist, 0, sizeof(int) * 255);

	//统计子图直方图
	for (int row_i = sub_height - 1; row_i >= 0; row_i--)
	{
		for (int col_j = sub_width - 1; col_j >= 0; col_j--)
		{
			int pixVal = subImg.data[row_i * sub_width + col_j];
			sub_hist[pixVal]++;
		}
	}

	for (int row_i = height - sub_height - 1; row_i >= 0; row_i--)
	{
		for (int col_j = width - sub_width - 1; col_j >= 0; col_j--)
		{
			memset(hist_temp, 0, sizeof(int)* 255);

			for (int row_x = sub_height - 1; row_x >= 0; row_x--)
			{
				for (int col_y = sub_width - 1; col_y >= 0; col_y--)
				{
					int row_index = row_i + row_x;
					int col_index = col_j + col_y;
					int bigImg_pix = grayImg.data[row_index * width + col_index];
					hist_temp[bigImg_pix]++;
				}
			}

			int total_diff = 0;
			for (int count = 254; count >= 0; count--)
			{
				total_diff += abs(hist_temp[count] - sub_hist[count]);
			}

			if (total_diff < sum)
			{
				*x = col_j+1;
				*y = row_i+1;
				sum = total_diff;
			}
		}
	}

	delete[] hist_temp;
	delete[] sub_hist;

	return SUB_IMAGE_MATCH_OK;
}

//快速开方算法
float SqrtByRSQRTSS(float a)
{
	float b = a;
	__m128 in = _mm_load_ss(&b);
	__m128 out = _mm_rsqrt_ss(in);
	_mm_store_ss(&b, out);

	return a*b;
}
//快速开方求到数
float InvSqrt(float x)
{
	float xhalf = 0.5f * x;
	int i = *(int *)& x;
	i = 0x5f3759df - (i >> 1);
	x = *(float *)& i;
	x = x * (1.5f - xhalf * x * x);

	return x;
}
