#include "SubImageMatch.h"

//#define IMG_SHOW

//函数功能：将bgr图像转化成灰度图像
//bgrImg：彩色图，像素排列顺序是bgr
//grayImg：灰度图，单通道
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_ConvertBgr2Gray(Mat bgrImg, Mat& grayImg)
{
	if (NULL == bgrImg.data)
	{
		std::cout << "image is NULL." << std::endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = bgrImg.cols;
	int height = bgrImg.rows;
	int gray_width = grayImg.cols;
	int gray_height = grayImg.rows;

	if (width != gray_width || height != gray_height)
	{
		std::cout << "The size of two image is not same." << std::endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	for (int row_i = 0; row_i < height; row_i++)
	{
		int temp = row_i * width;
		for (int col_j = 0; col_j < width; col_j++)
		{
			int b = bgrImg.data[3 * (temp + col_j) + 0];
			int g = bgrImg.data[3 * (temp + col_j) + 1];
			int r = bgrImg.data[3 * (temp + col_j) + 2];

			grayImg.data[temp + col_j] = (b * 0.114f + g * 0.587f + r * 0.229f);
		}
	}

#ifdef IMG_SHOW
	namedWindow("grayImg", 0);
	imshow("grayImg", grayImg);
	waitKey();
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
	if (NULL == grayImg.data)
	{
		std::cout << "image is NULL." << std::endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int channel = grayImg.channels();

	//图像清零
	gradImg_x.setTo(0);
	gradImg_y.setTo(0);
	
	//计算梯度x,y方向同时算
	for (int row_i = 1; row_i < height - 1; row_i++)
	{
		for (int col_j = 1; col_j < width - 1; col_j++)
		{
			//x方向梯度计算
			int grad_x = 
				grayImg.data[channel * ((row_i - 1) * width + col_j + 1)]
				- grayImg.data[channel * ((row_i - 1) * width + col_j - 1)]
				- 2 * grayImg.data[channel * (row_i * width + col_j - 1)]
				+ 2 * grayImg.data[channel * (row_i * width + col_j + 1)]
				+ grayImg.data[channel * ((row_i + 1) * width + col_j + 1)]
				- grayImg.data[channel * ((row_i + 1) * width + col_j - 1)];

			((float*)gradImg_x.data)[row_i * width + col_j] = grad_x;

			//y方向梯度计算
			int grad_y = 
				grayImg.data[channel * ((row_i + 1) * width + col_j - 1)]
				- grayImg.data[channel * ((row_i - 1) * width + col_j - 1)]
				- 2 * grayImg.data[channel * ((row_i - 1) * width + col_j)]
				+ 2 * grayImg.data[channel * ((row_i + 1) * width + col_j)]
				+ grayImg.data[channel * ((row_i + 1) * width + col_j + 1)]
				- grayImg.data[channel * ((row_i - 1) * width + col_j + 1)];

			((float*)gradImg_y.data)[row_i * width + col_j] = grad_y;
		}
	}
#ifdef IMG_SHOW
	Mat gradImg_8U(height, width, CV_8UC1);
	//为了方便观察，直接取绝对值
	for (int row_i = 0; row_i < height; row_i++)
	{
		for (int col_j = 0; col_j < width; col_j++)
		{
			gradImg_8U.data[row_i * width + col_j] = (int)((abs( ((float*)gradImg_x.data)[row_i * width + col_j] ) 
				                                       + abs( ((float*)gradImg_y.data)[row_i * width + col_j] )) / 2);
		}
	}
	imshow("gradImg", gradImg_8U);
	waitKey();
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
	if (NULL == gradImg_x.data || NULL == gradImg_y.data)
	{
		std::cout << "image is NULL." << std::endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = gradImg_x.cols;
	int height = gradImg_x.rows;

	angleImg.setTo(0);
	magImg.setTo(0);

	//计算角度图、幅度图
	float grad_x;
	float grad_y;
	float mag, angle;

	int sqrt_i;
	float sqrt_x2, sqrt_y;

	float atan_ax, atan_ay;
	float atan_A, atan_S;

	for (int row_i = 1; row_i < height - 1; row_i++)
	{
		int temp = row_i * width;
		for (int col_j = 1; col_j < width - 1; col_j++)
		{
			//计算角度
			grad_x = ((float*)gradImg_x.data)[temp + col_j];
			grad_y = ((float*)gradImg_y.data)[temp + col_j];

			//angle = atan2(grad_y, grad_x);
			
			//计算角度，反正切运算
			atan_ax = grad_x<0 ? (-1 * grad_x) : grad_x;
			atan_ay = grad_y<0 ? (-1 * grad_y) : grad_y;
            atan_A = (atan_ax>atan_ay ? atan_ay : atan_ax) 
				     / (atan_ax<atan_ay ? atan_ay : atan_ax);
            atan_S = atan_A * atan_A;
            angle = ((-0.0464964749f * atan_S + 0.15931422f) * atan_S - 0.327622764f) * atan_S * atan_A + atan_A;
            angle = ((atan_ay - atan_ax > 0) ? (1.57079637f - angle) : angle);
            angle = (grad_x < 0 ? (3.14159274f - angle) : angle);
			angle = (grad_y<0 ? (-1 * angle) : angle);
			angle = (atan_ax < 0.001f && atan_ax < 0.001f) ? 0 : angle;

			angle *= 180 / CV_PI;  
			angle += 180;

			((float*)angleImg.data)[temp + col_j] = angle;

			//计算幅值
			//开根号运算，输入mag_2，输出mag
			sqrt_y = (grad_x * grad_x + grad_y * grad_y);
			sqrt_x2 = sqrt_y * 0.5F;
			sqrt_i = *(int *) & sqrt_y;
			sqrt_i = 0x5f3759df - (sqrt_i >> 1);
			sqrt_y = *(float *) & sqrt_i;
			sqrt_y *= (1.5f - (sqrt_x2 * sqrt_y * sqrt_y));
			mag = 1.0 / sqrt_y;                                            
     
            ((float*)magImg.data)[temp + col_j] = mag;
		}
	}

#ifdef IMG_SHOW
	Mat angleImg_8U(height, width, CV_8UC1);
	Mat magImg_8U(height, width, CV_8UC1);
	//为了方便观察，进行些许变化
	for (int row_i = 0; row_i < height; row_i++)
	{
		for (int col_j = 0; col_j < width; col_j++)
		{
			float angle = ((float*)angleImg.data)[row_i * width + col_j];
			//为了能在8U上显示，缩小到0-180之间
			angle /= 2;
			angleImg_8U.data[row_i * width + col_j] = angle;

			float mag = ((float*)magImg.data)[row_i * width + col_j];
			mag = (mag > 255 ? 255 : mag);
			magImg_8U.data[row_i * width + col_j] = mag;
		}
	}

    //显示
	imshow("angleImg_8U", angleImg_8U);
	imshow("magImg_8U", magImg_8U);
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
	if (NULL == grayImg.data)
	{
		std::cout << "image is NULL." << std::endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int channel = grayImg.channels();
	int binary_width = binaryImg.cols;
	int binary_height = binaryImg.rows;

	if (width != binary_width || height != binary_height)
	{
		std::cout << "The size of two image is not same." << std::endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int temp0, temp1;
	int pixVal, dstVal;

	for (int row_i = 0; row_i < height; row_i++)
	{
		temp0 = row_i * width;
		for (int col_j = 0; col_j < width; col_j++)
		{
			temp1 = temp0 + col_j;
			pixVal = grayImg.data[temp1 * channel];
			dstVal = (((pixVal - th) >> 31) + 1) * 255;
			binaryImg.data[temp1] = dstVal;
		}
	}

#ifdef IMG_SHOW
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
	if (NULL == grayImg.data)
	{
		std::cout << "image is NULL." << std::endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (hist == NULL || hist_len == 0)
	{
		std::cout << "hist is NULL." << std::endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int channel = grayImg.channels();

	//直方图清零
	for (int i = 0; i < hist_len; i++)
	{
		hist[i] = 0;
	}

	//计算直方图
	for (int row_i = 0; row_i < height; row_i++)
	{
		int temp = row_i * width;
		for (int col_j = 0; col_j < width; col_j++)
		{
			int pixVal = grayImg.data[(temp + col_j) * channel];
			*(hist + pixVal) += 1;
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
		std::cout << "image is NULL." << std::endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int ch = grayImg.channels();
	int subch = subImg.channels();

	if (ch != subch)
	{
		std::cout << "channel mismatch." << std::endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
    
	if (sub_height > height || sub_width > width)
	{
		std::cout << "SubImg is laager than bigImg." << std::endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	//最小误差
	int min = INT32_MAX;
	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int row_i = 0; row_i < height - sub_height; row_i++)
	{
		for (int col_j = 0; col_j < width - sub_width; col_j++)
		{
			int total_diff = 0;
			//遍历模板图上的每一个像素
			for (int row_y = 0; row_y < sub_height; row_y++)
			{
				for (int col_x = 0; col_x < sub_width; col_x++)
				{
					//大图上的像素位置
					int row_index = row_i + row_y;
					int col_index = col_j + col_x;
					int bigImg_pix = grayImg.data[(row_index * width + col_index) * ch];
					//模板图上的像素
					int template_pix = subImg.data[(row_y * sub_width + col_x) * ch];
					int sub = bigImg_pix - template_pix;
					total_diff += (sub - 2 * sub * ((sub >> 31) & 0x00000001));
				}
			}
			//找最小误差坐标
			if (total_diff < min)
			{
				min = total_diff;

				*y = row_i;
				*x = col_j;
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
		std::cout << "image is NULL." << std::endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int ch = colorImg.channels();
	int subch = subImg.channels();

	if (ch != subch || ch != 3)
	{
		std::cout << "channel mismatch." << std::endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = colorImg.cols;
	int height = colorImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	if (sub_height > height || sub_width > width)
	{
		std::cout << "SubImg is laager than bigImg." << std::endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	//最小误差
	int min = INT32_MAX;

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int row_i = 0; row_i < height - sub_height; row_i++)
	{
		for (int col_j = 0; col_j < width - sub_width; col_j++)
		{
			int total_diff = 0;
			//遍历模板图上的每一个像素
			for (int row_y = 0; row_y < sub_height; row_y++)
			{
				for (int col_x = 0; col_x < sub_width; col_x++)
				{
					//大图上的像素位置
					int row_index = row_i + row_y;
					int col_index = col_j + col_x;
					int bigImg_pix_r = colorImg.data[(row_index * width + col_index) * ch + 2];
					int bigImg_pix_g = colorImg.data[(row_index * width + col_index) * ch + 1];
					int bigImg_pix_b = colorImg.data[(row_index * width + col_index) * ch + 0];
					//模板图上的像素
					int template_pix_r = subImg.data[(row_y * sub_width + col_x) * ch + 2];
					int template_pix_g = subImg.data[(row_y * sub_width + col_x) * ch + 1];
					int template_pix_b = subImg.data[(row_y * sub_width + col_x) * ch + 0];

					int sub_b = bigImg_pix_b - template_pix_b;
					int sub_g = bigImg_pix_g - template_pix_g;
					int sub_r = bigImg_pix_r - template_pix_r;
					
					sub_b = (sub_b - 2 * sub_b * ((sub_b >> 31) & 0x00000001));
					sub_g = (sub_g - 2 * sub_g * ((sub_g >> 31) & 0x00000001));
					sub_r = (sub_r - 2 * sub_r * ((sub_r >> 31) & 0x00000001));

					total_diff += (sub_b + sub_g + sub_r);
				}
			}
			//找最小误差坐标
			if (total_diff < min)
			{
				min = total_diff;

				*y = row_i;
				*x = col_j;
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
		std::cout << "image is NULL." << std::endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int ch = grayImg.channels();
	int subch = subImg.channels();

	if (ch != subch)
	{
		std::cout << "channel mismatch." << std::endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	if (sub_height > height || sub_width > width)
	{
		std::cout << "SubImg is laager than bigImg." << std::endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	//最大相关值
	float R_max = 0;

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int row_i = 0; row_i < height - sub_height; row_i++)
	{
		for (int col_j = 0; col_j < width - sub_width; col_j++)
		{
			float sum_ST = 0;
			float sum_SS = 0;
			float sum_TT = 0;
			float R = 0;
			//遍历模板图上的每一个像素
			for (int row_y = 0; row_y < sub_height; row_y++)
			{
				for (int col_x = 0; col_x < sub_width; col_x++)
				{
					//大图上的像素位置
					int row_index = row_i + row_y;
					int col_index = col_j + col_x;
					int bigImg_pix = grayImg.data[(row_index * width + col_index) * ch];
					//模板图上的像素
					int template_pix = subImg.data[(row_y * sub_width + col_x) * ch];

					sum_ST += bigImg_pix * template_pix;
					sum_SS += template_pix * template_pix;
					sum_TT += bigImg_pix * bigImg_pix;
				}
			}
			R = (float)sum_ST / sqrt(sum_SS * sum_TT);
			if (R > R_max)
			{
				R_max = R;
				*x = col_j;
				*y = row_i;
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
		std::cout << "image is NULL." << std::endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int ch = grayImg.channels();
	int subch = subImg.channels();

	if (ch != subch)
	{
		std::cout << "channel mismatch." << std::endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	if (sub_height > height || sub_width > width)
	{
		std::cout << "SubImg is laager than bigImg." << std::endl;
		return SUB_IMAGE_MATCH_FAIL;
	}


	float atan_ax, atan_ay, atan_A, atan_S;
	float angle, subangle;
	//计算子图角度
	int subL = sub_width * sub_height;
	int* SubAgl = new int[subL];
	memset(SubAgl, 0, sizeof(int) * subL);
	for (int row_y = 1; row_y < sub_height - 1; row_y++)
	{
		for (int col_x = 1; col_x < sub_width - 1; col_x++)
		{
			int subgrad_x =
				subImg.data[ch * ((row_y - 1) * sub_width + col_x + 1)]
				- subImg.data[ch * ((row_y - 1) * sub_width + col_x - 1)]
				- 2 * subImg.data[ch * (row_y * sub_width + col_x - 1)]
				+ 2 * subImg.data[ch * (row_y * sub_width + col_x + 1)]
				+ subImg.data[ch * ((row_y + 1) * sub_width + col_x + 1)]
				- subImg.data[ch * ((row_y + 1) * sub_width + col_x - 1)];
			int subgrad_y =
				subImg.data[ch * ((row_y + 1) * sub_width + col_x - 1)]
				- subImg.data[ch * ((row_y - 1) * sub_width + col_x - 1)]
				- 2 * subImg.data[ch * ((row_y - 1) * sub_width + col_x)]
				+ 2 * subImg.data[ch * ((row_y + 1) * sub_width + col_x)]
				+ subImg.data[ch * ((row_y + 1) * sub_width + col_x + 1)]
				- subImg.data[ch * ((row_y - 1) * sub_width + col_x + 1)];

            //计算小图角度
			atan_ax = subgrad_x<0 ? (-1 * subgrad_x) : subgrad_x;
			atan_ay = subgrad_y<0 ? (-1 * subgrad_y) : subgrad_y;
			atan_A = (atan_ax>atan_ay ? atan_ay : atan_ax)
				/ (atan_ax<atan_ay ? atan_ay : atan_ax);
			atan_S = atan_A * atan_A;
			subangle = ((-0.0464964749f * atan_S + 0.15931422f) * atan_S - 0.327622764f) * atan_S * atan_A + atan_A;
			subangle = ((atan_ay - atan_ax > 0) ? (1.57079637f - subangle) : subangle);
			subangle = (subgrad_x < 0 ? (3.14159274f - subangle) : subangle);
			subangle = (subgrad_y<0 ? (-1 * subangle) : subangle);
			subangle = (atan_ax < 0.001f && atan_ax < 0.001f) ? 0 : subangle;
			subangle *= 180 / CV_PI;
			subangle += 180;

			SubAgl[row_y * sub_width + col_x] = subangle;
		}
	}

	//计算大图幅度
	int L = width * height;
	int* Agl = new int[L];
	memset(Agl, 0, sizeof(int) * L);
	for (int row_i = 1; row_i < height - 1; row_i++)
	{
		for (int col_j = 1; col_j < width - 1; col_j++)
		{
			int grad_x =
				grayImg.data[ch * ((row_i - 1) * width + col_j + 1)]
				- grayImg.data[ch * ((row_i - 1) * width + col_j - 1)]
				- 2 * grayImg.data[ch * (row_i * width + col_j - 1)]
				+ 2 * grayImg.data[ch * (row_i * width + col_j + 1)]
				+ grayImg.data[ch * ((row_i + 1) * width + col_j + 1)]
				- grayImg.data[ch * ((row_i + 1) * width + col_j - 1)];
			int grad_y =
				grayImg.data[ch * ((row_i + 1) * width + col_j - 1)]
				- grayImg.data[ch * ((row_i - 1) * width + col_j - 1)]
				- 2 * grayImg.data[ch * ((row_i - 1) * width + col_j)]
				+ 2 * grayImg.data[ch * ((row_i + 1) * width + col_j)]
				+ grayImg.data[ch * ((row_i + 1) * width + col_j + 1)]
				- grayImg.data[ch * ((row_i - 1) * width + col_j + 1)];

			//计算大图角度
			atan_ax = grad_x<0 ? (-1 * grad_x) : grad_x;
			atan_ay = grad_y<0 ? (-1 * grad_y) : grad_y;
			atan_A = (atan_ax>atan_ay ? atan_ay : atan_ax)
				/ (atan_ax<atan_ay ? atan_ay : atan_ax);
			atan_S = atan_A * atan_A;
			angle = ((-0.0464964749f * atan_S + 0.15931422f) * atan_S - 0.327622764f) * atan_S * atan_A + atan_A;
			angle = ((atan_ay - atan_ax > 0) ? (1.57079637f - angle) : angle);
			angle = (grad_x < 0 ? (3.14159274f - angle) : angle);
			angle = (grad_y<0 ? (-1 * angle) : angle);
			angle = (atan_ax < 0.001f && atan_ax < 0.001f) ? 0 : angle;

			angle *= 180 / CV_PI;
			angle += 180;

			Agl[row_i * width + col_j] = angle;
		}
	}

	//最小角度差
	int aglSub_min = INT32_MAX;

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int row_i = 0; row_i < height - sub_height; row_i++)
	{
		for (int col_j = 0; col_j < width - sub_width; col_j++)
		{
			float atan_ax, atan_ay, atan_A, atan_S;
			float angle, subangle;
			int aglSub = 0;
			//遍历模板图上的每一个像素
			for (int row_y = 1; row_y < sub_height - 1; row_y++)
			{
				for (int col_x = 1; col_x < sub_width - 1; col_x++)
				{
					//大图上的像素位置
					int row_index = row_i + row_y;
					int col_index = col_j + col_x;

					angle = Agl[row_index * width + col_index];
					subangle = SubAgl[row_y * sub_width + col_x];

					//求差
					int Sub = abs(angle - subangle);
					aglSub += ((Sub > 180) ? (360 - Sub) : Sub);
				}
			}
			int temp0 = aglSub - aglSub_min;
			int temp1 = (temp0 >> 31) & 0x00000001;
			aglSub_min = temp0 * temp1 + aglSub_min;
			*x = (col_j - *x) * temp1 + *x;
			*y = (row_i - *y) * temp1 + *y;
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
		std::cout << "image is NULL." << std::endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int ch = grayImg.channels();
	int subch = subImg.channels();

	if (ch != subch)
	{
		std::cout << "channel mismatch." << std::endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	if (sub_height > height || sub_width > width)
	{
		std::cout << "SubImg is laager than bigImg." << std::endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	//计算子图幅度
	int subL = sub_width * sub_height;
	int* SubMag = new int[subL];
	memset(SubMag, 0, sizeof(int) * subL);
	for (int row_y = 1; row_y < sub_height - 1; row_y++)
	{
		for (int col_x = 1; col_x < sub_width - 1; col_x++)
		{
			int subgrad_x =
				subImg.data[ch * ((row_y - 1) * sub_width + col_x + 1)]
				- subImg.data[ch * ((row_y - 1) * sub_width + col_x - 1)]
				- 2 * subImg.data[ch * (row_y * sub_width + col_x - 1)]
				+ 2 * subImg.data[ch * (row_y * sub_width + col_x + 1)]
				+ subImg.data[ch * ((row_y + 1) * sub_width + col_x + 1)]
				- subImg.data[ch * ((row_y + 1) * sub_width + col_x - 1)];
			int subgrad_y =
				subImg.data[ch * ((row_y + 1) * sub_width + col_x - 1)]
				- subImg.data[ch * ((row_y - 1) * sub_width + col_x - 1)]
				- 2 * subImg.data[ch * ((row_y - 1) * sub_width + col_x)]
				+ 2 * subImg.data[ch * ((row_y + 1) * sub_width + col_x)]
				+ subImg.data[ch * ((row_y + 1) * sub_width + col_x + 1)]
				- subImg.data[ch * ((row_y - 1) * sub_width + col_x + 1)];
			
			SubMag[row_y * sub_width + col_x] = sqrt(subgrad_x * subgrad_x + subgrad_y * subgrad_y);
		}
	}

	//计算大图幅度
	int L = width * height;
	int* Mag = new int[L];
	memset(Mag, 0, sizeof(int) * L);
	for (int row_i = 1; row_i < height - 1; row_i++)
	{
		for (int col_j = 1; col_j < width - 1; col_j++)
		{
			int grad_x =
				grayImg.data[ch * ((row_i - 1) * width + col_j + 1)]
				- grayImg.data[ch * ((row_i - 1) * width + col_j - 1)]
				- 2 * grayImg.data[ch * (row_i * width + col_j - 1)]
				+ 2 * grayImg.data[ch * (row_i * width + col_j + 1)]
				+ grayImg.data[ch * ((row_i + 1) * width + col_j + 1)]
				- grayImg.data[ch * ((row_i + 1) * width + col_j - 1)];
			int grad_y =
				grayImg.data[ch * ((row_i + 1) * width + col_j - 1)]
				- grayImg.data[ch * ((row_i - 1) * width + col_j - 1)]
				- 2 * grayImg.data[ch * ((row_i - 1) * width + col_j)]
				+ 2 * grayImg.data[ch * ((row_i + 1) * width + col_j)]
				+ grayImg.data[ch * ((row_i + 1) * width + col_j + 1)]
				- grayImg.data[ch * ((row_i - 1) * width + col_j + 1)];

			Mag[row_i * width + col_j] = sqrt(grad_x * grad_x + grad_y * grad_y);
		}
	}

	//最小幅度差
	int magSub_min = INT32_MAX;

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int row_i = 0; row_i < height - sub_height; row_i++)
	{
		for (int col_j = 0; col_j < width - sub_width; col_j++)
		{
			int mag, submag;
			int magSub = 0;
			//遍历模板图上的每一个像素
			for (int row_y = 1; row_y < sub_height - 1; row_y++)
			{
				for (int col_x = 1; col_x < sub_width - 1; col_x++)
				{
					//大图上的像素位置
					int row_index = row_i + row_y;
					int col_index = col_j + col_x;

					mag = Mag[row_index * width + col_index];
					submag = SubMag[row_y * sub_width + col_x];

					magSub += abs(mag - submag);
				}
			}
		
			//记录幅度差最小处的坐标
			int temp0 = magSub - magSub_min;
			int temp1 = (temp0 >> 31) & 0x00000001;
			magSub_min = temp0 * temp1 + magSub_min;
			*x = (col_j - *x) * temp1 + *x;
			*y = (row_i - *y) * temp1 + *y;
			
		}
	}

	delete[] SubMag;
	delete[] Mag;

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
		std::cout << "image is NULL." << std::endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int ch = grayImg.channels();
	int subch = subImg.channels();

	if (ch != subch)
	{
		std::cout << "channel mismatch." << std::endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	if (sub_height > height || sub_width > width)
	{
		std::cout << "SubImg is laager than bigImg." << std::endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg(height, width, CV_32FC1);
	//匹配误差初始化
	searchImg.setTo(FLT_MAX);

	//定义大图的临时直方图
	int* hist_temp = new int[256];
	memset(hist_temp, 0, sizeof(int) * 256);
	//计算子图直方图
	int* sub_hist = new int[256];
	memset(sub_hist, 0, sizeof(int) * 256);
	int t = ustc_CalcHist(subImg,sub_hist, 256);
	
	if (t == SUB_IMAGE_MATCH_FAIL)
	{
		std::cout << "Subfunction Run Failed !" << std::endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int diff_min = INT32_MAX;

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int row_i = 0; row_i < height - sub_height; row_i++)
	{
		for (int col_j = 0; col_j < width - sub_width; col_j++)
		{
			//清零
			memset(hist_temp, 0, sizeof(int) * 256);
			int total_diff = 0;

			//遍历模板图上的每一个像素
			for (int row_y = 0; row_y < sub_height; row_y++)
			{
				for (int col_x = 0; col_x < sub_width; col_x++)
				{
					//大图上的像素位置
					int row_index = row_i + row_y;
					int col_index = col_j + col_x;
					int bigImg_pix = grayImg.data[(row_index * width + col_index) * ch];
					
					hist_temp[bigImg_pix]++;
				}
			}

			//根据直方图计算匹配误差
			for (int ii = 0; ii < 256; ii++)
			{
				int sub = hist_temp[ii] - sub_hist[ii];
				total_diff += (sub - 2 * sub * ((sub >> 31) & 0x00000001));
			}

			//记录最小误差坐标
			int temp0 = total_diff -diff_min;
			int temp1 = (temp0 >> 31) & 0x00000001;
			diff_min = temp0 * temp1 + diff_min;
			*x = (col_j - *x) * temp1 + *x;
			*y = (row_i - *y) * temp1 + *y;
			
		}
	}

	delete[] hist_temp;
	delete[] sub_hist;

	return SUB_IMAGE_MATCH_OK;
}
