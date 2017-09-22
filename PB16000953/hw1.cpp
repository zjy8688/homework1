#include"SubImageMatch.h"
int ustc_ConvertBgr2Gray(Mat bgrImg, Mat& grayImg) 
{
	//函数实现
	if (NULL == bgrImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (grayImg.channels() != 1||bgrImg.channels()!=3)
	{
		cout << "channels ERROR" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = bgrImg.cols;
	int height = bgrImg.rows;
	for (int row_i = 0; row_i < height; row_i++)
	{
		int t = row_i*width;
		for (int col_j = 0; col_j < width; col_j += 1)
		{
			int temp = t + col_j;
			int temp1 = temp + temp + temp;
			int b = bgrImg.data[temp1];
			int g = bgrImg.data[temp1 + 1];
			int r = bgrImg.data[temp1 + 2];
			int grayVal = (b * 7472 + g * 38469+ r * 19595)>>16;
			grayImg.data[t+col_j] = grayVal;
		}
	}
#ifdef IMG_SHOW
	namedWindow("grayImg", 0);
	imshow("grayImg", grayImg);
	waitKey();
#endif
	return(SUB_IMAGE_MATCH_OK);
}
int ustc_CalcGrad(Mat grayImg,Mat& gradImg_x, Mat&gradImg_y)
{
	//函数实现
	if (NULL == grayImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (grayImg.channels() != 1)
	{
		cout << "channels ERROR" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	gradImg_x.setTo(0);
	gradImg_y.setTo(0);
	for (int row_i = 1; row_i < height - 1; row_i++)
	{
		int t = row_i*width;
		int t1 = (row_i-1)*width;
		int t2 = (row_i+1)*width;
		for (int col_j = 1; col_j < width - 1; col_j++)
		{
			int temp0 = t + col_j ;
			int temp1 = t1 + col_j;
			int temp2 = t2 +col_j;
			int grad_x =
				grayImg.data[temp1 + 1]
				+ grayImg.data[temp0 + 1]*2
				+ grayImg.data[temp2 + 1]
				- grayImg.data[temp1 - 1]
				- grayImg.data[temp0 - 1]*2
				- grayImg.data[temp2 - 1];

			((float*)gradImg_x.data)[temp0] = grad_x;
			int grad_y =
				grayImg.data[temp2 - 1]
				+ grayImg.data[temp2]*2
				+ grayImg.data[temp2 + 1]
				- grayImg.data[temp1 - 1]
				- grayImg.data[temp1]*2
				- grayImg.data[temp1 + 1];
			((float*)gradImg_y.data)[temp0] = grad_y;
		}
	}
#ifdef IMG_SHOW
	Mat gradImg_x_8U(height, width, CV_8UC1);
	for (int row_i = 0; row_i < height; row_i++)
	{
		for (int col_j = 0; col_j < width; col_j++)
		{
			int val_x = ((float*)gradImg_x.data)[row_i * width + col_j];
			gradImg_x_8U.data[row_i * width + col_j] = abs(val_x);
		}
	}
	namedWindow("gradImg_x_8U", 0);
	imshow("gradImg_x_8U", gradImg_x_8U);
	Mat gradImg_y_8U(height, width, CV_8UC1);
	for (int row_i = 0; row_i < height; row_i++)
	{
		for (int col_j = 0; col_j < width; col_j++)
		{
			int val_y = ((float*)gradImg_y.data)[row_i * width + col_j];
			gradImg_y_8U.data[row_i * width + col_j] = abs(val_y);
		}
	}
	namedWindow("gradImg_y_8U", 0);
	imshow("gradImg_y_8U", gradImg_y_8U);
	waitKey();
#endif
	return SUB_IMAGE_MATCH_OK;
}
int ustc_CalcAngleMag( Mat gradImg_x, Mat gradImg_y, Mat&angleImg, Mat& magImg)
{
	//函数实现
	if (NULL == gradImg_x.data || NULL == gradImg_y.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = gradImg_x.cols;
	int height = gradImg_x.rows;
	angleImg.setTo(0);
	magImg.setTo(0);
	for (int row_i = 1; row_i < height-1; row_i++)
	{
		int temp0 = row_i*width;
		for (int col_j = 1; col_j < width-1; col_j++)
		{
			int temp1 = temp0 + col_j;
			float grad_x = ((float*)gradImg_x.data)[temp1];
			float grad_y = ((float*)gradImg_y.data)[temp1];
			float temp= grad_x*grad_x+grad_y*grad_y;
#define SQRT_MAGIC_F 0x5f3759df 
			//计算开方
			float xhalf = 0.5f*temp;
			union 
			{
				float x;
				int i;
			} u;
			u.x = temp;
			u.i = SQRT_MAGIC_F - (u.i >> 1);
			float mag = temp*u.x*(1.5f - xhalf*u.x*u.x);

			float t = atan2(grad_y, grad_x);
			float angle = t/CV_PI*180+180;
			//自己找办法优化三角函数速度，并且转化为角度制，规范化到0-360
			((float*)angleImg.data)[row_i * width + col_j] = angle;
			((float*)magImg.data)[row_i * width + col_j] = mag;
		}
	}
#ifdef IMG_SHOW
	Mat angleImg_8U(height, width, CV_8UC1);
	//为了方便观察，进行些许变化
	for (int row_i = 0; row_i < height; row_i++)
	{
		for (int col_j = 0; col_j < width; col_j++)
		{
			float angle = ((float*)angleImg.data)[row_i * width + col_j];
			//为了能在8U上显示，缩小到0-180之间
			angle /= 2;
			angleImg_8U.data[row_i * width + col_j] = angle;
		}
	}
	namedWindow("angleImg_x_8U", 0);
	imshow("angleImg_x_8U", angleImg_8U);
	namedWindow("magImg", 0);
	imshow("magImg", magImg);
	waitKey();
#endif
	return SUB_IMAGE_MATCH_OK;
}
int ustc_Threshold(Mat grayImg, Mat& binaryImg, int th)
{
	//函数实现
	if (NULL == grayImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	//检查通道数是否匹配
	if (grayImg.channels() != 1 || 1 != binaryImg.channels())
	{
		cout << "channels ERROR" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	//检测阈值
	if (th < 0 || th>255)
	{
		cout << "th ERROR" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	int val;
	for (int row_i = 0; row_i < height; row_i++)
	{
		int t = row_i*width;
		for (int col_j = 0; col_j < width; col_j++)
		{
			int temp = t + col_j;
			int pix = grayImg.data[temp];
			if (pix<th)
				val = 0;
			else val = 255;
			binaryImg.data[temp] = val;
		}
	}
#ifdef IMG_SHOW
namedWindow("binaryImg", 0);
	imshow("binaryImg", binaryImg);
	waitKey();
#endif
	return SUB_IMAGE_MATCH_OK;
}
int ustc_CalcHist(Mat grayImg, int* hist, int hist_len)
{
	//函数实现
	if (NULL == grayImg.data || NULL == hist)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (grayImg.channels() != 1 )
	{
		cout << "channels ERROR" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	//检查hist_len
	int width = grayImg.cols;
	int height = grayImg.rows;
	//直方图清零
	for (int i = 0; i < hist_len; i++)
	{
		hist[i] = 0;
	}
	//计算直方图
	for (int row_i = 0; row_i < height; row_i++)
	{
		int temp = row_i*width;
		for (int col_j = 0; col_j < width; col_j++)
		{
			int pixVal = grayImg.data[temp+col_j];
			if (pixVal > hist_len-1||pixVal<0) 
			{
				cout << "ERROR:histlen is too short or pix<0 " << endl;
				return SUB_IMAGE_MATCH_FAIL;
			}

			hist[pixVal]++;
		}
	}
	return SUB_IMAGE_MATCH_OK;
}
int ustc_SubImgMatch_gray(Mat grayImg, Mat subImg, int* x, int* y)
{
	//函数实现
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	float minsearch;
	int temp_x=0,temp_y=0;
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	//检测图像大小是否符合要求
	if (width < sub_width || height < sub_height)
	{
		cout << "width or height ERROR " << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	//检查通道数是否匹配
	if (grayImg.channels() !=1||1!= subImg.channels())
	{
		cout << "channels ERROR" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg(height, width, CV_32FC1);
	//匹配误差初始化
	searchImg.setTo(FLT_MAX);
	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int row_i = 0; row_i < height - sub_height; row_i++)
	{
		for (int col_j = 0; col_j < width - sub_width; col_j++)
		{
			int total_diff = 0;
			//遍历模板图上的每一个像素
			for (int i = 0; i < sub_height; i++)
			{
				int row_index = row_i + i;
				int temp1 = row_index * width;
				int temp2 = i * sub_width;

				for (int j = 0; j < sub_width; j++)
				{
					//大图上的像素位置
					int col_index = col_j + j;
					int bigImg_pix = grayImg.data[ temp1+ col_index];
					//模板图上的像素
					int template_pix = subImg.data[temp2 + j];
					int n = bigImg_pix - template_pix;
					total_diff += n*((n >> 31) - (~n >> 31));
				}
			}
			//存储当前像素位置的匹配误差
			((float*)searchImg.data)[row_i * width + col_j] = total_diff;
			if (0 == row_i && 0 == col_j)
				minsearch = total_diff;
			if (minsearch > total_diff)
			{
				minsearch = total_diff;
				temp_y = row_i;
				temp_x = col_j;
			}
		}
	}
	*x = temp_x;
	*y = temp_y;
	return SUB_IMAGE_MATCH_OK;
}
int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int* x, int* y)
{
	//函数实现
	if (NULL == colorImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	//检查通道数是否匹配
	if (colorImg.channels()!=3||3 != subImg.channels())
	{
		cout << "channels ERROR" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	float minsearch;
	int temp_x = 0, temp_y = 0;
	int width = colorImg.cols;
	int height = colorImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	//检测图像大小是否符合要求
	if (width < sub_width || height < sub_height)
	{
		cout << "width or height ERROR " << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg(height, width, CV_32FC1);
	//匹配误差初始化
	searchImg.setTo(FLT_MAX);
	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int row_i = 0; row_i < height - sub_height; row_i++)
	{
		for (int col_j = 0; col_j < width - sub_width; col_j++)
		{
			int total_diff = 0;
			//遍历模板图上的每一个像素
			for (int i = 0; i < sub_height; i++)
			{
				int row_index = row_i + i;
				int temp1 = row_index * width*3;
				int temp2 = i * sub_width*3;

				for (int j = 0; j < sub_width; j++)
				{
					//大图上的像素位置
					int col_index = col_j + j;
					int t1 = col_index + col_index + col_index+temp1;
					int bigImg_b = colorImg.data[t1];
					int bigImg_g = colorImg.data[t1+1];
					int bigImg_r = colorImg.data[t1+2];
					//模板图上的像素
					int t2 = temp2 + j + j + j;
					int template_b = subImg.data[t2];
					int template_g = subImg.data[t2+1];
					int template_r = subImg.data[t2+2];
					int n1 = bigImg_b - template_b;
					int n2 = bigImg_g - template_g;
					int n3 = bigImg_r - template_r;
					int n = n1*((n1 >> 31) - (~n1 >> 31)) + n2*((n2 >> 31) - (~n2 >> 31)) + n3*((n3 >> 31) - (~n3 >> 31));
					total_diff += n;
				}
			}
			//存储当前像素位置的匹配误差
			((float*)searchImg.data)[row_i * width + col_j] = total_diff;
			if (0 == row_i && 0 == col_j)
				minsearch = total_diff;
			if (minsearch > total_diff)
			{
				minsearch = total_diff;
				temp_y = row_i;
				temp_x = col_j;
			}
		}
	}
	*x = temp_x;
	*y = temp_y;
	return SUB_IMAGE_MATCH_OK;
}
int ustc_SubImgMatch_corr(Mat grayImg, Mat subImg, int* x, int* y) {
	//函数实现
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	//检查通道数是否匹配
	if (grayImg.channels() !=1||1!= subImg.channels())
	{
		cout << "channels ERROR" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int temp_x = 0, temp_y = 0;
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	//检测图像大小是否符合要求
	if (width < sub_width || height < sub_height)
	{
		cout << "width or height ERROR " << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg(height, width, CV_32FC1);
	//匹配误差初始化
	searchImg.setTo(FLT_MAX);
	float minsearch = FLT_MAX;
	for (int row_i = 0; row_i <height - sub_height; row_i++)
	{
		for (int col_j = 0; col_j< width - sub_width; col_j++)
		{
			float t0 = 0;
			float t1 = 0;
			float t2 = 0;
			for (int i = 0; i < sub_height; i++)
			{
				int row_index = row_i + i;
				int temp0 = row_index*width;
				int temp1 = i*sub_width;

				for (int j = 0; j < sub_width; j++)
				{
					int col_index = j + col_j;
					int bigImg_pix=grayImg.data[temp0 + col_index];
					int template_pix=subImg.data[temp1 + j];
						t0 += bigImg_pix*bigImg_pix;
						t1 += template_pix*template_pix;
						t2 += bigImg_pix*template_pix;
				}
			}
			int sqrt0 = sqrt(t0);
			int sqrt1 = sqrt(t1);
			float r = t2 / (sqrt0*sqrt1);
			if (abs(minsearch-1) > abs(r-1))
			{
				minsearch = r;
				temp_x = col_j;
				temp_y = row_i;
			}

		}
	}
	*x = temp_x;
	*y = temp_y;
	return SUB_IMAGE_MATCH_OK;
}
int ustc_SubImgMatch_angle(Mat grayImg, Mat subImg, int* x, int* y)
{
	//函数实现
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	//检查通道数是否匹配
	if (grayImg.channels() !=1||1!= subImg.channels())
	{
		cout << "channels ERROR" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	//检测图像大小是否符合要求

	if (width < sub_width || height < sub_height)
	{
		cout << "width or height ERROR " << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	Mat angleImg(height, width, CV_32FC1);
	Mat gradImg_x(height, width, CV_32FC1);
	Mat gradImg_y(height, width, CV_32FC1);
	ustc_CalcGrad(grayImg,gradImg_x,gradImg_y);

	Mat angleImg_sub(sub_height, sub_width, CV_32FC1);
	Mat gradImg_subx(sub_height, sub_width, CV_32FC1);
	Mat gradImg_suby(sub_height, sub_width, CV_32FC1);
	ustc_CalcGrad(subImg, gradImg_subx, gradImg_suby);
	angleImg.setTo(0);
	angleImg_sub.setTo(0);
	for (int row_i = 1; row_i < height - 1; row_i++)
	{
		for (int col_j = 1; col_j < width - 1; col_j++)
		{
			float grad_x = ((float*)gradImg_x.data)[row_i * width + col_j];
			float grad_y = ((float*)gradImg_y.data)[row_i * width + col_j];
			float angle = atan2(grad_y, grad_x);
			((float*)angleImg.data)[row_i * width + col_j] = angle;
		}
	}
	for (int row_i = 1; row_i < sub_height - 1; row_i++)
	{
		for (int col_j = 1; col_j < sub_width - 1; col_j++)
		{
			float grad_subx = ((float*)gradImg_subx.data)[row_i * sub_width + col_j];
			float grad_suby = ((float*)gradImg_suby.data)[row_i * sub_width + col_j];
			float angle_sub = atan2(grad_suby, grad_subx);
			((float*)angleImg_sub.data)[row_i * sub_width+ col_j] = angle_sub;
		}
	}
	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg(height, width, CV_32FC1);
	//匹配误差初始化
	searchImg.setTo(FLT_MAX);
	//遍历大图每一个像素，注意行列的起始、终止坐标
	int temp_x=0,temp_y=0;
	float minsearch = FLT_MAX;
	for (int row_i = 0; row_i < height - sub_height; row_i++)
	{
		for (int col_j = 0; col_j < width - sub_width; col_j++)
		{
			int total_diff = 0;

			for (int i = 0; i < sub_height; i++)
			{
				int row_index = row_i + i;
				int temp0 = row_index*width;
				int temp1 = i*sub_width;

				for (int j = 0; j < sub_width; j++)
				{
					int col_index = j + col_j;
					//快速fabs
					float x = ((float*)angleImg.data)[temp0 + col_index]
						- ((float*)angleImg_sub.data)[temp1 + j];
					int casted = *(int*)&x;
					casted &= 0x7FFFFFFF;
					x = *(float*)&casted;
					int n = x;
					//累加误差值
					total_diff += n;
				}
			}
			if (minsearch > total_diff)
			{
				minsearch = total_diff;
				temp_x = col_j;
				temp_y = row_i;
			}

		}
	}
	*x = temp_x;
	*y = temp_y;
	return SUB_IMAGE_MATCH_OK;
}
int ustc_SubImgMatch_mag(Mat grayImg, Mat subImg, int* x, int* y)
{
	//检查是否为空
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	//检查通道数是否匹配
	if (grayImg.channels() != 1 || 1 != subImg.channels())
	{
		cout << "channels ERROR" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	//检测图像大小是否符合要求
	if (width < sub_width || height < sub_height)
	{
		cout << "width or height ERROR " << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	Mat magImg(height, width, CV_32FC1);
	Mat gradImg_x(height, width, CV_32FC1);
	Mat gradImg_y(height, width, CV_32FC1);
	ustc_CalcGrad(grayImg, gradImg_x, gradImg_y);

	Mat magImg_sub(sub_height, sub_width, CV_32FC1);
	Mat gradImg_subx(sub_height, sub_width, CV_32FC1);
	Mat gradImg_suby(sub_height, sub_width, CV_32FC1);
	ustc_CalcGrad(subImg, gradImg_subx, gradImg_suby);
	magImg.setTo(0);
	magImg_sub.setTo(0);
	for (int row_i = 1; row_i < height - 1; row_i++)
	{
		int temp0 = row_i*width;
		for (int col_j = 1; col_j < width - 1; col_j++)
		{
			int temp1 = temp0 + col_j;
			float grad_x = ((float*)gradImg_x.data)[temp1];
			float grad_y = ((float*)gradImg_y.data)[temp1];
			float mag = sqrt(grad_x*grad_x + grad_y*grad_y);
			((float*)magImg.data)[temp1] = mag;
		}
	}

	for (int row_i = 1; row_i < sub_height - 1; row_i++)
	{
		int temp0 = row_i*sub_width;
		for (int col_j = 1; col_j < sub_width - 1; col_j++)
		{
			int temp1 = temp0 + col_j;
			float grad_x = ((float*)gradImg_subx.data)[temp1];
			float grad_y = ((float*)gradImg_suby.data)[temp1];
			float mag = sqrt(grad_x*grad_x + grad_y*grad_y);
			((float*)magImg_sub.data)[temp1] = mag;
		}
	}
	Mat searchImg(height, width, CV_32FC1);
	int temp_x = 0, temp_y = 0;
	//匹配误差初始化
	searchImg.setTo(FLT_MAX);
	float minsearch = FLT_MAX;
			for (int row_i = 0; row_i <height - sub_height; row_i++)
			{
					for (int col_j = 0; col_j< width - sub_width; col_j++)
					{
						int total_diff = 0;

						for (int i = 0; i < sub_height; i++)
						{
							int row_index = row_i + i;
							int temp0 = row_index*width;
							int temp1 = i*sub_width;

							for (int j = 0; j < sub_width; j++)
							{
								int col_index = j + col_j;
								//快速fabs
								float x = ((float*)magImg.data)[temp0 + col_index]
									- ((float*)magImg_sub.data)[temp1 + j];
								int casted = *(int*)&x;
								casted &= 0x7FFFFFFF;
								x = *(float*)&casted;
								int n = x;
								//累加误差值
								total_diff += n;
							}
						}
						if (minsearch > total_diff)
						{
							minsearch = total_diff;
							temp_x = col_j;
							temp_y = row_i;
						}

					}
				}
			*x = temp_x;
			*y = temp_y;
				return SUB_IMAGE_MATCH_OK;
}
int ustc_SubImgMatch_hist(Mat grayImg, Mat subImg, int* x, int* y) {
	//函数实现
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	//检查通道数是否匹配
	if (grayImg.channels() != 1 || 1 != subImg.channels())
	{
		cout << "channels ERROR" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	//检测图像大小是否符合要求
	if (width < sub_width || height < sub_height)
	{
		cout << "width or height ERROR " << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg(height, width, CV_32FC1);
	//匹配误差初始化
	searchImg.setTo(FLT_MAX);

	//遍历大图每一个像素，注意行列的起始、终止坐标
	int* hist_temp = new int[256];
	memset(hist_temp, 0, sizeof(int) * 256);
	int* hist_sub = new int[256];
	memset(hist_sub, 0, sizeof(int) * 256);
	float minsearch = FLT_MAX;
	int temp_x = 0, temp_y = 0;
	for (int i = 0; i < sub_height; i++)
	{
		for (int j = 0; j < sub_width; j++)
		{
			//大图上的像素位置

			int subImg_pix = subImg.data[i * sub_width + j];
			if (subImg_pix > 255 || subImg_pix<0)
			{
				cout << "ERROR:pix>255 or pix<0 " << endl;
				return SUB_IMAGE_MATCH_FAIL;
			}
			hist_sub[subImg_pix]++;
		}
	}
	for (int row_i = 0; row_i < height - sub_height; row_i++)
	{
		for (int col_j = 0; col_j < width - sub_width; col_j++)
		{
			//清零
			memset(hist_temp, 0, sizeof(int) * 256);

			//计算当前位置直方图
			for (int i = 0; i < sub_height; i++)
			{
				int row_index = row_i + i;
				int temp = row_index * width;
				for (int j = 0; j < sub_width; j++)
				{
					//大图上的像素位置
					int bigImg_pix = grayImg.data[temp + col_j + j];
					if (bigImg_pix > 255 || bigImg_pix<0)
					{
						cout << "ERROR:pix>255 or pix<0 " << endl;
						return SUB_IMAGE_MATCH_FAIL;
					}
					hist_temp[bigImg_pix]++;
				}
			}

			//根据直方图计算匹配误差
			int total_diff = 0;
			for (int ii = 0; ii < 256; ii++)
			{
				total_diff += abs(hist_temp[ii] - hist_sub[ii]);
			}
			//存储当前像素位置的匹配误差
			((float*)searchImg.data)[row_i * width + col_j] = total_diff;
			if (minsearch > total_diff)
			{
				minsearch = total_diff;
				temp_y = row_i;
				temp_x = col_j;
			}
		}
	}
	*x = temp_x;
	*y = temp_y;
	delete[] hist_temp;
	delete[] hist_sub;
	return SUB_IMAGE_MATCH_OK;
}
