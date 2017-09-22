#include "SubImageMatch.h"
int ustc_ConvertBgr2Gray(Mat bgrImg, Mat& grayImg)
{
	if (NULL == bgrImg.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}

	int width = bgrImg.cols;
	int height = bgrImg.rows;
	for (int row_i = 0; row_i < height; row_i++)
	{
		for (int col_j = 0; col_j < width; col_j += 1)
		{
			int b = bgrImg.data[3 * (row_i * width + col_j) + 0];
			int g = bgrImg.data[3 * (row_i * width + col_j) + 1];
			int r = bgrImg.data[3 * (row_i * width + col_j) + 2];

			int grayVal = (b * 11 + g * 59 + r * 30) / 100;
			grayImg.data[row_i * width + col_j] = grayVal;
		}
	}

#ifdef IMG_SHOW
	/*namedWindow("grayImg", 0);
	imshow("grayImg", grayImg);
	waitKey();*/
#endif
}
 
int ustc_CalcGrad(Mat grayImg, Mat& gradImg_x, Mat& gradImg_y)
{
	if (NULL == grayImg.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;

	Mat gradImg_x_ext(height, width, CV_32FC1);
	Mat gradImg_y_ext(height, width, CV_32FC1);
	//计算x方向梯度图
	for (int row_i = 1; row_i < height - 1; row_i++)
	{
		for (int col_j = 1; col_j < width - 1; col_j += 1)
		{
			int grad_x =
				grayImg.data[(row_i + 1) * width + col_j - 1]
				+ 2 * grayImg.data[(row_i +1)* width + col_j]
				+ grayImg.data[(row_i + 1)* width + col_j + 1]
				- grayImg.data[(row_i - 1) * width + col_j - 1]
				- 2 * grayImg.data[(row_i -1)* width + col_j]
				- grayImg.data[(row_i - 1)* width + col_j + 1];

			int grad_y =
				grayImg.data[(row_i - 1) * width + col_j + 1]
				+ 2 * grayImg.data[(row_i)* width + col_j + 1]
				+ grayImg.data[(row_i + 1)* width + col_j + 1]
				- grayImg.data[(row_i - 1) * width + col_j - 1]
				- 2 * grayImg.data[(row_i)* width + col_j - 1]
				- grayImg.data[(row_i + 1)* width + col_j - 1];
			((float*)gradImg_x_ext.data)[row_i * width + col_j] = grad_x;
			((float*)gradImg_y_ext.data)[row_i * width + col_j] = grad_y;
		}
	}
	for (int row_i = 0; row_i < height; row_i++)
	{
		for (int col_j = 0; col_j < width; col_j += 1)
		{
			int val_x = ((float*)gradImg_x_ext.data)[row_i * width + col_j];
			int val_y = ((float*)gradImg_y_ext.data)[row_i * width + col_j];
			gradImg_x.data[row_i * width + col_j] = abs(val_x);
			gradImg_y.data[row_i * width + col_j] = abs(val_y);
		}
	}
	/*namedWindow("gradImg_x", 0);
	imshow("gradImg_x", gradImg_x);
	namedWindow("gradImg_y", 0);
	imshow("gradImg_y", gradImg_y);
	waitKey();*/
}
int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y,Mat& angleImg, Mat& magImg)
{
	if (NULL == gradImg_x.data || NULL == gradImg_y.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}

	int width = gradImg_x.cols;
	int height = gradImg_x.rows;

	Mat angleImg_ext(height, width, CV_32FC1);
	Mat magImg_ext(height, width, CV_32FC1);
	angleImg.setTo(0);

	//计算角度图
	for (int row_i = 1; row_i < height - 1; row_i++)
	{
		for (int col_j = 1; col_j < width - 1; col_j++)
		{
			
			 
			float grad_x = gradImg_x.data[row_i * width + col_j];
			 
			float grad_y = gradImg_y.data[row_i * width + col_j];
			 
			float angle = atan2(grad_y, grad_x);
			float m = grad_x*grad_x;
			float n = grad_y*grad_y;
			float mag = sqrt(m+n);
			//自己找办法优化三角函数速度，并且转化为角度制，规范化到0-360
			angleImg_ext.data[row_i * width + col_j] = angle;
			magImg_ext.data[row_i * width + col_j] = mag;
			 
		}
	}

#ifdef IMG_SHOW
	//为了方便观察，进行些许变化
	for (int row_i = 0; row_i < height; row_i++)
	{
		for (int col_j = 0; col_j < width; col_j += 1)
		{
			float mag= magImg_ext.data[row_i * width + col_j];
			float angle = angleImg_ext.data[row_i * width + col_j];
			angle *= 180 / CV_PI;
			angle += 180;
			//为了能在8U上显示，缩小到0-180之间
			angle /= 2;
			angleImg.data[row_i * width + col_j] = angle;
			magImg.data[row_i*width + col_j] = mag;
		}
	}
	namedWindow("angle", 0);
	imshow("angle", angleImg);
	namedWindow("mag", 0);
	imshow("mag", magImg);
	waitKey();
#endif
}

int ustc_Threshold(Mat grayImg, Mat& binaryImg, int th)
{
	if (NULL == grayImg.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;

	for (int row_i = 0; row_i < height; row_i++)
	{
		int temp0 = row_i * width;
		for (int col_j = 0; col_j < width; col_j += 2)
		{
			//int pixVal = grayImg.at<uchar>(row_i, col_j);
			int temp1 = temp0 + col_j;
			int pixVal = grayImg.data[temp1];
			int dstVal = 0;
			if (pixVal > th)
			{
				dstVal = 255;
			}
			else if (pixVal <= th)
			{
				dstVal = 0;
			}
			//binaryImg.at<uchar>(row_i, col_j) = dstVal;
			binaryImg.data[temp1] = dstVal;
		}
	}

#ifdef IMG_SHOW
	namedWindow("binaryImg", 0);
	imshow("binaryImg", binaryImg);
	waitKey();
#endif

	return MY_OK;
}


int ustc_CalcHist(Mat grayImg, int* hist, int hist_len)
{
	if (NULL == grayImg.data || NULL == hist)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}

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
		for (int col_j = 0; col_j < width; col_j += 1)
		{
			int pixVal = grayImg.data[row_i * width + col_j];
			hist[pixVal]++;
		}
	}
	/*for (int i = 0; i < hist_len; i++)
		printf("%d: %d\n", i,hist[i]);
	waitKey();*/
}


int ustc_SubImgMatch_gray(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int height_diff = height - sub_height;
	int width_diff = width - sub_width;
	int result_i = 0;
	int result_j = 0;


	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg(height, width, CV_32FC1);
	//匹配误差初始化
	searchImg.setTo(FLT_MAX);
	Mat resultImg(sub_height, sub_width, CV_8UC1);

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i <height_diff; i++)
	{
		for (int j = 0; j <width_diff; j++)
		{
			int total_diff = 0;
			//遍历模板图上的每一个像素
			for (int x = 0; x < sub_height; x++)
			{
				for (int y = 0; y < sub_width; y++)
				{
					//大图上的像素位置
					int row_index = i + x;
					int col_index = j + y;
					int bigImg_pix = grayImg.data[row_index * width + col_index];
					//模板图上的像素
					int template_pix = subImg.data[x * sub_width + y];

					total_diff += abs(bigImg_pix - template_pix);

				}
			 
			
			}
			 
			int seq = i*width + j;
			//存储当前像素位置的匹配误差
			 
			((float*)searchImg.data)[seq] = total_diff;
			//printf("%d %f",total_diff, ((float*)searchImg.data)[seq]);

		}
	}
	float total_diff = FLT_MAX;
	 
	for (int i = 0; i < height_diff; i++)
	{
		for (int j = 0; j < width_diff; j++)
		{
			if (((float*)searchImg.data)[i*width + j] < total_diff)
			{
				total_diff = ((float*)searchImg.data)[i*width + j];
				result_i = i;
				result_j = j;
			}
		}
	}
	printf("%d %d\n", result_j, result_i);
	resultImg = grayImg(Rect(result_j, result_i, sub_width, sub_height)).clone();
	namedWindow("resultImg", 0);
	imshow("resultImg", resultImg);
	namedWindow("subImg", 0);
	imshow("subImg", subImg);
	waitKey();
}

int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int* x, int* y)
{
	if (NULL == colorImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}
	int width = colorImg.cols;
	int height = colorImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int height_diff = height - sub_height;
	int width_diff = width - sub_width;
	int result_i = 0;
	int result_j = 0;


	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg(height, width, CV_32FC3);
	//匹配误差初始化
	searchImg.setTo(FLT_MAX);
	Mat resultImg(sub_height, sub_width, CV_8UC3);

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i <height_diff; i++)
	{
		for (int j = 0; j <width_diff; j++)
		{
			int total_diff_B = 0;
			int total_diff_G = 0;
			int total_diff_R = 0;
			int total_diff = 0;
			
			//遍历模板图上的每一个像素
			for (int x = 0; x < sub_height; x++)
			{
				for (int y = 0; y < sub_width; y++)
				{
					//大图上的像素位置
					int row_index = i + x;
					int col_index = j + y;
					
					int bigImg_pix_B = colorImg.data[3*(row_index * width + col_index)];
					int template_pix_B = subImg.data[3*(x * sub_width + y)];
					total_diff_B += abs(bigImg_pix_B - template_pix_B);
					
					int bigImg_pix_G = colorImg.data[3*(row_index * width + col_index)+1];
					int template_pix_G = subImg.data[3*(x * sub_width + y)+1];
					total_diff_G += abs(bigImg_pix_G - template_pix_G);

					int bigImg_pix_R = colorImg.data[3*(row_index * width + col_index)+2];
					int template_pix_R = subImg.data[3*(x * sub_width + y)+2];
					total_diff_G += abs(bigImg_pix_R - template_pix_R);

					total_diff = total_diff_B + total_diff_G + total_diff_R;

				}
			}
			int seq = i*width + j;
			//存储当前像素位置的匹配误差
			((float*)searchImg.data)[seq] = total_diff;


		}
	}
	float total_diff = FLT_MAX;
	for (int i = 0; i < height_diff; i++)
	{
		for (int j = 0; j < width_diff; j++)
		{
			if (((float*)searchImg.data)[i*width + j] < total_diff)
			{
				total_diff = ((float*)searchImg.data)[i*width + j];
				result_i = i;
				result_j = j;
			}
		}
	}
	*x = result_i;
	*y = result_j;
	printf("%d %d\n",result_j, result_i);
	resultImg = colorImg(Rect(result_j, result_i, sub_width, sub_height)).clone();
	namedWindow("resultImg", 0);
	imshow("resultImg", resultImg);
	namedWindow("subImg", 0);
	imshow("subImg", subImg);
	waitKey();
}

int ustc_SubImgMatch_corr(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int height_diff = height - sub_height;
	int width_diff = width - sub_width;
	int result_i = 0;
	int result_j = 0;


	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg(height, width, CV_32FC1);
	//匹配误差初始化
	searchImg.setTo(FLT_MAX);
	Mat resultImg(sub_height, sub_width, CV_8UC1);

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i <height_diff; i++)
	{
		for (int j = 0; j <width_diff; j++)
		{
			int total_diff = 0;
			int pm_1 = 0;
			int pm_2 = 0;
			int pm_3 = 0;

			for (int x = 0; x < sub_height; x++)
			{
				for (int y = 0; y < sub_width; y++)
				{
					 
					int row_index = i + x;
					int col_index = j + y;
					int  vab_1= grayImg.data[row_index * width + col_index]*subImg.data[x*sub_width + y];
					int vab_2 = grayImg.data[row_index*width + col_index] * grayImg.data[row_index*width + col_index];
					int vab_3 = subImg.data[x*sub_width + y] * subImg.data[x*sub_width + y];
					pm_1 += vab_1;
					pm_2 += vab_2;
					pm_3 += vab_3;

				}
			}
			int seq = i*width + j;
			float brts ;
			//存储当前像素位置的匹配误差
			//printf("seq:%d\n");
			pm_2 = sqrt(pm_2);
			pm_3 = sqrt(pm_3);
			brts = ((float)pm_1/(pm_2*pm_3));
			((float*)searchImg.data)[seq] = brts;
		}
	}
	float brts = FLT_MAX;
	for (int i = 0; i < height_diff; i++)
	{
		for (int j = 0; j < width_diff; j++)
		{
			if (abs(((float*)searchImg.data)[i*width + j]-1 )<abs( brts-1))
			{
				brts = ((float*)searchImg.data)[i*width + j];
				result_i = i;
				result_j = j;
				 
			}
		}
	}
	*x = result_i;
	*y = result_j;
	printf("%d %d\n", result_j, result_i);
	resultImg = grayImg(Rect(result_j, result_i, sub_width, sub_height)).clone();
	namedWindow("resultImg", 0);
	imshow("resultImg", resultImg);
	namedWindow("subImg", 0);
	imshow("subImg", subImg);
	waitKey();
}

int ustc_SubImgMatch_angle(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int height_diff = height - sub_height;
	int width_diff = width - sub_width;
	int result_i = 0;
	int result_j = 0;


	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg(height, width, CV_32FC1);
	//匹配误差初始化
	searchImg.setTo(FLT_MAX);
	Mat resultImg(sub_height, sub_width, CV_8UC1);

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 1; i <height_diff - 1; i++)
	{
		for (int j = 1; j <width_diff - 1; j++)
		{
			float total_diff = 0;
			//遍历模板图上的每一个像素
			for (int x = 0; x < sub_height; x++)
			{
				for (int y = 0; y < sub_width; y++)
				{
					//大图上的像素位置
					//printf("%d\n", y);
					int row_i = i + x;
					int col_j = j + y;
					//printf("%d %d\n", (row_i + 1) * width + col_j - 1, (row_i - 1) * width + col_j - 1);
					int grayImg_grad_x = 
						grayImg.data[(row_i + 1) * width + col_j - 1]
						+ 2 * grayImg.data[(row_i + 1)* width + col_j]
						+ grayImg.data[(row_i + 1)* width + col_j + 1]
						- grayImg.data[(row_i - 1) * width + col_j - 1]
						- 2 * grayImg.data[(row_i - 1)* width + col_j]
						- grayImg.data[(row_i - 1)* width + col_j + 1];
					//printf("%d\n", y);
					int grayImg_grad_y =
						grayImg.data[(row_i - 1) * width + col_j + 1]
						+ 2 * grayImg.data[(row_i)* width + col_j + 1]
						+ grayImg.data[(row_i + 1)* width + col_j + 1]
						- grayImg.data[(row_i - 1) * width + col_j - 1]
						- 2 * grayImg.data[(row_i)* width + col_j - 1]
						- grayImg.data[(row_i + 1)* width + col_j - 1];
				
					float  grayImg_part_angle = atan2(grayImg_grad_y,grayImg_grad_x);

					int subImg_grad_x=
						subImg.data[(x + 1) *sub_width + y - 1]
						+ 2 * subImg.data[(x + 1)* sub_width + y]
						+ subImg.data[(x + 1)* sub_width + y + 1]
						- subImg.data[(x - 1) * sub_width + y - 1]
						- 2 * subImg.data[(x - 1)* sub_width + y]
						- subImg.data[(x - 1)* sub_width + y + 1];
					int subImg_grad_y=	
						subImg.data[(x - 1) * sub_width + y + 1]
						+ 2 * subImg.data[(x)* sub_width + y + 1]
						+ subImg.data[(x + 1)* sub_width + y + 1]
						- subImg.data[(x - 1) * sub_width + y - 1]
						- 2 * subImg.data[(x)* sub_width + y - 1]
						- subImg.data[(x + 1)* sub_width + y - 1];
					float subImg_angle = atan2(subImg_grad_y, subImg_grad_x);
					
					total_diff += abs(grayImg_part_angle - subImg_angle);

				}
				 
			}
			int seq = (i-1)*width + j-1;
			//存储当前像素位置的匹配误差
			 
			((float*)searchImg.data)[seq] = total_diff;
			 
		}
		 
	}
	 
	float total_diff = FLT_MAX;
	for (int i = 0; i < height_diff-1; i++)
	{
		for (int j = 0; j < width_diff-1; j++)
		{
			if (((float*)searchImg.data)[i*width + j] < total_diff)
			{
				total_diff =((float*)searchImg.data)[i*width + j];
				result_i = i+1;
				result_j = j+1;
				//printf("%f\n", total_diff);
			}
		}
	}
	*x = result_i;
	*y = result_j;
	printf("%d %d\n", result_j, result_i);
	resultImg = grayImg(Rect(result_j, result_i, sub_width, sub_height)).clone();
	namedWindow("resultImg", 0);
	imshow("resultImg", resultImg);
	namedWindow("subImg", 0);
	imshow("subImg", subImg);
	waitKey();
}

int ustc_SubImgMatch_mag(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int height_diff = height - sub_height;
	int width_diff = width - sub_width;
	int result_i = 0;
	int result_j = 0;


	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg(height, width, CV_32FC1);
	//匹配误差初始化
	searchImg.setTo(FLT_MAX);
	Mat resultImg(sub_height, sub_width, CV_8UC1);

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 1; i <height_diff-1; i++)
	{
		for (int j = 1; j <width_diff-1; j++)
		{
			float total_diff = 0;
			//遍历模板图上的每一个像素
			for (int x = 0; x < sub_height; x++)
			{
				for (int y = 0; y < sub_width; y++)
				{
					//大图上的像素位置
					int row_i = i + x;
					int col_j = j + y;

					int grayImg_grad_x =
						grayImg.data[(row_i + 1) * width + col_j - 1]
						+ 2 * grayImg.data[(row_i + 1)* width + col_j]
						+ grayImg.data[(row_i + 1)* width + col_j + 1]
						- grayImg.data[(row_i - 1) * width + col_j - 1]
						- 2 * grayImg.data[(row_i - 1)* width + col_j]
						- grayImg.data[(row_i - 1)* width + col_j + 1];
					int grayImg_grad_y =
						grayImg.data[(row_i - 1) * width + col_j + 1]
						+ 2 * grayImg.data[(row_i)* width + col_j + 1]
						+ grayImg.data[(row_i + 1)* width + col_j + 1]
						- grayImg.data[(row_i - 1) * width + col_j - 1]
						- 2 * grayImg.data[(row_i)* width + col_j - 1]
						- grayImg.data[(row_i + 1)* width + col_j - 1];
					int m = grayImg_grad_x*grayImg_grad_x;
					int n = grayImg_grad_y*grayImg_grad_y;
					float  grayImg_part_mag = sqrt(m+n);

					int subImg_grad_x =
						subImg.data[(x + 1) *sub_width + y - 1]
						+ 2 * subImg.data[(x + 1)* sub_width + y]
						+ subImg.data[(x + 1)* sub_width + y + 1]
						- subImg.data[(x - 1) * sub_width + y - 1]
						- 2 * subImg.data[(x - 1)* sub_width + y]
						- subImg.data[(x - 1)* sub_width + y + 1];
					int subImg_grad_y =
						subImg.data[(x - 1) * sub_width + y + 1]
						+ 2 * subImg.data[(x)* sub_width + y + 1]
						+ subImg.data[(x + 1)* sub_width + y + 1]
						- subImg.data[(x - 1) * sub_width + y - 1]
						- 2 * subImg.data[(x)* sub_width + y - 1]
						- subImg.data[(x + 1)* sub_width + y - 1];
					int p = subImg_grad_x*subImg_grad_x;
					int q = subImg_grad_y*subImg_grad_y;
					float subImg_mag =sqrt(p+q)  ;

					total_diff += abs(grayImg_part_mag - subImg_mag);

				}

			}
			int seq = i*width + j;
			//存储当前像素位置的匹配误差
			//printf("seq:%d\n");
			((float*)searchImg.data)[seq] = total_diff;
			//printf("%f\n",total_diff);
		}
	}
	float total_diff = FLT_MAX;
	for (int i = 1; i < height_diff-1; i++)
	{
		for (int j = 1; j < width_diff-1; j++)
		{
			if (((float*)searchImg.data)[i*width + j] < total_diff)
			{
				total_diff = ((float*)searchImg.data)[i*width + j];
				result_i = i;
				result_j = j;
			}
		}
	}
	*x = result_i;
	*y = result_j;
	printf("%d %d\n", result_j, result_i);
	resultImg = grayImg(Rect(result_j, result_i, sub_width, sub_height)).clone();
	namedWindow("resultImg", 0);
	imshow("resultImg", resultImg);
	namedWindow("subImg", 0);
	imshow("subImg", subImg);
	waitKey();
}

int ustc_SubImgMatch_hist(Mat grayImg, Mat subImg, int* x, int* y)
{
	int hist_len = 256;
	int *sub_hist = new int[hist_len];
	ustc_CalcHist(subImg, sub_hist, hist_len);
	if (NULL == grayImg.data || NULL == sub_hist)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int height_diff = height - sub_height;
	int width_diff = width - sub_width;
	int result_i=0, result_j=0;
	Mat resultImg(sub_height, sub_width, CV_8UC1);

	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg(height, width, CV_32FC1);
	//匹配误差初始化
	searchImg.setTo(FLT_MAX);

	//遍历大图每一个像素，注意行列的起始、终止坐标
	int* hist_temp = new int[hist_len];
	memset(hist_temp, 0, sizeof(int) * hist_len);

	for (int i = 1; i < height_diff; i++)
	{
		//printf("this is the start %d %d \n", height_difference,i);
		for (int j=0;j<width_diff;j++)
		{
			//清零
			 
			memset(hist_temp, 0, sizeof(int) * hist_len);
			for (int x = 0; x < sub_height; x++)
			{
				for (int y = 0; y < sub_width; y++)
				{
					//大图上的像素位置
					int row_index = i + x;
					int col_index = j + y;
					int bigImg_pix = grayImg.data[row_index * width + col_index];
					hist_temp[bigImg_pix]++;
				}
			}
			
			//根据直方图计算匹配误差
			int total_diff = 0;
			for (int ii = 0; ii < hist_len; ii++)
			{
				total_diff += abs((hist_temp[ii] - sub_hist[ii])*ii);
				 
			}
			 
			//存储当前像素位置的匹配误差
			((float*)searchImg.data)[i * width + j] = total_diff;
			 
		}
	}
	float total_diff = FLT_MAX;
	for (int i = 0; i < height_diff; i++)
	{
		for (int j = 0; j < width_diff; j++)
		{
			if (((float*)searchImg.data)[i*width + j] < total_diff)
			{
				total_diff = searchImg.data[i*width + j];
				result_i = i;
				result_j = j;
			}
		}
	}
	*x = result_i;
	*y = result_j;
	printf("%d %d\n", result_j, result_i);
	resultImg = grayImg(Rect(result_j, result_i, sub_width, sub_height)).clone();
	namedWindow("resultImg", 0);
	imshow("resultImg", resultImg);
	namedWindow("subImg", 0);
	imshow("subImg", subImg);
	waitKey();
	delete[] hist_temp;
}
