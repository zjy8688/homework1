#include"SubImageMatch.h"
#include"opencv2/opencv.hpp"
#include"cv.h"
using namespace cv;
using namespace std;

///姜博文 PB14203249


#define SUB_IMAGE_MATCH_OK 1
#define SUB_IMAGE_MATCH_FAIL -1
//#define IMG_SHOW 		//控制是否输出 注释掉定义则不输出
 

//代码优化作业
//子项一 彩色转灰度
int  ustc_ConvertBgr2Gray(Mat bgrImg, Mat& grayImg)
{
	if (NULL == bgrImg.data)
	{
		cout << "image is NULL" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (int(bgrImg.channels()) != 3)
	{
		cout << "image is not bgr" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int height = bgrImg.rows;
	int width = bgrImg.cols;
	grayImg.setTo(0);

	int location = 0;
	for (int row_i = 0; row_i < height; ++row_i)//++在前省去构造对象的指令
	{
		int row_location = row_i * width;//充分利用缓存
		int location = row_location;
		for (int col_j = 0; col_j < width - 1; ++col_j)
		{
			//int location = row_location + col_j;//充分利用缓存
			++location;
			int B = 7472 * bgrImg.data[3 * location];
			int G = 38469 * bgrImg.data[3 * location + 1];//也许可以让cpu忙起来？
			int R = 19595 * bgrImg.data[3 * location + 2];
			//整数移位算法 16位精度 0.299 * 65536 = 19595
			int gray = (R + G + B) >> 16;//移位运算提升计算速度
										 //int gray = (R * 19595 + G * 38469 + B * 7472) >> 16;
			grayImg.data[location] = gray;

			++col_j;
			++location;
			B = 7472 * bgrImg.data[3 * location];
			G = 38469 * bgrImg.data[3 * location + 1];//也许可以让cpu忙起来？
			R = 19595 * bgrImg.data[3 * location + 2];
			//整数移位算法 16位精度 0.299 * 65536 = 19595
			gray = (R + G + B) >> 16;//移位运算提升计算速度
									 //int gray = (R * 19595 + G * 38469 + B * 7472) >> 16;
			grayImg.data[location] = gray;

		}
	}

#ifdef IMG_SHOW
	namedWindow("grayImg", 1);
	imshow("grayImg", grayImg);
	waitKey(0);
#endif // IMG_SHOW
	return SUB_IMAGE_MATCH_OK;
}
//子项二 梯度计算
int ustc_CalcGrad(Mat grayImg, Mat& gradImg_x, Mat& gradImg_y)
{
	if (NULL == grayImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (int(grayImg.channels()) != 1)
	{
		cout << "image is not gray" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int height = grayImg.rows;
	int width = grayImg.cols;
	gradImg_x.setTo(0);
	gradImg_y.setTo(0);
	int row_i;
	int col_j;
	//计算x/y方向梯度

	for (row_i = height - 1; row_i > 1; --row_i)
	{
		//为中间和右侧列赋初始值
		int lastMidle_UpAddDown = -grayImg.data[(row_i - 1) * width - 1] + grayImg.data[(row_i + 1) * width - 1];
		int lastRight_UpAddDown = -grayImg.data[(row_i - 1) * width] + grayImg.data[(row_i + 1) * width];

		int row_location = row_i * width;
		for (col_j = width - 1; col_j > 1; --col_j)
		{
			int midle = row_location + col_j;
			//int up = midle - width;
			//int down = midle + width;

			int grad_y =
				+lastMidle_UpAddDown + 2 * lastRight_UpAddDown;
			//-grayImg.data[up - 1]
			//- 2 * grayImg.data[up]
			///- grayImg.data[up + 1]
			//+ grayImg.data[down - 1]
			//+ 2 * grayImg.data[down ]
			///+ grayImg.data[down + 1];

			lastMidle_UpAddDown = lastRight_UpAddDown;
			int nowRightUp = grayImg.data[midle - width + 1];
			int nowRightdown = grayImg.data[midle + width + 1];
			lastRight_UpAddDown = -nowRightUp + nowRightdown;
			grad_y += lastRight_UpAddDown;
			((float*)gradImg_y.data)[midle] = grad_y;

			int grad_x =
				-grayImg.data[midle - width - 1]
				- 2 * grayImg.data[midle - 1]
				- grayImg.data[midle + width - 1]
				+ nowRightUp
				+ 2 * grayImg.data[midle + 1]
				+ nowRightdown;
			((float*)gradImg_x.data)[midle] = grad_x;
		}
	}
#ifdef IMG_SHOW
	Mat gradImg_x_8U(height, width, CV_8UC1);
	Mat gradImg_y_8U(height, width, CV_8UC1);
	//Mat gradImg_8U(height, width, CV_8UC1);
	gradImg_x_8U.setTo(0);
	gradImg_y_8U.setTo(0);

	for (row_i = 1; row_i < height; row_i++)
	{
		for (col_j = 1; col_j < width; col_j++)
		{
			int val_x = ((float*)gradImg_x.data)[row_i*width + col_j];
			int val_y = ((float*)gradImg_y.data)[row_i*width + col_j];
			//int val = sqrt(val_x*val_x + val_y*val_y);
			gradImg_x_8U.data[row_i*width + col_j] = abs(val_x);
			gradImg_y_8U.data[row_i*width + col_j] = abs(val_y);
			//gradImg_8U.data[row_i*width + col_j] = abs(val);

		}
	}
	namedWindow("gradImg_x", 1);
	imshow("gradImg_x", gradImg_x_8U);
	namedWindow("gradImg_y", 1);
	imshow("gradImg_y", gradImg_y_8U);
	//namedWindow("gradImg", 1);
	//imshow("gradImg", gradImg_8U);
	waitKey(0);
#endif // IMG_SHOW
	return SUB_IMAGE_MATCH_OK;

}
//子项三 角度和幅值计算
int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y, Mat& angleImg, Mat& magImg)
{
	if (NULL == gradImg_x.data || NULL == gradImg_y.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (int(gradImg_x.channels()) != 1 || int(gradImg_y.channels()) != 1)
	{
		cout << "image is not gray" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int height = gradImg_x.rows;
	int width = gradImg_x.cols;
	angleImg.setTo(0);
	magImg.setTo(0);
	int row_i;
	int col_j;

	//float my_atan2[10000];
	//float my_sqrt[500][500];
	//for (int i = 0; i < 500; ++i)
	//{
	//	for (int j = 0; j < 500; ++j)
	//	{
	//		my_sqrt[i][j] = sqrt(i*i+j*j);
	//	}
	//}
	///Mat originalangleImg(height, width, CV_32FC1);

	for (row_i = height - 1; row_i > 1; --row_i)
	{
		int row_location = row_i * width;

		for (col_j = width - 1; col_j > 1; --col_j)
		{
			int location = row_location + col_j;
			int grad_x = (int)((float*)gradImg_x.data)[location];
			int grad_y = (int)((float*)gradImg_y.data)[location];
			float angle = atan2(grad_y, grad_x);
			float mag = sqrt(grad_x*grad_x + grad_y*grad_y);

			((float*)angleImg.data)[location] = (angle*57.29 + 180);

			((float*)magImg.data)[location] = mag;
		}
	}
#ifdef IMG_SHOW
	Mat angleImg_8U(height, width, CV_8UC1);
	///Mat originalangleImg_8U(height, width, CV_8UC1);
	Mat magImg_8U(height, width, CV_8UC1);
	for (row_i = 0; row_i < height; row_i++)
	{
		for (col_j = 0; col_j < width; col_j++)
		{
			float angle = ((float*)angleImg.data)[row_i * width + col_j];
			angle /= 2;
			angleImg_8U.data[row_i * width + col_j] = angle;
			///float originalangle = ((float*)originalangleImg.data)[row_i * width + col_j];
			///originalangle *= (float)180 / CV_PI;
			///originalangle += 180;
			///originalangleImg_8U.data[row_i * width + col_j] = originalangle;
			magImg_8U.data[row_i * width + col_j] = ((float*)magImg.data)[row_i * width + col_j];
			//cout << angle << endl;
		}
	}
	namedWindow("angleImg", 1);
	imshow("angleImg", angleImg_8U);
	///namedWindow("originalangleImg", 1);
	///	imshow("originalangleImg", originalangleImg_8U);
	namedWindow("magImg", 1);
	imshow("magImg", magImg_8U);
	waitKey();

#endif // IMG_SHOW
	return SUB_IMAGE_MATCH_OK;
}
//子项四 图像二值化
int ustc_Threshold(Mat grayImg, Mat& binaryImg, int th)
{
	if (NULL == grayImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (int(grayImg.channels()) != 1 )
	{
		cout << "image is not gray" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int height = grayImg.rows;
	int width = grayImg.cols;

	for (int row_i = height - 1; row_i >= 0; row_i--)
		for (int col_j = width - 1; col_j >= 0; col_j--)
		{
			int value = grayImg.data[row_i * width + col_j] - th >> 10;
			binaryImg.data[row_i * width + col_j] = (value + 1) * 255;
			/*if (int(grayImg.data[row_i * width + col_j]) >= th)
			{
			binaryImg.data[row_i * width + col_j] = 255;
			}
			else
			{
			binaryImg.data[row_i * width + col_j] = 0;
			}*/
		}

#ifdef IMG_SHOW
	namedWindow("Threshood", 1);
	imshow("Threshood", binaryImg);
	waitKey(0);
#endif // IMG_SHOW
	return SUB_IMAGE_MATCH_OK;
}
//子项五 统计灰度直方图
int ustc_CalcHist(Mat grayImg, int* hist, int hist_len)
{
	if (NULL == grayImg.data)
	{
		cout << "Image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (int(grayImg.channels()) != 1)
	{
		cout << "image is not gray2" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (hist_len != 256)
	{
		cout << "hist_len should be equal to 256" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	//hist = new int[256];

	for (int i = 0; i < hist_len; i++)
	{
		hist[i] = 0;
	}
	int height = grayImg.rows;
	int width = grayImg.cols;

	for (int row_i = height; row_i > 0; --row_i)
	{
		int row_location = row_i * width;
		for (int col_j = width; col_j > 0; --col_j)
		{
			hist[grayImg.data[row_location++]]++;
		}
	}
#ifdef  IMG_SHOW
	for (int i = 0; i < hist_len; i++)
	{
		int num = hist[i];
		cout << i << " ";
		while (num > 0)
		{
			cout << "*";
			num = num - 100;
		}
		cout << endl;
	}
#endif //  IMG_SHOW

	return SUB_IMAGE_MATCH_OK;
}
//字块六 基于亮度比较的字块匹配
int ustc_SubImgMatch_gray(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (grayImg.data == NULL || subImg.data == NULL)			//增加通道数比较是否相等的防御 //大图小于小图fail、
	{
		cout << "Image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (int(grayImg.channels()) != 1 || int(subImg.channels()) != 1)
	{
		cout << "image is not gray" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int height = grayImg.rows;
	int width = grayImg.cols;
	int sub_height = subImg.rows;
	int sub_width = subImg.cols;

	if (width < sub_width || height < sub_height)
	{
		cout << "subImg is larger than grayImg." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	Mat searchImg(height, width, CV_32FC1);
	searchImg.setTo(FLT_MAX);

	for (int i = 0; i < height - sub_height; ++i)
	{
		int originalRowlocation = i * width;
		for (int j = 0; j < width - sub_width; ++j)
		{
			float total_diff = 0;
			for (int x = sub_height - 1; x > 0; --x)
			{
				int row_location = originalRowlocation + x * width + j;
				int sub_row_location = x * sub_width;
				for (int y = sub_width - 1; y > 0; --y)
				{
					int out = int(grayImg.data[row_location + y] - subImg.data[sub_row_location + y]);
					int temp = out >> 31;
					out ^= temp;
					total_diff += (out - temp);
				}
			}
			((float*)searchImg.data)[originalRowlocation + j] = total_diff;
		}
	}
	int  tempMin = 1000000;
	for (int i = 0; i < height - sub_height; i++)
	{
		int row_location = i *width;
		for (int j = 0; j < width - sub_width; j++)
		{

			int diff = ((float*)searchImg.data)[row_location + j];
			//int temp = (diff - tempMin) >> 31 ;;
			if (diff < tempMin)
			{
				tempMin = diff;
				*x = j;
				*y = i;
			}
		}
	}
#ifdef IMG_SHOW
	for (int i = 0; i < sub_width; i++)
	{
		grayImg.data[*y * width + *x + i] = 0;
		grayImg.data[(*y + sub_height) * width + *x + i] = 0;
	}
	for (int i = 0; i < sub_height; i++)
	{
		grayImg.data[(*y + i) * width + *x] = 0;
		grayImg.data[(*y + i) * width + *x + sub_width] = 0;
	}
	namedWindow("SubImgMatch_gray", 1);
	imshow("SubImgMatch_gray", grayImg);
	waitKey(0);
#endif // IMG_SHOW

	return SUB_IMAGE_MATCH_OK;
}
//字块七 基于色彩比较的字块匹配
int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int* x, int* y)
{
	if (colorImg.data == NULL || subImg.data == NULL)
	{
		cout << "Image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (int(colorImg.channels()) != 3 || int(subImg.channels()) != 3)
	{
		cout << "image is not bgr" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int height = colorImg.rows;
	int width = colorImg.cols;
	int sub_height = subImg.rows;
	int sub_width = subImg.cols;

	if (width < sub_width || height < sub_height)
	{
		cout << "subImg is larger than grayImg." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	Mat searchImg(height, width, CV_32FC1);
	searchImg.setTo(FLT_MAX);

	for (int i = 0; i < height - sub_height; ++i)
	{
		int originalRowlocation = i * width;
		for (int j = 0; j < width - sub_width; ++j)
		{
			float total_diff = 0;
			for (int x = sub_height - 1; x > 0; --x)
			{
				int row_location = 3 * (originalRowlocation + x * width + j);
				int sub_row_location = 3 * (x * sub_width);
				for (int y = sub_width - 1; y > 0; --y)
				{
					int out = int(colorImg.data[row_location++] - subImg.data[sub_row_location++]);
					int temp = out >> 31;
					out ^= temp;
					total_diff += (out - temp);
					out = int(colorImg.data[row_location++] - subImg.data[sub_row_location++]);
					temp = out >> 31;
					out ^= temp;
					total_diff += (out - temp);
					out = int(colorImg.data[row_location++] - subImg.data[sub_row_location++]);
					temp = out >> 31;
					out ^= temp;
					total_diff += (out - temp);
				}
			}
			((float*)searchImg.data)[originalRowlocation + j] = total_diff;
		}
	}
	int  tempMin = 1000000;
	for (int i = 0; i < height - sub_height; i++)
	{
		int row_location = i * width;
		for (int j = 0; j < width - sub_width; j++)
		{

			int diff = ((float*)searchImg.data)[row_location + j];
			//cout << i << "\t" << j << "\t" << tempMin << "\t" << diff << endl;
			//int temp = (diff - tempMin) >> 31 ;;
			if (diff < tempMin)
			{
				tempMin = diff;
				*x = j;
				*y = i;
			}
		}
	}
#ifdef IMG_SHOW
	for (int i = 0; i < sub_width; i++)
	{
		int location1 = 3 * (*y * width + *x + i);
		int location2 = location1 + 3 * (sub_height * width);
		colorImg.data[location1] = 0;
		colorImg.data[location1 + 1] = 0;
		colorImg.data[location1 + 2] = 0;
		colorImg.data[location2] = 0;
		colorImg.data[location2 + 1] = 0;
		colorImg.data[location2 + 2] = 0;
	}
	for (int i = 0; i < sub_height; i++)
	{
		int location1 = 3 * ((*y + i) * width + *x);
		int location2 = location1 + 3 * sub_width;
		colorImg.data[location1] = 0;
		colorImg.data[location1 + 1] = 0;
		colorImg.data[location1 + 2] = 0;
		colorImg.data[location2] = 0;
		colorImg.data[location2 + 1] = 0;
		colorImg.data[location2 + 2] = 0;
	}
	namedWindow("SubImgMatch_color", 1);
	imshow("SubImgMatch_color", colorImg);
	waitKey(0);
#endif // IMG_SHOW
	return SUB_IMAGE_MATCH_OK;
}
//字块八 基于亮度相关性比较的字块匹配
int ustc_SubImgMatch_corr(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (grayImg.data == NULL || subImg.data == NULL)
	{
		cout << "Image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (int(grayImg.channels()) != 1 || int(subImg.channels()) != 1)
	{
		cout << "image is not gray" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int height = grayImg.rows;
	int width = grayImg.cols;
	int sub_height = subImg.rows;
	int sub_width = subImg.cols;

	if (width < sub_width || height < sub_height)
	{
		cout << "subImg is larger than grayImg."  << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	Mat searchImg(height, width, CV_32FC1);
	searchImg.setTo(FLT_MAX);

	int fenmy = 0;
	for (int x = sub_height - 1; x > 0; --x)
	{
		for (int y = sub_width - 1; y > 0; --y)
		{
			fenmy += subImg.data[x * sub_width + y] * subImg.data[x * sub_width + y];
		}
	}
	fenmy = sqrt(fenmy);

	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			int fenzi = 0;
			int fenmz = 0;

			for (int x = sub_height - 1; x > 0; --x)
			{
				int sub_row_location = x * sub_width;
				for (int y = sub_width - 1; y > 0; --y)
				{
					int location_data = grayImg.data[(i + x) * width + j + y];
					fenzi += location_data * subImg.data[sub_row_location + y];
					fenmz += location_data * location_data;

				}
			}
			fenmz = sqrt(fenmz);
			((float*)searchImg.data)[i * width + j] = (float)fenzi / (float)(fenmz * fenmy);
		}
	}
	float  tempMin = 1;
	for (int i = 0; i < height - sub_height; i++)
	{
		int row_location = i *width;
		for (int j = 0; j < width - sub_width; j++)
		{

			float diff = 1 - ((float*)searchImg.data)[row_location + j];
			//cout << i << "\t" << j << "\t" << tempMin << "\t" << diff << endl;
			//int temp = (diff - tempMin) >> 31 ;
			if (diff <= tempMin)
			{
				tempMin = diff;
				*x = j;
				*y = i;
			}
		}
	}
#ifdef IMG_SHOW
	for (int i = 0; i < sub_width; i++)
	{
		grayImg.data[*y * width + *x + i] = 0;
		grayImg.data[(*y + sub_height) * width + *x + i] = 0;
	}
	for (int i = 0; i < sub_height; i++)
	{
		grayImg.data[(*y + i) * width + *x] = 0;
		grayImg.data[(*y + i) * width + *x + sub_width] = 0;
	}
	namedWindow("SubImgMatch_gray", 1);
	imshow("SubImgMatch_gray", grayImg);
	waitKey(0);
#endif // IMG_SHOW

	return SUB_IMAGE_MATCH_OK;
}
//字块九 基于角度值比较的字块匹配
int ustc_SubImgMatch_angle(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (grayImg.data == NULL || subImg.data == NULL)			//增加通道数比较是否相等的防御 //大图小于小图fail、
	{
		cout << "Image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (int(grayImg.channels()) != 1 || int(subImg.channels()) != 1)
	{
		cout << "image is not gray" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int height = grayImg.rows;
	int width = grayImg.cols;
	int sub_height = subImg.rows;
	int sub_width = subImg.cols;

	if (width < sub_width || height < sub_height)
	{
		cout << "subImg is larger than grayImg."  << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	Mat gradImg_x(height, width, CV_32FC1);
	Mat gradImg_y(height, width, CV_32FC1);
	ustc_CalcGrad(grayImg, gradImg_x, gradImg_y);
	Mat angleImg(height, width, CV_32FC1);
	Mat magImg(height, width, CV_32FC1);
	ustc_CalcAngleMag(gradImg_x, gradImg_y, angleImg, magImg);
	Mat searchImg(height, width, CV_32FC1);
	searchImg.setTo(FLT_MAX);

	for (int i = 0; i < height - sub_height; i++)
	{
		int originalRowlocation = i * width;
		for (int j = 0; j < width - sub_width; j++)
		{
			int total_diff = 0;
			for (int x = 0; x < sub_height; x++)
			{
				int row_location = originalRowlocation + x * width + j;
				int sub_row_location = x * sub_width;
				for (int y = 0; y < sub_width; y++)
				{
					int out = int(((float*)angleImg.data)[row_location + y] - ((float*)subImg.data)[sub_row_location + y]);
					//cout << out;
					int temp = out >> 31;
					out ^= temp;
					total_diff += (out - temp);
					//cout << "\t" << out - temp << endl;
				}

			}
			((float*)searchImg.data)[i * width + j] = total_diff;
			//cout << i << "\t" << j << "\t" << total_diff << "\t" << ((float*)searchImg.data)[i * width + j] << endl;
		}
	}

	int  tempMin = 2000000;
	for (int i = 0; i < height - sub_height; i++)
	{
		int row_location = i *width;
		for (int j = 0; j < width - sub_width; j++)
		{

			int diff = ((float*)searchImg.data)[row_location + j];
			//int temp = (diff - tempMin) >> 31 ;;
			if (diff < tempMin)
			{
				tempMin = diff;
				*x = j;
				*y = i;
				cout << i << "\t" << j << "\t" << diff << "\t" << tempMin << endl;
			}
			//cout << i << "\t" << j << "\t" << diff << "\t" << tempMin << endl;
		}
	}
#ifdef IMG_SHOW
	Mat angleImg_8U(height, width, CV_8UC1);
	for (int row_i = 0; row_i < height; row_i++)
	{
		for (int col_j = 0; col_j < width; col_j++)
		{
			angleImg_8U.data[row_i * width + col_j] = ((float*)angleImg.data)[row_i * width + col_j];
		}
	}
	for (int i = 0; i < sub_width; i++)
	{
		angleImg_8U.data[*y * width + *x + i] = 0;
		angleImg_8U.data[(*y + sub_height) * width + *x + i] = 0;
	}
	for (int i = 0; i < sub_height; i++)
	{
		angleImg_8U.data[(*y + i) * width + *x] = 0;
		angleImg_8U.data[(*y + i) * width + *x + sub_width] = 0;
	}
	namedWindow("SubImgMatch_mag", 1);
	imshow("SubImgMatch_mag", angleImg_8U);
	waitKey(0);
#endif // IMG_SHOW
	return SUB_IMAGE_MATCH_OK;
}
//字块十 基于幅值比较的字块匹配
int ustc_SubImgMatch_mag(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (grayImg.data == NULL || subImg.data == NULL)		
	{
		cout << "Image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (int(grayImg.channels()) != 1 || int(subImg.channels()) != 1)
	{
		cout << "image is not gray" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int height = grayImg.rows;
	int width = grayImg.cols;
	int sub_height = subImg.rows;
	int sub_width = subImg.cols;

	if (width < sub_width || height < sub_height)
	{
		cout << "subImg is larger than grayImg." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	Mat gradImg_x(height, width, CV_32FC1);
	Mat gradImg_y(height, width, CV_32FC1);
	ustc_CalcGrad(grayImg, gradImg_x, gradImg_y);
	Mat angleImg(height, width, CV_32FC1);
	Mat magImg(height, width, CV_32FC1);
	ustc_CalcAngleMag(gradImg_x, gradImg_y, angleImg, magImg);

	Mat searchImg(height, width, CV_32FC1);
	searchImg.setTo(FLT_MAX);

	for (int i = 0; i < height - sub_height; i++)
	{
		int originalRowlocation = i * width;
		for (int j = 0; j < width - sub_width; j++)
		{
			int total_diff = 0;
			for (int x = 0; x < sub_height; x++)
			{
				int row_location = originalRowlocation + x * width + j;
				int sub_row_location = x * sub_width;
				for (int y = 0; y < sub_width; y++)
				{
					int out = int(((float*)magImg.data)[row_location + y] - ((float*)subImg.data)[sub_row_location + y]);
					//cout << out;
					int temp = out >> 31;
					out ^= temp;
					total_diff += (out - temp);
					//cout << "\t" << out - temp << endl;
				}
			}
			((float*)searchImg.data)[i * width + j] = total_diff;
			//cout << i << "\t" << j << "\t" << total_diff << "\t" << ((float*)searchImg.data)[i * width + j] << endl;
		}
	}

	int  tempMin = 1000000;
	for (int i = 0; i < height - sub_height; i++)
	{
		int row_location = i *width;
		for (int j = 0; j < width - sub_width; j++)
		{

			int diff = ((float*)searchImg.data)[row_location + j];
			//int temp = (diff - tempMin) >> 31 ;;
			if (diff < tempMin)
			{
				tempMin = diff;
				*x = j;
				*y = i;
				cout << i << "\t" << j << "\t" << diff << "\t" << tempMin << endl;
			}
			//cout << i << "\t" << j << "\t" << diff << "\t" << tempMin << endl;
		}
	}
#ifdef IMG_SHOW
	Mat magImg_8U(height, width, CV_8UC1);
	for (int row_i = 0; row_i < height; row_i++)
	{
		for (int col_j = 0; col_j < width; col_j++)
		{
			magImg_8U.data[row_i * width + col_j] = ((float*)magImg.data)[row_i * width + col_j];
		}
	}
	for (int i = 0; i < sub_width; i++)
	{
		magImg_8U.data[*y * width + *x + i] = 0;
		magImg_8U.data[(*y + sub_height) * width + *x + i] = 0;
	}
	for (int i = 0; i < sub_height; i++)
	{
		magImg_8U.data[(*y + i) * width + *x] = 0;
		magImg_8U.data[(*y + i) * width + *x + sub_width] = 0;
	}
	namedWindow("SubImgMatch_mag", 1);
	imshow("SubImgMatch_mag", magImg_8U);
	waitKey(0);
#endif // IMG_SHOW
	return SUB_IMAGE_MATCH_OK;
}
//字块十一 基于直方图比较的字块优化
int ustc_SubImgMatch_hist(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data )
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (int(grayImg.channels()) != 1 || int(subImg.channels()) != 1)
	{
		cout << "image is not gray1" << "\t"<< int(grayImg.channels()) <<"\t"<< int(subImg.channels()) <<endl;
		///return SUB_IMAGE_MATCH_FAIL;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	if (width < sub_width || height < sub_height)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg(height, width, CV_32FC1);
	//匹配误差初始化
	searchImg.setTo(FLT_MAX);
	int sub_hist[256];
	int hist_len = 256;
	ustc_CalcHist(subImg, sub_hist, hist_len);///

	//遍历大图每一个像素，注意行列的起始、终止坐标
	int* hist_temp = new int[hist_len];
	memset(hist_temp, 0, sizeof(int) * hist_len);

	for (int i = 0; i < height - sub_height; i++)
	{
		int row_loca = i * width;
		for (int j = 0; j < width - sub_width; j++)
		{
			int location = row_loca + j;
			//清零
			memset(hist_temp, 0, sizeof(int) * hist_len);

			//计算当前位置直方图
			for (int x = 0; x < sub_height; x++)
			{
				int row_location =  x * width + location;
				for (int y = 0; y < sub_width; y++)
				{
					hist_temp[grayImg.data[row_location++]]++;
				}
			}
			//根据直方图计算匹配误差
			int total_diff = 0;
			for (int ii = 0; ii < hist_len; ii++)
			{
				int out = int(hist_temp[ii] - sub_hist[ii]);
				int temp = out >> 31;
				out ^= temp;
				total_diff += (out - temp);
			}
			//存储当前像素位置的匹配误差
			((float*)searchImg.data)[location] = total_diff;
			
		}
	}
	float tempMin = 2000000;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			float diff = ((float*)searchImg.data)[i * width + j];
			if (diff < tempMin)
			{
				tempMin = diff;
				*x = j;
				*y = i;
				//cout << i << "\t" << j << "\t" << diff << endl;
			}
		}
	}
	delete[] hist_temp;
#ifdef IMG_SHOW
	for (int i = 0; i < sub_width; i++)
	{
		grayImg.data[*y * width + *x + i] = 0;
		grayImg.data[(*y + sub_height) * width + *x + i] = 0;
	}
	for (int i = 0; i < sub_height; i++)
	{
		grayImg.data[(*y + i) * width + *x] = 0;
		grayImg.data[(*y + i) * width + *x + sub_width] = 0;
	}
	namedWindow("SubImgMatch_hist", 1);
	imshow("SubImgMatch_hist", grayImg);
	waitKey(0);
#endif // IMG_SHOW
	return SUB_IMAGE_MATCH_OK;
}
/*



/*for (col_j = width - 1; col_j > 1; --col_j)
{
int location = row_location + col_j;
float grad_x = ((float*)gradImg_x.data)[location];
float grad_y = ((float*)gradImg_y.data)[location];
float truetan = grad_y / grad_x;
if (truetan > 573)
truetan = 573;
if (truetan < -573)
truetan = -573;
tru(500) *
if (tan > 99998 ||tan <0)
tan = 99999;
//float angle1 = tan - tan*tan*tan / 3 + tan*tan*tan*tan*tan / 5 - tan*tan*tan*tan*tan*tan*tan/7;
float angle = my_atan2[tan];
float originalangle = atan2(grad_y, grad_x);
//cout << abs(angle - angle1) << endl;
//float mag = my_sqrt[(int)(grad_x*grad_x + grad_y*grad_y)];
((float*)angleImg.data)[location] = angle;
((float*)originalangleImg.data)[location] = originalangle;
//((float*)magImg.data)[location] = mag;

}*/

//子项一 测试彩色转灰度
void test_ustc_ConvertBgr2Gray()
{
	Mat bgrImg = imread("D:\\bwjiang\\practice\\Modern Software Engineering\\hy.jpg");
	if (NULL == bgrImg.data)
	{
		cout << "image is NULL" << endl;
		return;
	}

#ifdef IMG_SHOW
	namedWindow("bgrImg", 1);
	imshow("bgrImg", bgrImg);
	waitKey(1);
#endif
	int height = bgrImg.rows;
	int width = bgrImg.cols;
	Mat grayImg(height, width, CV_8UC1);

	time_t start = clock();
	for (int i = 0; i < 1; i++)
	{
		int flag = ustc_ConvertBgr2Gray(bgrImg, grayImg);
	}
	time_t end = clock();

	cout << "time = " << (end - start) / 1000.0 << endl;
}
//子项四 测试图像二值化
void test_ustc_Threshold()
{
	Mat bgrImg = imread("D:\\bwjiang\\practice\\Modern Software Engineering\\hy.jpg");
	if (NULL == bgrImg.data)
	{
		cout << "image is NULL" << endl;
		return;
	}
#ifdef IMG_SHOW
	namedWindow("bgrImg", 1);
	imshow("bgrImg", bgrImg);
	waitKey(1);
#endif
	int height = bgrImg.rows;
	int width = bgrImg.cols;
	Mat grayImg(height, width, CV_8UC1);
	int flag = ustc_ConvertBgr2Gray(bgrImg, grayImg);
	Mat binaryImg(height, width, CV_8UC1);
	int th = 50;
	time_t start = clock();
	for (int i = 0; i < 3000; i++)
	{
		ustc_Threshold(grayImg, binaryImg, th);
	}
	time_t end = clock();
	cout << "time = " << (end - start) / 1000.0 << endl;
}
//子项二 测试梯度计算
void test_ustc_CalcGrad()
{
	Mat bgrImg = imread("D:\\bwjiang\\practice\\Modern Software Engineering\\hy.jpg");
	if (NULL == bgrImg.data)
	{
		cout << "image is NULL" << endl;
		return;
	}
#ifdef IMG_SHOW
	namedWindow("bgrImg", 1);
	imshow("bgrImg", bgrImg);
	waitKey(1);
#endif
	int height = bgrImg.rows;
	int width = bgrImg.cols;
	Mat grayImg(height, width, CV_8UC1);
	int flag = ustc_ConvertBgr2Gray(bgrImg, grayImg);
	Mat gradImg_x(height, width, CV_32FC1);
	Mat gradImg_y(height, width, CV_32FC1);

	time_t start = clock();
	for (int i = 0; i < 3000; i++)
	{
		ustc_CalcGrad(grayImg, gradImg_x, gradImg_y);
	}
	time_t end = clock();
	cout << "time = " << (end - start) / 1000.0 << endl;
}
//子项三 测试角度和幅值计算
void test_ustc_CalcAngleMag()
{
	Mat bgrImg = imread("D:\\bwjiang\\practice\\Modern Software Engineering\\hy.jpg");
	if (NULL == bgrImg.data)
	{
		cout << "image is NULL" << endl;
		return;
	}
#ifdef IMG_SHOW
	namedWindow("bgrImg", 1);
	imshow("bgrImg", bgrImg);
	waitKey(1);
#endif
	int height = bgrImg.rows;
	int width = bgrImg.cols;
	Mat grayImg(height, width, CV_8UC1);
	ustc_ConvertBgr2Gray(bgrImg, grayImg);
	Mat gradImg_x(height, width, CV_32FC1);
	Mat gradImg_y(height, width, CV_32FC1);
	ustc_CalcGrad(grayImg, gradImg_x, gradImg_y);
	Mat angleImg(height, width, CV_32FC1);
	Mat magImg(height, width, CV_32FC1);

	time_t start = clock();
	for (int i = 0; i < 1; i++)
	{
		ustc_CalcAngleMag(gradImg_x, gradImg_y, angleImg, magImg);
	}
	time_t end = clock();
	cout << "time = " << (end - start) / 1000.0 << endl;
}
//子项五 测试灰度直方图
void test_ustc_CalcHist()
{
	Mat bgrImg = imread("D:\\bwjiang\\practice\\Modern Software Engineering\\hy.jpg");
	if (NULL == bgrImg.data)
	{
		cout << "image is NULL" << endl;
		return;
	}
#ifdef IMG_SHOW
	namedWindow("bgrImg", 1);
	imshow("bgrImg", bgrImg);
	waitKey(1);
#endif
	int height = bgrImg.rows;
	int width = bgrImg.cols;
	Mat grayImg(height, width, CV_8UC1);
	int flag = ustc_ConvertBgr2Gray(bgrImg, grayImg);
	int hist[256];
	int hist_len = 256;
	time_t start = clock();
	for (int i = 0; i < 1; i++)
	{
		ustc_CalcHist(grayImg, hist, hist_len);
	}
	time_t end = clock();
	cout << "CalcHist_time = " << (end - start) / 1000.0 << endl;
}
//子项六 测试基于亮度比较的字块匹配
void test_ustc_SubImgMatch_gray()
{
	Mat bgrImg = imread("D:\\bwjiang\\practice\\Modern Software Engineering\\hy.jpg");
	if (NULL == bgrImg.data)
	{
		cout << "image is NULL" << endl;
		return;
	}

	int height = bgrImg.rows;
	int width = bgrImg.cols;
	Mat grayImg(height, width, CV_8UC1);
	ustc_ConvertBgr2Gray(bgrImg, grayImg);
	Mat subImg = imread("D:\\bwjiang\\practice\\Modern Software Engineering\\subhy1.jpg");
	if (NULL == subImg.data)
	{
		cout << "subimage is NULL" << endl;
		return;
	}
	int sub_height = subImg.rows;
	int sub_width = subImg.cols;
	Mat subGrayImg(sub_height, sub_width, CV_8UC1);
	ustc_ConvertBgr2Gray( subImg , subGrayImg);
	int *x = (int *)malloc(256 * sizeof(int));
	int *y = (int *)malloc(256 * sizeof(int));
	time_t start = clock();
	for (int i = 0; i < 1; i++)
	{
		ustc_SubImgMatch_gray(grayImg, subGrayImg, x, y);

	}
	time_t end = clock();
	cout << "time = " << (end - start) / 1000.0 << endl;
}
//子项七 测试基于色彩比较的字块匹配
void test_ustc_SubImgMatch_bgr()
{
	Mat bgrImg = imread("D:\\bwjiang\\practice\\Modern Software Engineering\\hy.jpg");
	if (NULL == bgrImg.data)
	{
		cout << "image is NULL" << endl;
		return;
	}
	int height = bgrImg.rows;
	int width = bgrImg.cols;

	Mat subImg = imread("D:\\bwjiang\\practice\\Modern Software Engineering\\subhy1.jpg");
	if (NULL == subImg.data)
	{
		cout << "subimage is NULL" << endl;
		return;
	}
	int sub_height = subImg.rows;
	int sub_width = subImg.cols;

	int *x = (int *)malloc(256 * sizeof(int));
	int *y = (int *)malloc(256 * sizeof(int));
	time_t start = clock();
	for (int i = 0; i < 1; i++)
	{
		ustc_SubImgMatch_bgr(bgrImg, subImg, x, y);

	}
	time_t end = clock();
	cout << "time = " << (end - start) / 1000.0 << endl;
}
//子项八 测试基于亮度相关性比较的字块匹配
void test_ustc_SubImgMatch_corr()
{
	Mat bgrImg = imread("D:\\bwjiang\\practice\\Modern Software Engineering\\hy.jpg");
	if (NULL == bgrImg.data)
	{
		cout << "image is NULL" << endl;
		return;
	}
	int height = bgrImg.rows;
	int width = bgrImg.cols;
	Mat grayImg(height, width, CV_8UC1);
	ustc_ConvertBgr2Gray(bgrImg, grayImg);
	Mat subImg = imread("D:\\bwjiang\\practice\\Modern Software Engineering\\subhy1.jpg");
	if (NULL == subImg.data)
	{
		cout << "subimage is NULL" << endl;
		return;
	}
	int sub_height = subImg.rows;
	int sub_width = subImg.cols;
	Mat subGrayImg(sub_height, sub_width, CV_8UC1);
	ustc_ConvertBgr2Gray(subImg, subGrayImg);
	int *x = (int *)malloc(256 * sizeof(int));
	int *y = (int *)malloc(256 * sizeof(int));
	time_t start = clock();
	for (int i = 0; i < 1; i++)
	{
		ustc_SubImgMatch_corr(grayImg, subGrayImg, x, y);
	}
	time_t end = clock();
	cout << "time = " << (end - start) / 1000.0 << endl;
}
//子项九/十 测试基于幅度和角度比较的字块匹配
void test_ustc_SubImgMatch_angle_mag()
{
	Mat bgrImg = imread("D:\\bwjiang\\practice\\Modern Software Engineering\\hy.jpg");
	if (NULL == bgrImg.data)
	{
		cout << "image is NULL" << endl;
		return;
	}
	int height = bgrImg.rows;
	int width = bgrImg.cols;
	Mat grayImg(height, width, CV_8UC1);
	ustc_ConvertBgr2Gray(bgrImg, grayImg);

	Mat subImg = imread("D:\\bwjiang\\practice\\Modern Software Engineering\\subhy1.jpg");
	if (NULL == subImg.data)
	{
		cout << "subimage is NULL" << endl;
		return;
	}
	int sub_height = subImg.rows;
	int sub_width = subImg.cols;
	Mat subGrayImg(sub_height, sub_width, CV_8UC1);
	ustc_ConvertBgr2Gray(subImg, subGrayImg);
	Mat subGradImg_x(sub_height, sub_width, CV_32FC1);
	Mat subGradImg_y(sub_height, sub_width, CV_32FC1);
	ustc_CalcGrad(subGrayImg, subGradImg_x, subGradImg_y);
	Mat subAngleImg(sub_height, sub_width, CV_32FC1);
	Mat subMagImg(sub_height, sub_width, CV_32FC1);
	ustc_CalcAngleMag(subGradImg_x, subGradImg_y, subAngleImg, subMagImg);
	int *x = (int *)malloc(256 * sizeof(int));
	int *y = (int *)malloc(256 * sizeof(int));

	time_t start = clock();
	for (int i = 0; i < 1; i++)
	{
		//ustc_SubImgMatch_angle(grayImg, subAngleImg, x, y);
		ustc_SubImgMatch_mag(grayImg, subMagImg, x, y);
	}
	time_t end = clock();
	cout << "time = " << (end - start) / 1000.0 << endl;
}
//子项十一 测试基于直方图比较的字块优化
void test_ustc_SubImgMatch_hist()
{
	Mat bgrImg = imread("D:\\bwjiang\\practice\\Modern Software Engineering\\hy.jpg");
	if (NULL == bgrImg.data)
	{
		cout << "image is NULL" << endl;
		return;
	}
	int height = bgrImg.rows;
	int width = bgrImg.cols;
	Mat grayImg(height, width, CV_8UC1);
	ustc_ConvertBgr2Gray(bgrImg, grayImg);

	Mat subImg = imread("D:\\bwjiang\\practice\\Modern Software Engineering\\subhy1.jpg");
	if (NULL == subImg.data)
	{
		cout << "subimage is NULL" << endl;
		return;
	}
	int sub_height = subImg.rows;
	int sub_width = subImg.cols;
	Mat subGrayImg(sub_height, sub_width, CV_8UC1);
	ustc_ConvertBgr2Gray(subImg, subGrayImg);
	time_t start = clock();
	int sub_hist[256];
	int hist_len = 256;

	int *x = (int *)malloc(256 * sizeof(int));
	int *y = (int *)malloc(256 * sizeof(int));
	for (int i = 0; i < 1; i++)
	{
		ustc_SubImgMatch_hist(grayImg, subGrayImg, x, y);
	}
	time_t end = clock();
	cout << "time = " << (end - start) / 1000.0 << endl;
}
