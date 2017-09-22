
#include "SubImageMatch.h"
/*#include "opencv2/opencv.hpp"
using namespace cv;
#include <iostream>
#include <vector>
using namespace std;
#include <time.h>
#define IMG_SHOW*/

#define SUB_IMAGE_MATCH_OK 1
#define SUB_IMAGE_MATCH_FAIL 0

//函数功能：将bgr图像转化成灰度图像
//bgrImg：彩色图，像素排列顺序是bgr
//grayImg：灰度图，单通道
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_ConvertBgr2Gray(Mat bgrImg, Mat& grayImg)

{
	if (NULL == bgrImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (bgrImg.channels() != 3 || grayImg.channels() != 1)
	{
		cout << "image channel  wrong." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = bgrImg.cols;
	int height = bgrImg.rows;
	if (width != grayImg.cols || height != grayImg.rows)
	{
		cout << "image channel  wrong." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int all = width*height;
	int b, g, r, grayVal;
	for (int line = 0; line < all; line++)
	{
		b = bgrImg.data[3*line+0];
			g = bgrImg.data[3*line + 1];
			r = bgrImg.data[3*line + 2];
			grayVal = (b * 117 + g * 601 + r * 306) >> 10;
			grayImg.data[line] = grayVal;
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
	if (th<0||th>255)
	{
		cout << "th fail." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (binaryImg.channels() != 1|| grayImg.channels() != 1)
	{
		cout << "image channel  wrong." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	if (width != binaryImg.cols || height != binaryImg.rows)
	{
		cout << "image channel  wrong." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int all = width*height;
	int pixlabel[256] = { 0 };
	for (int i = 100; i < 256; i++)
		pixlabel[i] = 255;

	for (int line = 0; line < all; line++)
	{
			int pixVal = grayImg.data[line];
			binaryImg.data[line] = pixlabel[pixVal];
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
	if (gradImg_x.channels() != 1 || grayImg.channels() != 1 || gradImg_y.channels() != 1)
	{
		cout << "image channel  wrong." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	if (width != gradImg_x.cols || height != gradImg_x.rows || width != gradImg_y.cols || height != gradImg_y.rows)
	{
		cout << "image   wrong." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int height_true=height - 1;
	int width_true =width - 1;
	if (height_true <= 1 || width_true <= 1)
	{
		cout << "image size fail." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int num_x, grad_x, grad_y, num_y;
	gradImg_x.setTo(0);
	gradImg_y.setTo(0);

	//计算x方向梯度图
	for (int col_j = 1; col_j<width_true; col_j++)
	{
		for (int row_i = 1; row_i<height_true; row_i++)
		{
			num_x = row_i * width + col_j;
			grad_x =
				grayImg.data[num_x - width + 1]
				+ (grayImg.data[num_x + 1] << 1)
				+ grayImg.data[num_x + width + 1]
				- grayImg.data[num_x - width - 1]
				- (grayImg.data[num_x - 1] << 1)
				- grayImg.data[num_x + width - 1];

			((float*)gradImg_x.data)[num_x] = grad_x;
		}
	}

	//计算y方向梯度图
	for (int row_i = 1; row_i < height_true; row_i++)
	{
		for (int col_j = 1; col_j < width_true; col_j++)
		{
			num_y = row_i * width + col_j;
			grad_y =
				-grayImg.data[num_y - width + 1]
				+ grayImg.data[num_y + width + 1]
				- (grayImg.data[num_y - width] << 1)
				+ (grayImg.data[num_y + width] << 1)
				- grayImg.data[num_y - width - 1]
				+ grayImg.data[num_y + width - 1];

			((float*)gradImg_y.data)[num_y] = grad_y;
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

float SqrtByCarmack(float number)
{
	int i;
	float x2, y;
	const float threehalfs = 1.5F;

	x2 = number * 0.5F;
	y = number;
	i = *(int *)&y;
	i = 0x5f375a86 - (i >> 1);
	y = *(float *)&i;
	y = y * (threehalfs - (x2 * y * y));
	y = y * (threehalfs - (x2 * y * y));
	y = y * (threehalfs - (x2 * y * y));
	return number*y;
}

int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y, Mat& angleImg, Mat& magImg)
{
	if (NULL == gradImg_x.data || NULL == gradImg_y.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (gradImg_x.channels() != 1 || angleImg.channels() != 1 || gradImg_y.channels() != 1 || magImg.channels() != 1)
	{
		cout << "image channel  wrong." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = gradImg_x.cols;
	int height = gradImg_x.rows;
	if (width != gradImg_y.cols || height != gradImg_y.rows || width != angleImg.cols || height != angleImg.rows||width!=magImg.cols||height!=magImg.rows)
	{
		cout << "image channel  wrong." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (height <=2 || width <=2)
	{
		cout << "fail." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	angleImg.setTo(0);
	magImg.setTo(0);
	//计算角度图
	for (int row_i = 1; row_i < height - 1; row_i++)
	{
		for (int col_j = 1; col_j < width - 1; col_j += 1)
		{
			int numangle=row_i * width + col_j;
			float grad_x = ((float*)gradImg_x.data)[numangle];
			float grad_y = ((float*)gradImg_y.data)[numangle];
			float xielv = grad_y / grad_x;
			float angle = xielv - xielv*xielv*xielv/3;
			angle=angle / 3.14 * 180;
			if (angle < 0)
				angle += 180;
			if(grad_y<0)
				angle += 180;
			//自己找办法优化三角函数速度，并且转化为角度制，规范化到0-360
			((float*)angleImg.data)[numangle] = angle;
		}
	}

	//计算模
	for (int row_i = 1; row_i < height - 1; row_i++)
	{
		for (int col_j = 1; col_j < width - 1; col_j += 1)
		{
			int nummag = row_i * width + col_j;
			float grad_x = ((float*)gradImg_x.data)[nummag];
			float grad_y = ((float*)gradImg_y.data)[nummag];
			float magval = grad_x*grad_x + grad_y*grad_y;
			float mag = SqrtByCarmack(magval);
			((float*)magImg.data)[nummag] = mag;
		}
	}
	return SUB_IMAGE_MATCH_OK;

}

//函数功能：对灰度图像计算直方图
//grayImg：灰度图，单通道
//hist：直方图
//hist_len：直方图的亮度等级，直方图数组的长度
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_CalcHist(Mat grayImg, int* hist, int hist_len)
{
	if (NULL == grayImg.data || NULL == hist)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (256 != hist_len)
	{
		cout << "hist_len is wrong." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	int all = width*height;
	//直方图清零
	for (int i = 0; i < hist_len; i++)
	{
		hist[i] = 0;
	}

	//计算直方图
	for (int i = 0; i < all; i++)
	{
		int pixVal = grayImg.data[i];
		hist[pixVal]++;

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
	if (grayImg.channels() != 1 || subImg.channels() != 1)
	{
		cout << "image channel  wrong." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int checkhang = height - sub_height;
	int checklie=width - sub_width;
	if (checkhang<0 || checklie<0)
		{
			cout << "subimage > grayimage,fail" << endl;
			return SUB_IMAGE_MATCH_FAIL;
		}
	int v_size = width*height;
	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg(height, width, CV_32FC1);
	//匹配误差初始化
	searchImg.setTo(FLT_MAX);

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i <=checkhang; i++)
	{
		for (int j = 0; j <=checklie; j++)
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
					int chazhi = bigImg_pix - template_pix;
					total_diff += ((chazhi ^ (chazhi >> 31)) - (chazhi >> 31));
				}
			}
			//存储当前像素位置的匹配误差
			((float *)searchImg.data)[i * width + j] = total_diff;
		}
	}
	int min_num = 0;
	for (int i = 1; i < v_size; i++)
	{
		if(((float *)searchImg.data)[i]<((float *) searchImg.data)[min_num])
			min_num = i;

	}
	*x = min_num / width;
	*y = min_num%width;
	return SUB_IMAGE_MATCH_OK;
}

// 函数功能：利用色彩进行子图匹配
//colorImg：彩色图，三通单
//subImg：模板子图，三通道
//x：最佳匹配子图左上角x坐标
//y：最佳匹配子图左上角y坐标
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int* x, int* y)
{
	if (NULL == colorImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (colorImg.channels() != 3 || subImg.channels() != 3)
	{
		cout << "image channel  wrong." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = colorImg.cols;
	int height = colorImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int checkhang = height - sub_height;
	int checklie = width - sub_width;
	if (checkhang<0 || checklie<0)
	{
		cout << "subimage > colorimage,fail" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int v_size = width*height;
	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg(height, width, CV_32FC1);
	//匹配误差初始化
	searchImg.setTo(FLT_MAX);

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i <=checkhang; i++)
	{
		for (int j = 0; j <= checklie; j++)
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
					int bignum = row_index * width + col_index;
					int tempnum=x * sub_width + y;
					int big_b = colorImg.data[3 * bignum + 0];
					int big_g = colorImg.data[3 * bignum + 1];
					int big_r = colorImg.data[3 * bignum + 2];
					int chazhi_b = subImg.data[3 * tempnum + 0]-big_b;
					int chazhi_g = subImg.data[3 * tempnum + 1]-big_g;
					int chazhi_r = subImg.data[3 * tempnum + 2]-big_r;

					total_diff += ((chazhi_b ^ (chazhi_b >> 31)) - (chazhi_b >> 31));
					total_diff += ((chazhi_g ^ (chazhi_g >> 31)) - (chazhi_g >> 31));
					total_diff += ((chazhi_r ^ (chazhi_r >> 31)) - (chazhi_r >> 31));
				}
			}
			//存储当前像素位置的匹配误差
			((float*)searchImg.data)[i * width + j] = total_diff;
		}
	}
	int min_num = 0;
	for (int i = 1; i < v_size; i++)
	{
		if (((float *)searchImg.data)[i]<((float *)searchImg.data)[min_num])
			min_num = i;
	}
	*x = min_num / width;
	*y = min_num%width;
	return SUB_IMAGE_MATCH_OK;
}


// 函数功能：利用亮度相关性进行子图匹配
//grayImg：灰度图，单通道
//subImg：模板子图，单通道
//x：最佳匹配子图左上角x坐标
//y：最佳匹配子图左上角y坐标
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

float FastInvSqrt(float x)
{
	int tmp = ((0x3f800000 << 1) + 0x3f800000
		- *(long*)&x) >> 1;
	float y = *(float*)&tmp;
	return y * (1.47f - 0.47f * x * y * y);
}

int ustc_SubImgMatch_corr(Mat grayImg,Mat subImg, int *x, int *y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (grayImg.channels() != 1 || subImg.channels() != 1)
	{
		cout << "image channel  wrong." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int corr_i, corr_j;
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int sub_h2 = height - sub_height;
	int sub_w2 = width - sub_width;

	if (sub_w2<0 || sub_h2<0)
	{
		cout << "size wrong" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	Mat searchImg(sub_h2, sub_w2, CV_32FC1);
	int best_x = 0, best_y = 0;
	int need = sub_height*sub_width;
	for (corr_i = 0; corr_i <sub_h2; corr_i++)
	 {
		for (corr_j = 0; corr_j <sub_w2; corr_j++)
		 {
		
			float xy = 0, x = 0, y = 0, xx = 0, yy = 0;
			for (int y1 = 0; y1 < sub_height; y1++)
			 {
				int row_index = corr_i + y1;
				int big = row_index * width;
				int sub = y1 * sub_width;
				for (int x1 = 0; x1 < sub_width; x1++)
				 {
					int col_index = corr_j + x1;
					int bigImg_pix = grayImg.data[big + col_index];
					int template_pix = subImg.data[sub + x1];
					xy += bigImg_pix*template_pix;
					x += bigImg_pix;
					y += template_pix;
					xx += bigImg_pix*bigImg_pix;
					yy += template_pix*template_pix;
				
				}
			}
	float fenzi = need*xy - x*y;
	float fenmu1 = FastInvSqrt(need*xx - x*x);
	float fenmu2 = FastInvSqrt(need*yy - y*y);
	float fenmu = fenmu1*fenmu2;
	float total_corr = fenzi / fenmu;
		 //存储当前像素位置的匹配误差
	((float*)searchImg.data)[corr_i *sub_w2 + corr_j] = total_corr;
		
		  }
	}
float maxcorr = ((float*)searchImg.data)[0];
for (int i = 0; i < sub_h2; i++)
	 {
	for (int j = 0; j < sub_w2; j++)
		 {
		float a = ((float*)searchImg.data)[i * sub_w2 + j];
		int abs_a = (((a - 1) ^ ((a - 1) >> 31)) - ((a - 1) >> 31));
		int abs_maxcorr = (((maxcorr - 1) ^ ((maxcorr - 1) >> 31)) - ((maxcorr - 1) >> 31));
		if (abs_a<abs_maxcorr) {
			maxcorr = a;
			best_x = j;
			best_y = i;
			
		}
		}
	}
*x = best_x;
*y = best_y;
return SUB_IMAGE_MATCH_FAIL;
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
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (grayImg.channels() != 1 || subImg.channels() != 1)
	{
		cout << "image channel  wrong." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	Mat gradImg_x(height, width, CV_32FC1);
	Mat gradImg_y(height, width, CV_32FC1);
	Mat angleImg(height, width, CV_32FC1);
	Mat magImg(height, width, CV_32FC1);

	Mat subgradImg_x(sub_height, sub_width, CV_32FC1);
	Mat subgradImg_y(sub_height, sub_width, CV_32FC1);
	Mat subangleImg(sub_height, sub_width, CV_32FC1);
	Mat submagImg(sub_height, sub_width, CV_32FC1);

	int flag1 = ustc_CalcGrad(grayImg, gradImg_x, gradImg_y);
	int flag2 = ustc_CalcAngleMag(gradImg_x, gradImg_y, angleImg, magImg);
	int subflag1 = ustc_CalcGrad(subImg, subgradImg_x, subgradImg_y);
	int subflag2 = ustc_CalcAngleMag(subgradImg_x, subgradImg_y, subangleImg, submagImg);
	if (flag1<0 || flag2<0 || subflag1<0 || subflag2<0)
	{
		cout << "ustc_CalcGrad or ustc_CalcAngleMag fail." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	/*cout <<angleImg << endl;
	cout << subangleImg << endl;
	*/
	int checkhang = height - sub_height;
	int checklie = width - sub_width;
	if (checkhang<0 || checklie<0)
	{
		cout << "subimage > grayimage,fail" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int v_size = width*height;
	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg(height, width, CV_32FC1);
	//匹配误差初始化
	searchImg.setTo(FLT_MAX);

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i <= checkhang; i++)
	{
		for (int j = 0; j <= checklie; j++)
		{
			int total_diff = 0;
			//遍历模板图上的每一个像素
			for (int x = 1; x < sub_height-1; x++)
			{
				for (int y = 1; y < sub_width-1; y++)
				{
					int row_index = i + x;
					int col_index = j + y;
					int bigImg_pix = angleImg.data[row_index * width + col_index];
					int template_pix = subangleImg.data[x * sub_width + y];
					//角度的换算
					int chazhi = bigImg_pix - template_pix;
					int chazhiabs= ((chazhi ^ (chazhi >> 31)) - (chazhi >> 31));
					if (chazhiabs > 180)
						total_diff += 360 - chazhiabs;
					else
						total_diff += chazhiabs;
				}
			}
			//存储当前像素位置的匹配误差
			((float *)searchImg.data)[i * width + j] = total_diff;
		}
	}
	int min_num = 0;
	for (int i = 1; i < v_size; i++)
	{
		if (((float *)searchImg.data)[i]<((float *)searchImg.data)[min_num])
			min_num = i;

	}
	*x = min_num / width;
	*y = min_num%width;
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
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (grayImg.channels() != 1 || subImg.channels() != 1)
	{
		cout << "image channel  wrong." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	Mat gradImg_x(height, width, CV_32FC1);
	Mat gradImg_y(height, width, CV_32FC1);
	Mat angleImg(height, width, CV_32FC1);
	Mat magImg(height, width, CV_32FC1);

	Mat subgradImg_x(height, width, CV_32FC1);
	Mat subgradImg_y(height, width, CV_32FC1);
	Mat subangleImg(height, width, CV_32FC1);
	Mat submagImg(height, width, CV_32FC1);

	int flag1 = ustc_CalcGrad(grayImg, gradImg_x, gradImg_y);
	int flag2 = ustc_CalcAngleMag(gradImg_x, gradImg_y, angleImg, magImg);
	int subflag1 = ustc_CalcGrad(subImg, subgradImg_x, subgradImg_y);
	int subflag2 = ustc_CalcAngleMag(subgradImg_x, subgradImg_y, subangleImg, submagImg);
	if (flag1<0 || flag2<0 || subflag1<0 || subflag2<0)
	{
		cout << "ustc_CalcGrad or ustc_CalcAngleMag fail." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int checkhang = height - sub_height;
	int checklie = width - sub_width;
	if (checkhang<0 || checklie<0)
	{
		cout << "subimage > grayimage,fail" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int v_size = width*height;
	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg(height, width, CV_32FC1);
	//匹配误差初始化
	searchImg.setTo(FLT_MAX);

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i <= checkhang; i++)
	{
		for (int j = 0; j <= checklie; j++)
		{
			int total_diff = 0;
			//遍历模板图上的每一个像素
			for (int x = 1; x < sub_height-1; x++)
			{
				for (int y = 1; y < sub_width-1; y++)
				{
					int row_index = i + x;
					int col_index = j + y;
					int bigImg_pix = magImg.data[row_index * width + col_index];
					int template_pix = submagImg.data[x * sub_width + y];
					int chazhi = bigImg_pix - template_pix;
					total_diff += ((chazhi ^ (chazhi >> 31)) - (chazhi >> 31));
				}
			}
			//存储当前像素位置的匹配误差
			((float *)searchImg.data)[i * width + j] = total_diff;
		}
	}
	int min_num = 0;
	for (int i = 1; i < v_size; i++)
	{
		if (((float *)searchImg.data)[i]<((float *)searchImg.data)[min_num])
			min_num = i;

	}
	*x = min_num / width;
	*y = min_num%width;
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
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int checkhang = height - sub_height;
	int checklie = width - sub_width;
	if (checkhang<0 || checklie<0)
	{
		cout << "subimage > colorimage,fail" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int v_size = width*height;
	int hist_len = 256;
	int* hist_sub = new int[hist_len];
	memset(hist_sub, 0, sizeof(int) * hist_len);
	int subflag = ustc_CalcHist(subImg, hist_sub, hist_len);
	if (subflag<0)
	{
		cout << "sub hist fail." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg(height, width, CV_32FC1);
	//匹配误差初始化
	searchImg.setTo(FLT_MAX);
	//遍历大图每一个像素，注意行列的起始、终止坐标
	int* hist_temp = new int[hist_len];
	memset(hist_temp, 0, sizeof(int) * hist_len);

	for (int i = 0; i < checkhang; i++)
	{
		for (int j = 0; j <checklie; j++)
		{
			//清零
			memset(hist_temp, 0, sizeof(int) * hist_len);

			//计算当前位置直方图
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
				int chazhi=hist_temp[ii] - hist_sub[ii];
				total_diff += ((chazhi ^ (chazhi >> 31)) - (chazhi >> 31));
			}
			//存储当前像素位置的匹配误差
			((float*)searchImg.data)[i * width + j] = total_diff;
		}
	}
	delete[] hist_temp;
	delete[] hist_sub;
	int min_num = 0;
	for (int i = 1; i < v_size; i++)
	{
		if (((float *)searchImg.data)[i]<((float *)searchImg.data)[min_num])
			min_num = i;
	}
	*x = min_num / width;
	*y = min_num%width;
	return SUB_IMAGE_MATCH_OK;
}
