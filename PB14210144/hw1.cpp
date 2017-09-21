#include "opencv.hpp"
using namespace cv;
#include <iostream>
using namespace std;
#include <time.h>

//#define IMG_SHOW
#define MY_OK 1
#define MY_FAIL -1

int ustc_ConvertBgr2Gray(Mat colorImg, Mat& grayImg)
{
	if (NULL == colorImg.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}
	int size = colorImg.rows * colorImg.cols;
	static const uint k1 = 0.114f * 16777216 + 0.5, k2 = 0.587f * 16777216 + 0.5, k3 = 0.229f * 16777216 + 0.5;
	/*if (size % 4 == 0)
		for (int i = size - 1, j = 3 * size - 1; i >= 0; i--, j -= 3){
			grayImg.data[i] = ((uint)colorImg.data[j - 2] * k1 + (uint)colorImg.data[j - 1] * k2 + (uint)colorImg.data[j - 0] * k3) >> 24;
			i--, j -= 3;
			grayImg.data[i] = ((uint)colorImg.data[j - 2] * k1 + (uint)colorImg.data[j - 1] * k2 + (uint)colorImg.data[j - 0] * k3) >> 24;
			i--, j -= 3;
			grayImg.data[i] = ((uint)colorImg.data[j - 2] * k1 + (uint)colorImg.data[j - 1] * k2 + (uint)colorImg.data[j - 0] * k3) >> 24;
			i--, j -= 3;
			grayImg.data[i] = ((uint)colorImg.data[j - 2] * k1 + (uint)colorImg.data[j - 1] * k2 + (uint)colorImg.data[j - 0] * k3) >> 24;
		}
	else if(size % 2 == 0)
		for (int i = size - 1, j = 3 * size - 1; i >= 0; i--, j -= 3){
			grayImg.data[i] = ((uint)colorImg.data[j - 2] * k1 + (uint)colorImg.data[j - 1] * k2 + (uint)colorImg.data[j - 0] * k3) >> 24;
			i--, j -= 3;
			grayImg.data[i] = ((uint)colorImg.data[j - 2] * k1 + (uint)colorImg.data[j - 1] * k2 + (uint)colorImg.data[j - 0] * k3) >> 24;
		}
	else
		for (int i = size - 1, j = 3 * size - 1; i >= 0; i--, j -= 3)
			grayImg.data[i] = ((uint)colorImg.data[j - 2] * k1 + (uint)colorImg.data[j - 1] * k2 + (uint)colorImg.data[j - 0] * k3) >> 24;*/
	uchar * buffer = colorImg.data;
	static const __m128i factor = _mm_set_epi16(0x0, 0x0, (ushort)(0.229f * 256), (ushort)(0.587f * 256), (ushort)(0.114f * 256), (ushort)(0.229f * 256), (ushort)(0.587f * 256), (ushort)(0.114f * 256)), zeros = _mm_setzero_si128();
	__m128i data, red, green, pixel;
	uchar temp[16];
	for (int i = 3; i < size; i += 4, buffer += 12){
		data = _mm_loadu_si128((__m128i *) buffer);
		pixel = _mm_unpacklo_epi8(data, zeros);
		pixel = _mm_mullo_epi16(pixel, factor);
		green = _mm_srli_si128(pixel, 2);
		red = _mm_srli_si128(pixel, 4);
		pixel = _mm_add_epi16(pixel, green);
		pixel = _mm_add_epi16(pixel, red);

		

		_mm_storeu_si128((__m128i *)temp, pixel);
		grayImg.data[i - 3] = temp[1];
		grayImg.data[i - 2] = temp[7];

		data = _mm_slli_si128(data, 2);
		pixel = _mm_unpackhi_epi8(data, zeros);
		pixel = _mm_mullo_epi16(pixel, factor);
		green = _mm_srli_si128(pixel, 2);
		red = _mm_srli_si128(pixel, 4);
		pixel = _mm_add_epi16(pixel, green);
		pixel = _mm_add_epi16(pixel, red);

		_mm_storeu_si128((__m128i *)temp, pixel);
		grayImg.data[i - 1] = temp[1];
		grayImg.data[i] = temp[7];
	}
	for (int i = size - (size % 4), j = 3 * i; i < size; i++, j += 3)
		grayImg.data[i] = ((uint)colorImg.data[j] * k1 + (uint)colorImg.data[j + 1] * k2 + (uint)colorImg.data[j + 2] * k3) >> 24;


#ifdef IMG_SHOW
	namedWindow("grayImg", 0);
	imshow("grayImg", grayImg);
	waitKey();
#endif
}


int ustc_CalcGrad(Mat grayImg, Mat& gradImg_x, Mat& gradImg_y)
{
	if (NULL == grayImg.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}

	ushort width = grayImg.cols;
	ushort height = grayImg.rows;

	gradImg_x.setTo(0);
	gradImg_y.setTo(0);

	/*int col_j, Img_j, Img_i;
	for (Img_i = (height - 2) * width; Img_i > 0; Img_i -= width)
		for (col_j = width - 2, Img_j = Img_i + 1; col_j > 0; col_j--, Img_j++)
		{

				
				((float*)gradImg_y.data)[Img_j] = grayImg.data[Img_j + width - 1]
					+ grayImg.data[Img_j + width] << 1
					+ grayImg.data[Img_j + width + 1]
					- grayImg.data[Img_j - width - 1]
					- grayImg.data[Img_j - width] << 1
					- grayImg.data[Img_j - width + 1];
				((float*)gradImg_x.data)[Img_j] = grayImg.data[Img_j - width + 1]
					+ grayImg.data[Img_j + 1] << 1
					+ grayImg.data[Img_j + width + 1]
					- grayImg.data[Img_j - width - 1]
					- grayImg.data[Img_j - 1] << 1
					- grayImg.data[Img_j + width - 1];
		}*/

	int col_j, Img_j, Img_i, k, remain = ((width - 2) % 16);
	__m128i m1, m2, m3, m4, m6, m7, m8, m9, grad_x, grad_y, temp;
	static const __m128i zeros = _mm_setzero_si128();
	for (Img_i = (height - 2) * width; Img_i > 0; Img_i -= width){
		for (col_j = width - 2 - 16, Img_j = Img_i + 1; col_j > 0; col_j -= 16, Img_j += 16){

			m1 = _mm_loadu_si128((__m128i *)(grayImg.data + Img_j - width - 1));
			m2 = _mm_loadu_si128((__m128i *)(grayImg.data + Img_j - width));
			m3 = _mm_loadu_si128((__m128i *)(grayImg.data + Img_j - width + 1));
			m4 = _mm_loadu_si128((__m128i *)(grayImg.data + Img_j - 1));
			m6 = _mm_loadu_si128((__m128i *)(grayImg.data + Img_j + 1));
			m7 = _mm_loadu_si128((__m128i *)(grayImg.data + Img_j + width - 1));
			m8 = _mm_loadu_si128((__m128i *)(grayImg.data + Img_j + width));
			m9 = _mm_loadu_si128((__m128i *)(grayImg.data + Img_j + width + 1));

			grad_x = _mm_unpacklo_epi8(m3, zeros);

			temp = _mm_unpacklo_epi8(m6, zeros);
			temp = _mm_slli_epi16(temp, 1);

			grad_x = _mm_add_epi16(grad_x, temp);

			temp = _mm_unpacklo_epi8(m9, zeros);
			grad_x = _mm_add_epi16(grad_x, temp);

			temp = _mm_unpacklo_epi8(m1, zeros);
			grad_x = _mm_sub_epi16(grad_x, temp);

			temp = _mm_unpacklo_epi8(m4, zeros);
			temp = _mm_slli_epi16(temp, 1);
			grad_x = _mm_sub_epi16(grad_x, temp);

			temp = _mm_unpacklo_epi8(m7, zeros);
			grad_x = _mm_sub_epi16(grad_x, temp);


			for (k = 7; k >= 0; k--)
				((float*)gradImg_x.data)[Img_j + k] = (float)grad_x.m128i_i16[k];

			grad_y = _mm_unpacklo_epi8(m7, zeros);

			temp = _mm_unpacklo_epi8(m8, zeros);
			temp = _mm_slli_epi16(temp, 1);
			grad_y = _mm_add_epi16(grad_y, temp);

			temp = _mm_unpacklo_epi8(m9, zeros);
			grad_y = _mm_add_epi16(grad_y, temp);

			temp = _mm_unpacklo_epi8(m1, zeros);
			grad_y = _mm_sub_epi16(grad_y, temp);

			temp = _mm_unpacklo_epi8(m2, zeros);
			temp = _mm_slli_epi16(temp, 1);
			grad_y = _mm_sub_epi16(grad_y, temp);

			temp = _mm_unpacklo_epi8(m3, zeros);
			grad_y = _mm_sub_epi16(grad_y, temp);

			for (k = 7; k >= 0; k--)
				((float*)gradImg_y.data)[Img_j + k] = (float)grad_y.m128i_i16[k];





			grad_x = _mm_unpackhi_epi8(m3, zeros);

			temp = _mm_unpackhi_epi8(m6, zeros);
			temp = _mm_slli_epi16(temp, 1);
			grad_x = _mm_add_epi16(grad_x, temp);

			temp = _mm_unpackhi_epi8(m9, zeros);
			grad_x = _mm_add_epi16(grad_x, temp);

			temp = _mm_unpackhi_epi8(m1, zeros);
			grad_x = _mm_sub_epi16(grad_x, temp);

			temp = _mm_unpackhi_epi8(m4, zeros);
			temp = _mm_slli_epi16(temp, 1);
			grad_x = _mm_sub_epi16(grad_x, temp);

			temp = _mm_unpackhi_epi8(m7, zeros);
			grad_x = _mm_sub_epi16(grad_x, temp);

			for (k = 8; k < 16; k++)
				((float*)gradImg_x.data)[Img_j + k] = (float)grad_x.m128i_i16[k - 8];

			grad_y = _mm_unpackhi_epi8(m7, zeros);

			temp = _mm_unpackhi_epi8(m8, zeros);
			temp = _mm_slli_epi16(temp, 1);
			grad_y = _mm_add_epi16(grad_y, temp);

			temp = _mm_unpackhi_epi8(m9, zeros);
			grad_y = _mm_add_epi16(grad_y, temp);

			temp = _mm_unpackhi_epi8(m1, zeros);
			grad_y = _mm_sub_epi16(grad_y, temp);

			temp = _mm_unpackhi_epi8(m2, zeros);
			temp = _mm_slli_epi16(temp, 1);
			grad_y = _mm_sub_epi16(grad_y, temp);

			temp = _mm_unpackhi_epi8(m3, zeros);
			grad_y = _mm_sub_epi16(grad_y, temp);

			for (k = 8; k < 16; k++)
				((float*)gradImg_y.data)[Img_j + k] = (float)grad_y.m128i_i16[k - 8];





		}

		for (col_j = remain, Img_j = Img_i + width - remain; col_j > 0; col_j--, Img_j++){
			((float*)gradImg_y.data)[Img_j] = grayImg.data[Img_j + width - 1]
				+ grayImg.data[Img_j + width] << 1
				+ grayImg.data[Img_j + width + 1]
				- grayImg.data[Img_j - width - 1]
				- grayImg.data[Img_j - width] << 1
				- grayImg.data[Img_j - width + 1];
			((float*)gradImg_x.data)[Img_j] = grayImg.data[Img_j - width + 1]
				+ grayImg.data[Img_j + 1] << 1
				+ grayImg.data[Img_j + width + 1]
				- grayImg.data[Img_j - width - 1]
				- grayImg.data[Img_j - 1] << 1
				- grayImg.data[Img_j + width - 1];
		}

	}
	

#ifdef IMG_SHOW
	Mat gradImg_x_8U(grayImg.rows, grayImg.cols, CV_8UC1);
	//为了方便观察，直接取绝对值
	for (int row_i = 0; row_i < height; row_i++)
	{
		for (int col_j = 0; col_j < width; col_j += 1)
		{
			float val = ((float*)gradImg_x.data)[row_i * width + col_j];
			gradImg_x_8U.data[row_i * width + col_j] = abs(val);
		}
	}

	namedWindow("gradImg_x_8U", WINDOW_AUTOSIZE);
	imshow("gradImg_x_8U", gradImg_x_8U);
	waitKey();
#endif
}

static const float atan2_p1 = 0.9997878412794807f*(float)(180 / CV_PI);
static const float atan2_p3 = -0.3258083974640975f*(float)(180 / CV_PI);
static const float atan2_p5 = 0.1555786518463281f*(float)(180 / CV_PI);
static const float atan2_p7 = -0.04432655554792128f*(float)(180 / CV_PI);

inline float my_fastAtan2(const float & y, const float & x)
{
	float dy = y, dx = x;
	float ax = std::abs(x), ay = std::abs(y);
	float a, c, c2;
	if (ax >= ay)
	{
		c = ay / (ax + (float)DBL_EPSILON);
		c2 = c*c;
		a = (((atan2_p7*c2 + atan2_p5)*c2 + atan2_p3)*c2 + atan2_p1)*c;
	}
	else
	{
		c = ax / (ay + (float)DBL_EPSILON);
		c2 = c*c;
		a = 90.f - (((atan2_p7*c2 + atan2_p5)*c2 + atan2_p3)*c2 + atan2_p1)*c;
	}
	if (x < 0)
		a = 180.f - a;
	if (y < 0)
		a = 360.f - a;


	return a;
}

inline float SqrtBySQRTSS(const float & a)
{
	float b = a;
	__m128 in = _mm_load_ss(&b);
	__m128 out = _mm_sqrt_ss(in);
	_mm_store_ss(&b, out);

	return b;
}

int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y, Mat & angleImg, Mat & magImg)
{
	if (NULL == gradImg_x.data || NULL == gradImg_y.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}

	int width = gradImg_x.cols;
	int height = gradImg_x.rows;

	//计算角度图
	int size = gradImg_x.cols * gradImg_x.rows;
	/*float grad_x, grad_y, angle;
	for (int i = size - 1; i >= 0; i--){
		grad_x = ((float*)gradImg_x.data)[i];
		grad_y = ((float*)gradImg_y.data)[i];
		angle = my_fastAtan2(grad_y, grad_x);
		((float*)angleImg.data)[i] = angle;
		((float*)magImg.data)[i] += SqrtByRSQRTSS(grad_x * grad_x + grad_y * grad_y);
	};*/
	static const uint mask = 0x7fffffff; 
	static const __m128 k1 = _mm_set_ps1(atan2_p1), k2 = _mm_set_ps1(atan2_p3), k3 = _mm_set_ps1(atan2_p5), k4 = _mm_set_ps1(atan2_p7), DE = _mm_set_ps1((float)DBL_EPSILON), a1 = _mm_set_ps1(90.0), a2 = _mm_set_ps1(180.0), a3 = _mm_set_ps1(360.0), zeros = _mm_set_ps1(0.0);
	static const __m128 masks = _mm_setr_ps(*(float *)(&mask), *(float *)(&mask), *(float *)(&mask), *(float *)(&mask));
	__m128 grad_x, grad_y, max, min, ax, ay, a, c, c2, flag, temp;
	
	for (int i = size - 1 - 4, j = 0; i >= 0; i -= 4, j += 4){
		grad_x = _mm_loadu_ps((float*)gradImg_x.data + j);
		grad_y = _mm_loadu_ps((float*)gradImg_y.data + j);
		ax = _mm_and_ps(grad_x, masks);
		ay = _mm_and_ps(grad_y, masks);
		max = _mm_max_ps(ax, ay);
		min = _mm_min_ps(ax, ay);
		c = _mm_div_ps(min, _mm_add_ps(max, DE));
		c2 = _mm_mul_ps(c, c);
		a = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(k4, c2), k3), c2), k2), c2), k1), c);

		flag = _mm_cmplt_ps(ax, ay);
		temp = _mm_sub_ps(a1, a);
		temp = _mm_and_ps(flag, temp);
		flag = _mm_cmpge_ps(ax, ay);
		a = _mm_and_ps(a, flag);
		a = _mm_or_ps(a, temp);

		flag = _mm_cmplt_ps(grad_x, zeros);
		temp = _mm_sub_ps(a2, a);
		temp = _mm_and_ps(flag, temp);
		flag = _mm_cmpge_ps(grad_x, zeros);
		a = _mm_and_ps(a, flag);
		a = _mm_or_ps(a, temp);

		flag = _mm_cmplt_ps(grad_y, zeros);
		temp = _mm_sub_ps(a3, a);
		temp = _mm_and_ps(flag, temp);
		flag = _mm_cmpge_ps(grad_y, zeros);
		a = _mm_and_ps(a, flag);
		a = _mm_or_ps(a, temp);

		_mm_storeu_ps(((float*)angleImg.data) + j, a);
		_mm_storeu_ps(((float*)magImg.data) + j, _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(grad_x, grad_x), _mm_mul_ps(grad_y, grad_y))));




	};
	float gx, gy;

	for (int i = size - (size % 4); i < size; i++){
		gx = ((float*)gradImg_x.data)[i];
		gy = ((float*)gradImg_y.data)[i];
		((float*)angleImg.data)[i] = my_fastAtan2(gy, gx);
		((float*)magImg.data)[i] += SqrtBySQRTSS(gx * gx + gy * gy);
	};



#ifdef IMG_SHOW
	Mat angleImg_8U(height, width, CV_8UC1);
	Mat magImg_8U(height, width, CV_8UC1);
	//为了方便观察，进行些许变化
	for (int row_i = 0; row_i < height; row_i++)
	{
		for (int col_j = 0; col_j < width; col_j += 1)
		{
			float angle = ((float*)angleImg.data)[row_i * width + col_j];
			angle *= 180 / CV_PI;
			angle += 180;
			//为了能在8U上显示，缩小到0-180之间
			angle /= 2;
			angleImg_8U.data[row_i * width + col_j] = angle;
			magImg_8U.data[row_i * width + col_j] = ((float*)magImg.data)[row_i * width + col_j] / 10;
		}
	}

	namedWindow("gradImg_x_8U", 0);
	imshow("gradImg_x_8U", magImg_8U);
	waitKey();
#endif
}

int ustc_Threshold(Mat grayImg)
{
	if (NULL == grayImg.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}

	Mat binaryImg(grayImg.rows, grayImg.cols, CV_8UC1);

	int th = 100;
	if (th > 255 || th < 0){
		cout << "error!" << endl;
		return MY_FAIL;
	}
	int size = grayImg.rows * grayImg.cols;
	int i, j, k;
	uchar * buffer = grayImg.data;
	__m128i data, pixel, mask, out, temp;
	static const uchar masks1[16] = { 0, 2, 4, 6, 8, 10, 12, 14, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80 }, masks2[16] = { 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0, 2, 4, 6, 8, 10, 12, 14 };
	static const __m128i zeros = _mm_setzero_si128();
	//static const __m128i factorl = _mm_set_epi16((ushort)0xffff, (ushort)0xffff, (ushort)0xffff, (ushort)0xffff, 0x0, 0x0, 0x0, 0x0), factorh = _mm_set_epi16(0x0, 0x0, 0x0, 0x0, (ushort)0xffff, (ushort)0xffff, (ushort)0xffff, (ushort)0xffff);
	static const __m128i mask1 = _mm_loadu_si128((__m128i *) masks1), mask2 = _mm_loadu_si128((__m128i *) masks2);
	for (i = size - 16, j = 0; i > 0; i -= 16, j += 16){
		data = _mm_loadu_si128((__m128i *) (buffer + j));
		pixel = _mm_unpacklo_epi8(data, zeros);
		mask = _mm_set1_epi16(th);
		pixel = _mm_cmpgt_epi16(pixel, mask);
		/*for(k = 7; k >= 0; k--)
		binaryImg.data[j + k] = pixel.m128i_i8[k << 1];*/
		/*pixel = _mm_shufflehi_epi16(pixel, _MM_SHUFFLE(3, 1, 2, 0));
		pixel = _mm_shufflelo_epi16(pixel, _MM_SHUFFLE(3, 1, 2, 0));
		pixel = _mm_shuffle_epi32(pixel, _MM_SHUFFLE(3, 1, 2, 0));
		temp = _mm_srli_si128(pixel, 8);
		out = _mm_and_si128(_mm_unpacklo_epi8(pixel, temp), factorh);*/
		out = _mm_shuffle_epi8(pixel, mask1);

		pixel = _mm_unpackhi_epi8(data, zeros);
		mask = _mm_set1_epi16(th);
		pixel = _mm_cmpgt_epi16(pixel, mask);
		/*for (k = 8; k < 16; k++)
		binaryImg.data[j + k] = pixel.m128i_i8[(k - 8) << 1];*/
		/*pixel = _mm_shufflehi_epi16(pixel, _MM_SHUFFLE(3, 1, 2, 0));
		pixel = _mm_shufflelo_epi16(pixel, _MM_SHUFFLE(3, 1, 2, 0));
		pixel = _mm_shuffle_epi32(pixel, _MM_SHUFFLE(3, 1, 2, 0));
		temp = _mm_srli_si128(pixel, 8);
		temp = _mm_unpacklo_epi8(pixel, temp);
		temp = _mm_and_si128(_mm_slli_si128(pixel, 8), factorl);
		out = _mm_or_si128(out, temp);*/
		temp = _mm_shuffle_epi8(pixel, mask2);
		out = _mm_or_si128(out, temp);
		_mm_storeu_si128((__m128i *)(binaryImg.data + j), out);
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

	int size = grayImg.rows * grayImg.cols, i;
	//直方图清零
	memset(hist, 0, sizeof(int) * hist_len);
	//计算直方图
	
	for (i = size - 1; i >= 0; i--)
		hist[grayImg.data[i]]++;

	

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
	if (sub_height > height || sub_width > width){
		cout << "error!" << endl;
		return MY_FAIL;
	}
	int sub_size = sub_height * sub_width;
	int h_range = height - sub_height;
	int w_range = width - sub_width;
	uint min_diff = 0 - 1, min_x = 0, min_y = 0, total_diff;
	int i, j, m, n, img_i, sub_i, a, b;

	/*//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			int total_diff = 0;
			//遍历模板图上的每一个像素
			for (int x = 0; x < sub_height; x++)
			{
				for (int y = 0; y < sub_width; y++)
				{
					//大图上的像素位置
					int row_index = i + y;
					int col_index = j + x;
					int bigImg_pix = grayImg.data[row_index * width + col_index];
					//模板图上的像素
					int template_pix = subImg.data[y * sub_width + x];

					total_diff += abs(bigImg_pix - template_pix);
				}
			}
			//存储当前像素位置的匹配误差
			((float*)searchImg.data)[i * width + j] = total_diff;
		}
	}*/

	__m128i sub_data, img_data, temp, out;

	for (i = 0; i < h_range; i++)
	{
		for (j = 0; j < w_range; j++)
		{
			total_diff = 0;
			out = _mm_setzero_si128();
			//遍历模板图上的每一个像素
			for (m = 0; m < sub_height; m++)
			{
				for (n = sub_width - 16, sub_i = m * sub_width, img_i = (i + m) * width + j; n >= 0; n -= 16, sub_i += 16, img_i += 16)
				{
					sub_data = _mm_loadu_si128((__m128i *) (subImg.data + sub_i));
					img_data = _mm_loadu_si128((__m128i *) (grayImg.data + img_i));
					temp = _mm_sad_epu8(sub_data, img_data);
					out = _mm_add_epi32(temp, out);
				}
				for (n += 16; n > 0; n--, sub_i++, img_i++)
					total_diff += std::abs(grayImg.data[img_i] - subImg.data[sub_i]);
			}
			/*for (sub_i = 0, n = 16, m = 0, img_i = i * width + j; sub_i < sub_size - 16; sub_i += 16, n += 16, img_i += 16)
			{
				sub_data = _mm_loadu_si128((__m128i *) (subImg.data + sub_i));
				
				if (n >= sub_width){
					n -= sub_width;
					memcpy(masks, grayImg.data + img_i, 16 - n - 1);
					img_i += width - sub_width;
					memcpy(masks + 16 - n - 1, grayImg.data + img_i, n + 1);
					img_data = _mm_loadu_si128((__m128i *) masks);
					m++;
				}else
					img_data = _mm_loadu_si128((__m128i *) (grayImg.data + img_i));
				temp = _mm_sad_epu8(sub_data, img_data);
				out = _mm_add_epi32(temp, out);
			}
			for (; sub_i < sub_size; sub_i++, img_i++)
				total_diff += std::abs(grayImg.data[img_i] - subImg.data[sub_i]);*/

			total_diff += out.m128i_i32[0] + out.m128i_i32[2];
			
			if (total_diff < min_diff){
				min_diff = total_diff;
				min_y = i;
				min_x = j;
			}
		}
	}
	*y = min_y;
	*x = min_x;
}

int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int* x, int* y)
{
	if (NULL == colorImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}

	int width = colorImg.cols * 3;
	int height = colorImg.rows;
	int sub_width = subImg.cols * 3;
	int sub_height = subImg.rows;
	if (sub_height > height || sub_width > width){
		cout << "error!" << endl;
		return MY_FAIL;
	}
	int sub_size = sub_height * sub_width;
	int h_range = height - sub_height;
	int w_range = width - sub_width;
	uint min_diff = 0 - 1, min_x = 0, min_y = 0, total_diff;
	int i, j, m, n, img_i, sub_i, a, b;

	

	__m128i sub_data, img_data, temp, out;

	for (i = 0; i < h_range; i++)
	{
		for (j = 0; j < w_range; j += 3)
		{
			total_diff = 0;
			out = _mm_setzero_si128();
			//遍历模板图上的每一个像素
			for (m = 0; m < sub_height; m++)
			{
				for (n = sub_width - 16, sub_i = m * sub_width, img_i = (i + m) * width + j; n >= 0; n -= 16, sub_i += 16, img_i += 16)
				{
					sub_data = _mm_loadu_si128((__m128i *) (subImg.data + sub_i));
					img_data = _mm_loadu_si128((__m128i *) (colorImg.data + img_i));
					temp = _mm_sad_epu8(sub_data, img_data);
					out = _mm_add_epi32(temp, out);
				}
				for (n += 16; n > 0; n--, sub_i++, img_i++)
					total_diff += std::abs(colorImg.data[img_i] - subImg.data[sub_i]);
			}
			

			total_diff += out.m128i_i32[0] + out.m128i_i32[2];

			if (total_diff < min_diff){
				min_diff = total_diff;
				min_y = i;
				min_x = j / 3;
			}
		}
	}
	*y = min_y;
	*x = min_x;
}


inline float SqrtByRSQRTSS(const float & a)
{
	float b = a;
	__m128 in = _mm_load_ss(&b);
	__m128 out = _mm_rsqrt_ss(in);
	_mm_store_ss(&b, out);

	return b;
}
int ustc_SubImgMatch_corr(Mat grayImg, Mat subImg, int* x, int* y){
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	if (sub_height > height || sub_width > width){
		cout << "error!" << endl;
		return MY_FAIL;
	}
	int sub_size = sub_height * sub_width;
	int h_range = height - sub_height;
	int w_range = width - sub_width;
	uint min_x = 0, min_y = 0;
	float min_diff = -1;
	int i, j, m, n, img_i, sub_i, a, b;

	float total_diff, st, s2, t2, s, t;


	__m128 sub_data, img_data, ST, S2, T2;

	for (i = 0; i < h_range; i++)
	{
		for (j = 0; j < w_range; j++)
		{
			st = 0, s2 = 0, t2 = 0;
			ST = _mm_setzero_ps(), S2 = _mm_setzero_ps(), T2 = _mm_setzero_ps();
			//遍历模板图上的每一个像素
			for (m = 0; m < sub_height; m++)
			{
				for (n = sub_width - 4, sub_i = m * sub_width, img_i = (i + m) * width + j; n >= 0; n -= 4, sub_i += 4, img_i += 4)
				{
					sub_data = _mm_set_ps((float)subImg.data[sub_i], (float)subImg.data[sub_i + 1], (float)subImg.data[sub_i + 2], (float)subImg.data[sub_i + 3]);
					img_data = _mm_set_ps((float)grayImg.data[img_i], (float)grayImg.data[img_i + 1], (float)grayImg.data[img_i + 2], (float)grayImg.data[img_i + 3]);
					ST = _mm_add_ps(ST, _mm_mul_ps(sub_data, img_data));
					S2 = _mm_add_ps(S2, _mm_mul_ps(sub_data, sub_data));
					T2 = _mm_add_ps(T2, _mm_mul_ps(img_data, img_data));
					
				}
				for (n += 4; n > 0; n--, sub_i++, img_i++){
					s = (float)subImg.data[sub_i];
					t = (float)grayImg.data[img_i];
					st += s * t;
					s2 += s * s;
					t2 += t * t;
				}
			}
			st += ST.m128_f32[0] + ST.m128_f32[1] + ST.m128_f32[2] + ST.m128_f32[3];
			s2 += S2.m128_f32[0] + S2.m128_f32[1] + S2.m128_f32[2] + S2.m128_f32[3];
			t2 += T2.m128_f32[0] + T2.m128_f32[1] + T2.m128_f32[2] + T2.m128_f32[3];

			total_diff = st * SqrtByRSQRTSS(s2) * SqrtByRSQRTSS(t2);
			

			if (total_diff > min_diff){
				min_diff = total_diff;
				min_y = i;
				min_x = j;
			}
		}
	}
	*y = min_y;
	*x = min_x;
	cout << min_x << "  " << min_y << endl;
};

int CalcAngle(const Mat & gradImg_x, const Mat & gradImg_y, Mat & angleImg)
{
	if (NULL == gradImg_x.data || NULL == gradImg_y.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}

	int width = gradImg_x.cols;
	int height = gradImg_x.rows;

	//计算角度图
	int size = gradImg_x.cols * gradImg_x.rows;
	
	static const uint mask = 0x7fffffff;
	static const __m128 k1 = _mm_set_ps1(atan2_p1), k2 = _mm_set_ps1(atan2_p3), k3 = _mm_set_ps1(atan2_p5), k4 = _mm_set_ps1(atan2_p7), DE = _mm_set_ps1((float)DBL_EPSILON), a1 = _mm_set_ps1(90.0), a2 = _mm_set_ps1(180.0), a3 = _mm_set_ps1(360.0), zeros = _mm_set_ps1(0.0);
	static const __m128 masks = _mm_setr_ps(*(float *)(&mask), *(float *)(&mask), *(float *)(&mask), *(float *)(&mask));
	__m128 grad_x, grad_y, max, min, ax, ay, a, c, c2, flag, temp;

	for (int i = size - 1 - 4, j = 0; i >= 0; i -= 4, j += 4){
		grad_x = _mm_loadu_ps((float*)gradImg_x.data + j);
		grad_y = _mm_loadu_ps((float*)gradImg_y.data + j);
		ax = _mm_and_ps(grad_x, masks);
		ay = _mm_and_ps(grad_y, masks);
		max = _mm_max_ps(ax, ay);
		min = _mm_min_ps(ax, ay);
		c = _mm_div_ps(min, _mm_add_ps(max, DE));
		c2 = _mm_mul_ps(c, c);
		a = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(k4, c2), k3), c2), k2), c2), k1), c);

		flag = _mm_cmplt_ps(ax, ay);
		temp = _mm_sub_ps(a1, a);
		temp = _mm_and_ps(flag, temp);
		flag = _mm_cmpge_ps(ax, ay);
		a = _mm_and_ps(a, flag);
		a = _mm_or_ps(a, temp);

		flag = _mm_cmplt_ps(grad_x, zeros);
		temp = _mm_sub_ps(a2, a);
		temp = _mm_and_ps(flag, temp);
		flag = _mm_cmpge_ps(grad_x, zeros);
		a = _mm_and_ps(a, flag);
		a = _mm_or_ps(a, temp);

		flag = _mm_cmplt_ps(grad_y, zeros);
		temp = _mm_sub_ps(a3, a);
		temp = _mm_and_ps(flag, temp);
		flag = _mm_cmpge_ps(grad_y, zeros);
		a = _mm_and_ps(a, flag);
		a = _mm_or_ps(a, temp);

		_mm_storeu_ps(((float*)angleImg.data) + j, a);




	};
	float gx, gy;

	for (int i = size - (size % 4); i < size; i++){
		gx = ((float*)gradImg_x.data)[i];
		gy = ((float*)gradImg_y.data)[i];
		((float*)angleImg.data)[i] = my_fastAtan2(gy, gx);
	};
}
int ustc_SubImgMatch_angle(Mat grayImg, Mat subImg, int* x, int* y){
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	if (sub_height > height || sub_width > width){
		cout << "error!" << endl;
		return MY_FAIL;
	}
	int sub_size = sub_height * sub_width;
	int h_range = height - sub_height;
	int w_range = width - sub_width;
	uint min_x = 0, min_y = 0;
	
	int i, j, m, n, img_i, sub_i, a, b;
	
	Mat grad_x(grayImg.rows, grayImg.cols, CV_32F);
	Mat grad_y(grayImg.rows, grayImg.cols, CV_32F);
	Mat angelImg(grayImg.rows, grayImg.cols, CV_32F);
	ustc_CalcGrad(grayImg, grad_x, grad_y);
	CalcAngle(grad_x, grad_y, angelImg);

	Mat sgrad_x(subImg.rows, subImg.cols, CV_32F);
	Mat sgrad_y(subImg.rows, subImg.cols, CV_32F);
	Mat sangelImg(subImg.rows, subImg.cols, CV_32F);
	ustc_CalcGrad(subImg, sgrad_x, sgrad_y);
	CalcAngle(sgrad_x, sgrad_y, sangelImg);

	static const uint mask = 0x7fffffff;
	static const __m128 masks = _mm_setr_ps(*(float *)(&mask), *(float *)(&mask), *(float *)(&mask), *(float *)(&mask));

	__m128 sub_data, img_data, temp, out;
	float total_diff, min_diff = 9999999999999;

	for (i = 0; i < h_range; i++)
	{
		for (j = 0; j < w_range; j++)
		{
			total_diff = 0;
			out = _mm_setzero_ps();
			//遍历模板图上的每一个像素
			for (m = 1; m < sub_height - 1; m++)
			{
				for (n = sub_width - 4 - 2, sub_i = m * sub_width + 1, img_i = (i + m) * width + j + 1; n >= 0; n -= 4, sub_i += 4, img_i += 4)
				{
					sub_data = _mm_loadu_ps(((float *)sangelImg.data) + sub_i);
					img_data = _mm_loadu_ps(((float *)angelImg.data) + img_i);
					temp = _mm_sub_ps(sub_data, img_data);
					temp = _mm_and_ps(temp, masks);
					out = _mm_add_ps(out, temp);
					
				}
				for (n += 4; n > 0; n--, sub_i++, img_i++)
					total_diff += std::abs(((float *)sangelImg.data)[sub_i] - ((float *)angelImg.data)[img_i]);
			}

			total_diff += out.m128_f32[0] + out.m128_f32[1] + out.m128_f32[2] + out.m128_f32[3];
			if (total_diff < min_diff){
				min_diff = total_diff;
				min_y = i;
				min_x = j;
			}
		}
	}
	*y = min_y;
	*x = min_x;

	cout << *y << " " << *x << endl;
};

int CalcMag(const Mat & gradImg_x, const Mat & gradImg_y, Mat & magImg)
{
	if (NULL == gradImg_x.data || NULL == gradImg_y.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}

	int width = gradImg_x.cols;
	int height = gradImg_x.rows;

	//计算角度图
	int size = gradImg_x.cols * gradImg_x.rows;

	static const uint mask = 0x7fffffff;
	static const __m128 k1 = _mm_set_ps1(atan2_p1), k2 = _mm_set_ps1(atan2_p3), k3 = _mm_set_ps1(atan2_p5), k4 = _mm_set_ps1(atan2_p7), DE = _mm_set_ps1((float)DBL_EPSILON), a1 = _mm_set_ps1(90.0), a2 = _mm_set_ps1(180.0), a3 = _mm_set_ps1(360.0), zeros = _mm_set_ps1(0.0);
	static const __m128 masks = _mm_setr_ps(*(float *)(&mask), *(float *)(&mask), *(float *)(&mask), *(float *)(&mask));
	__m128 grad_x, grad_y, max, min, ax, ay, a, c, c2, flag, temp;

	for (int i = size - 1 - 4, j = 0; i >= 0; i -= 4, j += 4){
		grad_x = _mm_loadu_ps((float*)gradImg_x.data + j);
		grad_y = _mm_loadu_ps((float*)gradImg_y.data + j);
		_mm_storeu_ps(((float*)magImg.data) + j, _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(grad_x, grad_x), _mm_mul_ps(grad_y, grad_y))));
	}
		



	float gx, gy;

	for (int i = size - (size % 4); i < size; i++){
		gx = ((float*)gradImg_x.data)[i];
		gy = ((float*)gradImg_y.data)[i];
		((float*)magImg.data)[i] += SqrtBySQRTSS(gx * gx + gy * gy);
		
	};
}
int ustc_SubImgMatch_mag(Mat grayImg, Mat subImg, int* x, int* y){
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	if (sub_height > height || sub_width > width){
		cout << "error!" << endl;
		return MY_FAIL;
	}
	int sub_size = sub_height * sub_width;
	int h_range = height - sub_height;
	int w_range = width - sub_width;
	uint min_x = 0, min_y = 0;

	int i, j, m, n, img_i, sub_i, a, b;

	Mat grad_x(grayImg.rows, grayImg.cols, CV_32F);
	Mat grad_y(grayImg.rows, grayImg.cols, CV_32F);
	Mat magImg(grayImg.rows, grayImg.cols, CV_32F);
	ustc_CalcGrad(grayImg, grad_x, grad_y);
	CalcMag(grad_x, grad_y, magImg);

	Mat sgrad_x(subImg.rows, subImg.cols, CV_32F);
	Mat sgrad_y(subImg.rows, subImg.cols, CV_32F);
	Mat smagImg(subImg.rows, subImg.cols, CV_32F);
	ustc_CalcGrad(subImg, sgrad_x, sgrad_y);
	CalcMag(sgrad_x, sgrad_y, smagImg);

	static const uint mask = 0x7fffffff;
	static const __m128 masks = _mm_setr_ps(*(float *)(&mask), *(float *)(&mask), *(float *)(&mask), *(float *)(&mask));

	__m128 sub_data, img_data, temp, out;
	float total_diff, min_diff = 9999999999999;

	for (i = 0; i < h_range; i++)
	{
		for (j = 0; j < w_range; j++)
		{
			total_diff = 0;
			out = _mm_setzero_ps();
			//遍历模板图上的每一个像素
			for (m = 1; m < sub_height - 1; m++)
			{
				for (n = sub_width - 4 - 2, sub_i = m * sub_width + 1, img_i = (i + m) * width + j + 1; n >= 0; n -= 4, sub_i += 4, img_i += 4)
				{
					sub_data = _mm_loadu_ps(((float *)smagImg.data) + sub_i);
					img_data = _mm_loadu_ps(((float *)magImg.data) + img_i);
					temp = _mm_sub_ps(sub_data, img_data);
					temp = _mm_and_ps(temp, masks);
					out = _mm_add_ps(out, temp);

				}
				for (n += 4; n > 0; n--, sub_i++, img_i++)
					total_diff += std::abs(((float *)magImg.data)[sub_i] - ((float *)magImg.data)[img_i]);
			}

			total_diff += out.m128_f32[0] + out.m128_f32[1] + out.m128_f32[2] + out.m128_f32[3];
			if (total_diff < min_diff){
				min_diff = total_diff;
				min_y = i;
				min_x = j;
			}
		}
	}
	*y = min_y;
	*x = min_x;

	cout << *y << " " << *x << endl;
};



int ustc_SubImgMatch_hist(Mat grayImg, Mat subImg, int* x, int* y){
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	if (sub_height > height || sub_width > width){
		cout << "error!" << endl;
		return MY_FAIL;
	}
	int sub_size = sub_height * sub_width;
	int h_range = height - sub_height;
	int w_range = width - sub_width;
	uint min_diff = 0 - 1, min_x = 0, min_y = 0, total_diff;
	int i, j, m, n, img_i, sub_i, a, b;

	int sub_hist[256], hist[256];
	ustc_CalcHist(subImg, sub_hist, 256);

	
	static const __m128i zeros = _mm_setzero_si128();
	static const uint mask = 0x7fffffff ;
	static const __m128i masks = _mm_set1_epi32(mask);
	__m128i sub_data, img_data, temp, out;

	for (i = 0; i < h_range; i++)
	{
		for (j = 0; j < w_range; j++)
		{
			out = _mm_setzero_si128();
			//hist设为0
			for (m = 0; m < 256; m += 16)
				_mm_storeu_si128((__m128i *)(hist + m), zeros);
			//遍历模板图上的每一个像素
			for (m = 0; m < sub_height; m++)
			{
				for (n = sub_width, img_i = (i + m) * width + j; n > 0; n--, img_i++)
					hist[grayImg.data[img_i]]++;
			}

			for (m = 0; m < 256; m += 16){
				sub_data = _mm_loadu_si128((__m128i *) (sub_hist + m));
				img_data = _mm_loadu_si128((__m128i *) (hist + m));
				temp = _mm_sub_epi32(sub_data, img_data);
				temp = _mm_and_si128(temp, masks);
				out = _mm_add_epi32(out, temp);
			}
			

			total_diff = out.m128i_i32[0] + out.m128i_i32[1] + out.m128i_i32[2] + out.m128i_i32[3];

			if (total_diff < min_diff){
				min_diff = total_diff;
				min_y = i;
				min_x = j;
			}
		}
	}
	*y = min_y;
	*x = min_x;
};


void test_bgr2gray()
{
	Mat colorImg = imread("1.jpg", 1);
	if (NULL == colorImg.data)
	{
		cout << "image read failed." << endl;
		return;
	}
#ifdef IMG_SHOW
	namedWindow("colorImg", 0);
	imshow("colorImg", colorImg);
	waitKey(1);
#endif
	Mat grayImg(colorImg.rows, colorImg.cols, CV_8UC1);

	time_t start = clock();
	for (int i = 100; i > 0; i--)
	{
		int flag = ustc_ConvertBgr2Gray(colorImg, grayImg);
	}
	time_t end = clock();
	cout << "time: " << (end - start) << endl;
}

void test_threshold()
{
	Mat grayImg = imread("1.jpg", 0);
	if (NULL == grayImg.data)
	{
		cout << "image read failed." << endl;
		return;
	}
#ifdef IMG_SHOW
	namedWindow("grayImg", 0);
	imshow("grayImg", grayImg);
	waitKey(1);
#endif

	time_t start = clock();
	for (int i = 0; i < 100; i++)
	{
		int flag = ustc_Threshold(grayImg);
	}
	time_t end = clock();
	cout << "time: " << end - start << endl;
}

void test_hist()
{
	Mat grayImg = imread("1.jpg", 0);
	if (NULL == grayImg.data)
	{
		cout << "image read failed." << endl;
		return;
	}
	int hist[256];
#ifdef IMG_SHOW
	namedWindow("grayImg", 0);
	imshow("grayImg", grayImg);
	waitKey(1);
#endif

	time_t start = clock();
	for (int i = 0; i < 100; i++)
	{
		int flag = ustc_CalcHist(grayImg, hist, 256);
	}
	time_t end = clock();
	cout << "time: " << end - start << endl;
}


void test_grad()
{
	Mat grayImg = imread("1.jpg", 0);
	if (NULL == grayImg.data)
	{
		cout << "image read failed." << endl;
		return;
	}
#ifdef IMG_SHOW
	namedWindow("grayImg", 0);
	imshow("grayImg", grayImg);
	waitKey(1);
#endif
	Mat grad_x(grayImg.rows, grayImg.cols, CV_32F);
	Mat grad_y(grayImg.rows, grayImg.cols, CV_32F);

	time_t start = clock();
	for (int i = 0; i < 100; i++)
	{
		int flag = ustc_CalcGrad(grayImg, grad_x, grad_y);
	}
	time_t end = clock();
	cout << "time: " << end - start << endl;
}

void test_angel(){
	Mat grayImg = imread("1.jpg", 0);
	if (NULL == grayImg.data)
	{
		cout << "image read failed." << endl;
		return;
	}
	Mat grad_x(grayImg.rows, grayImg.cols, CV_32F);
	Mat grad_y(grayImg.rows, grayImg.cols, CV_32F);
	Mat angleImg(grayImg.rows, grayImg.cols, CV_32F);
	Mat magImg(grayImg.rows, grayImg.cols, CV_32F);
	ustc_CalcGrad(grayImg, grad_x, grad_y);
#ifdef IMG_SHOW
	namedWindow("grayImg", 0);
	imshow("grayImg", grayImg);
	waitKey(1);
#endif

	time_t start = clock();
	for (int i = 0; i < 100; i++)
	{
		int flag = ustc_CalcAngleMag(grad_x, grad_y, angleImg, magImg);
	}
	time_t end = clock();
	cout << "time: " << end - start << endl;
}

void test_match_gray()
{
	Mat grayImg = imread("1.jpg", 0);
	if (NULL == grayImg.data)
	{
		cout << "image read failed." << endl;
		return;
	}
	Mat subImg(100, 100, CV_8U);
	grayImg(Rect(100, 100, 100, 100)).convertTo(subImg, subImg.type(), 1, 0);
#ifdef IMG_SHOW
	namedWindow("grayImg", 0);
	imshow("grayImg", grayImg);
	waitKey(1);
	namedWindow("subImg", 0);
	imshow("subImg", subImg);
	waitKey(1);
	system("pause");
#endif
	
	int x, y;
	time_t start = clock();
	for (int i = 0; i < 1; i++)
	{
		int flag = ustc_SubImgMatch_gray(grayImg, subImg, &x, &y);
	}
	time_t end = clock();
	cout << "time: " << end - start << endl;
}

void test_match_bgr()
{
	Mat colorImg = imread("1.jpg", 1);
	if (NULL == colorImg.data)
	{
		cout << "image read failed." << endl;
		return;
	}
	Mat subImg(100, 100, CV_8U);
	colorImg(Rect(100, 100, 100, 100)).convertTo(subImg, subImg.type(), 1, 0);
#ifdef IMG_SHOW
	namedWindow("grayImg", 0);
	imshow("grayImg", grayImg);
	waitKey(1);
	namedWindow("subImg", 0);
	imshow("subImg", subImg);
	waitKey(1);
	system("pause");
#endif

	int x, y;
	time_t start = clock();
	for (int i = 0; i < 1; i++)
	{
		int flag = ustc_SubImgMatch_bgr(colorImg, subImg, &x, &y);
	}
	time_t end = clock();
	cout << "time: " << end - start << endl;
}

void test_match_corr()
{
	Mat grayImg = imread("1.jpg", 0);
	if (NULL == grayImg.data)
	{
		cout << "image read failed." << endl;
		return;
	}
	Mat subImg(20, 20, CV_8U);
	grayImg(Rect(100, 100, 20, 20)).convertTo(subImg, subImg.type(), 1, 0);
#ifdef IMG_SHOW
	namedWindow("grayImg", 0);
	imshow("grayImg", grayImg);
	waitKey(1);
	namedWindow("subImg", 0);
	imshow("subImg", subImg);
	waitKey(1);
	system("pause");
#endif

	int x, y;
	time_t start = clock();
	for (int i = 0; i < 1; i++)
	{
		int flag = ustc_SubImgMatch_corr(grayImg, subImg, &x, &y);
	}
	time_t end = clock();
	cout << "time: " << end - start << endl;
}

void test_match_angle()
{
	Mat grayImg = imread("1.jpg", 0);
	if (NULL == grayImg.data)
	{
		cout << "image read failed." << endl;
		return;
	}
	Mat subImg(100, 100, CV_8U);
	grayImg(Rect(100, 100, 100, 100)).convertTo(subImg, subImg.type(), 1, 0);
#ifdef IMG_SHOW
	namedWindow("grayImg", 0);
	imshow("grayImg", grayImg);
	waitKey(1);
	namedWindow("subImg", 0);
	imshow("subImg", subImg);
	waitKey(1);
	system("pause");
#endif

	int x, y;
	time_t start = clock();
	for (int i = 0; i < 1; i++)
	{
		int flag = ustc_SubImgMatch_angle(grayImg, subImg, &x, &y);
	}
	time_t end = clock();
	cout << "time: " << end - start << endl;
}

void test_match_mag()
{
	Mat grayImg = imread("1.jpg", 0);
	if (NULL == grayImg.data)
	{
		cout << "image read failed." << endl;
		return;
	}
	Mat subImg(100, 100, CV_8U);
	grayImg(Rect(100, 100, 100, 100)).convertTo(subImg, subImg.type(), 1, 0);
#ifdef IMG_SHOW
	namedWindow("grayImg", 0);
	imshow("grayImg", grayImg);
	waitKey(1);
	namedWindow("subImg", 0);
	imshow("subImg", subImg);
	waitKey(1);
	system("pause");
#endif

	int x, y;
	time_t start = clock();
	for (int i = 0; i < 1; i++)
	{
		int flag = ustc_SubImgMatch_mag(grayImg, subImg, &x, &y);
	}
	time_t end = clock();
	cout << "time: " << end - start << endl;
}

void test_match_hist()
{
	Mat grayImg = imread("1.jpg", 0);
	if (NULL == grayImg.data)
	{
		cout << "image read failed." << endl;
		return;
	}
	Mat subImg(30, 30, CV_8U);
	grayImg(Rect(100, 100, 30, 30)).convertTo(subImg, subImg.type(), 1, 0);
#ifdef IMG_SHOW
	namedWindow("grayImg", 0);
	imshow("grayImg", grayImg);
	waitKey(1);
	namedWindow("subImg", 0);
	imshow("subImg", subImg);
	waitKey(1);
	system("pause");
#endif

	int x, y;
	time_t start = clock();
	for (int i = 0; i < 1; i++)
	{
		int flag = ustc_SubImgMatch_hist(grayImg, subImg, &x, &y);
	}
	time_t end = clock();
	cout << "time: " << end - start << endl;
}


int main()
{
	//test_bgr2gray();

	//test_threshold();
	//test_grad();
	//test_match_bgr();
	//test_match_corr();
	//test_match_angle();
	//test_match_mag();
	//test_hist();
	//test_match_hist();
	for (int i = 0; i < 2; i++){
		test_match_hist();
	}


	system("pause");
	return 0;
}

