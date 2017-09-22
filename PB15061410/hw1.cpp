#include "SubImageMatch.h"
#include <emmintrin.h>
#include "iostream"

#define SSE2

const float invtwopi = 0.1591549f;
const float twopi = 6.283185f;
const float threehalfpi = 4.7123889f;
const float pi = 3.141593f;
const float halfpi = 1.570796f;
const float quarterpi = 0.7853982f;
static const __m128 SIGNMASK = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));

__m128 atan_ps(__m128 x)
{
	//quarterpi*x
	//- x*(fabs(x) - 1)
	//*(0.2447f+0.0663f*fabs(x));
	return _mm_sub_ps(_mm_mul_ps(_mm_set1_ps(quarterpi), x),
		_mm_mul_ps(_mm_mul_ps(x, _mm_sub_ps(_mm_andnot_ps(SIGNMASK, x), _mm_set1_ps(1.f))),
		(_mm_add_ps(_mm_set1_ps(0.2447f), _mm_mul_ps(_mm_set1_ps(0.0663f), _mm_andnot_ps(SIGNMASK, x))))));
}

__m128 atan2_ps(__m128 y, __m128 x)
{
	__m128 absxgreaterthanabsy = _mm_cmpgt_ps(_mm_andnot_ps(SIGNMASK, x), _mm_andnot_ps(SIGNMASK, y));
	__m128 ratio = _mm_div_ps(_mm_add_ps(_mm_and_ps(absxgreaterthanabsy, y), _mm_andnot_ps(absxgreaterthanabsy, x)),
		_mm_add_ps(_mm_and_ps(absxgreaterthanabsy, x), _mm_andnot_ps(absxgreaterthanabsy, y)));
	__m128 atan = atan_ps(ratio);

	__m128 xgreaterthan0 = _mm_cmpgt_ps(x, _mm_set1_ps(0.f));
	__m128 ygreaterthan0 = _mm_cmpgt_ps(y, _mm_set1_ps(0.f));

	atan = _mm_xor_ps(atan, _mm_andnot_ps(absxgreaterthanabsy, _mm_and_ps(xgreaterthan0, SIGNMASK))); //negate atan if absx<=absy & x>0

	__m128 shift = _mm_set1_ps(pi);
	shift = _mm_sub_ps(shift, _mm_andnot_ps(absxgreaterthanabsy, _mm_set1_ps(halfpi))); //substract halfpi if absx<=absy
	shift = _mm_xor_ps(shift, _mm_andnot_ps(ygreaterthan0, SIGNMASK)); //negate shift if y<=0
	shift = _mm_andnot_ps(_mm_and_ps(absxgreaterthanabsy, xgreaterthan0), shift); //null if abs>absy & x>0

	return _mm_add_ps(atan, shift);
}


//函数功能：将bgr图像转化成灰度图像
//bgrImg：彩色图，像素排列顺序是bgr
//grayImg：灰度图，单通道
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_ConvertBgr2Gray(Mat bgrImg, Mat& grayImg)
{
	if (bgrImg.data == NULL) return SUB_IMAGE_MATCH_FAIL;
	const int rows = bgrImg.rows;
	const int cols = bgrImg.cols;
	const int pixel = rows*cols;
	const int pixel_radio = pixel / 2;
	const int pixel_remain = pixel % 2;
	grayImg.create(rows, cols, CV_8UC1);

	int16_t temp = 0;

	const uchar* inData = bgrImg.ptr<uchar>(0);
	uchar* outData = grayImg.ptr<uchar>(0);

#ifdef SSE2
	__m128i _temp;
	__m128i _zero = _mm_setzero_si128();
	__m128i _weight = _mm_set_epi16(0, 0, 77, 151, 28, 77, 151, 28);

	for (int j = pixel_radio; j > 0; j--)
	{
		_temp = _mm_mullo_epi16(_mm_unpacklo_epi8(_mm_loadu_si128((__m128i *)inData), _zero), _weight);
		temp = _mm_extract_epi16(_temp, 0) + _mm_extract_epi16(_temp, 1) + _mm_extract_epi16(_temp, 2);
		*outData = temp >> 8;
		outData++;
		temp = _mm_extract_epi16(_temp, 3) + _mm_extract_epi16(_temp, 4) + _mm_extract_epi16(_temp, 5);
		*outData = temp >> 8;
		outData++;
		inData = inData + 6;
	}

	if (pixel_remain == 1)
	{
		temp = *inData * 28;
		inData++;
		temp += *inData * 151;
		inData++;
		temp += *inData * 77;
		inData++;
		*outData = temp >> 8;
		outData++;
	}
#else
	for (int j = 0; j < pixel; j++)
	{
		temp = *inData * 28;
		inData++;
		temp += *inData * 151;
		inData++;
		temp += *inData * 77;
		inData++;
		*outData = temp >> 8;
		outData++;
	}
#endif // SSE2

	if (grayImg.data == NULL)return SUB_IMAGE_MATCH_FAIL;
	return SUB_IMAGE_MATCH_OK;
}

//函数功能：根据灰度图像计算梯度图像
//grayImg：灰度图，单通道
//gradImg_x：水平方向梯度，浮点类型图像，CV32FC1
//gradImg_y：垂直方向梯度，浮点类型图像，CV32FC1
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_CalcGrad(Mat grayImg, Mat& gradImg_x, Mat& gradImg_y)
{
	if (grayImg.data == NULL) return SUB_IMAGE_MATCH_FAIL;
	if (grayImg.rows < 3 || grayImg.cols < 3) return SUB_IMAGE_MATCH_FAIL;

	const int rows = grayImg.rows;
	const int cols = grayImg.cols; 
	gradImg_x.create(rows, cols, CV_32FC1);	
	gradImg_y.create(rows, cols, CV_32FC1);
	gradImg_x.setTo(0);
	gradImg_y.setTo(0);

	float* grad_x = ((float*)gradImg_x.data) + cols + 1;
	float* grad_y = ((float*)gradImg_y.data) + cols + 1;

	const uchar* gray_1 = (uchar*)grayImg.data;
	const uchar* gray_2 = gray_1+ cols;
	const uchar* gray_3 = gray_2+ cols;

	for (int row_i = 1; row_i < rows - 1; row_i++)
	{
		for (int col_j = 1; col_j < cols - 1; col_j++)
		{
			*grad_x = *(gray_1 + 2) + ((*(gray_2 + 2)) * 2) + (*(gray_3 + 2)) - *gray_1 - ((*gray_2) * 2) - *gray_3;
			*grad_y = -*gray_1 - ((*(gray_1 + 2)) * 2) - *(gray_1 + 2) + *gray_3 + ((*(gray_3 + 2)) * 2) + *(gray_3 + 2);
			
			grad_x++;
			grad_y++;
			gray_1++;
			gray_2++;
			gray_3++;

		}
		grad_x += 2;
		grad_y += 2;
		gray_1+=2;
		gray_2+=2;
		gray_3+=2;			
	}

	if (gradImg_x.data == NULL)return SUB_IMAGE_MATCH_FAIL;
	if (gradImg_y.data == NULL)return SUB_IMAGE_MATCH_FAIL;

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
	if (gradImg_x.data == NULL) return SUB_IMAGE_MATCH_FAIL;
	if (gradImg_y.data == NULL) return SUB_IMAGE_MATCH_FAIL;

	if (gradImg_x.rows != gradImg_y.rows) return SUB_IMAGE_MATCH_FAIL;
	if (gradImg_x.cols != gradImg_y.cols) return SUB_IMAGE_MATCH_FAIL;

	const int rows = gradImg_x.rows;
	const int cols = gradImg_x.cols;
	const int pixie = rows*cols;
	const int pixie_ratio = pixie / 4;
	const int pixie_remain = pixie - pixie_ratio * 4;

	angleImg.create(rows, cols, CV_32FC1);
	magImg.create(rows, cols, CV_32FC1);

	float grad_x = 0, grad_y = 0;
	float mag = 0, angle = 0;

#ifdef SSE2
	__m128 _grad_x, _grad_y;
	__m128 _temp;

	const float* xData = gradImg_x.ptr<float>(0);
	const float* yData = gradImg_y.ptr<float>(0);
	float* magData = magImg.ptr<float>(0);	
	float* angleData = angleImg.ptr<float>(0);

	for (int col_j = 0; col_j < pixie_ratio; col_j++)
	{
		_grad_x = _mm_loadu_ps(xData);		
		_grad_y = _mm_loadu_ps(yData); 

		_temp = _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(_grad_x, _grad_x), _mm_mul_ps(_grad_y, _grad_y)));
		_mm_store_ps(magData, _temp); magData += 4;

		angle = atan2(*xData, *yData);
		if (angle < 0) *angleData = angle * (float)57.29578 + 360;
		else if (angle > 0) *angleData = angle * (float)57.29578;
		else  *angleData = 0;
		angleData++;

		angle = atan2(*(xData + 1), *(yData + 1));
		if (angle < 0) *angleData = angle * (float)57.29578 + 360;
		else if (angle > 0) *angleData = angle * (float)57.29578;
		else  *angleData = 0;
		angleData++;

		angle = atan2(*(xData + 2), *(yData + 2));
		if (angle < 0) *angleData = angle * (float)57.29578 + 360;
		else if (angle > 0) *angleData = angle * (float)57.29578;
		else  *angleData = 0;
		angleData++;

		angle = atan2(*(xData + 3), *(yData + 3));
		if (angle < 0) *angleData = angle * (float)57.29578 + 360;
		else if (angle > 0) *angleData = angle * (float)57.29578;
		else  *angleData = 0;
		angleData++;

		xData += 4; yData += 4;
	}

	for (int col_j = 0; col_j < pixie_remain; col_j++)
	{
		grad_x = *xData; xData++;
		grad_y = *yData; yData++;

		angle = atan2(grad_y, grad_x);
		if (angle < 0) *angleData = angle * (float)57.29578 + 360;
		else if (angle > 0) *angleData = angle * (float)57.29578;
		else  *angleData = 0;
		angleData++;

		mag = sqrt(grad_x*grad_x + grad_y*grad_y);
		*magData = mag;
		magData++;
	}


#else
	for (int row_i = 1; row_i < rows; row_i++)
	{
		for (int col_j = 1; col_j < cols; col_j++)
		{
			grad_x = ((float*)gradImg_x.data)[row_i * cols + col_j];
			grad_y = ((float*)gradImg_y.data)[row_i * cols + col_j];
			angle = atan2(grad_y, grad_x);
			//if (angle < -181) 
			//	angle = 0;
			//assert(isnan(angle) == 0);
			if (angle < 0)((float*)angleImg.data)[row_i * cols + col_j] = angle * 180 / 3.1415926 + 360;
			else if (angle > 0)((float*)angleImg.data)[row_i * cols + col_j] = angle * 180 / 3.1415926;
			else ((float*)angleImg.data)[row_i * cols + col_j] = 0;

			mag = sqrt(grad_x*grad_x + grad_y*grad_y);
			((float*)magImg.data)[row_i * cols + col_j] = mag;
		}
	}
#endif // SSE2

	if (angleImg.data == NULL)return SUB_IMAGE_MATCH_FAIL;
	if (magImg.data == NULL)return SUB_IMAGE_MATCH_FAIL;

	return SUB_IMAGE_MATCH_OK;
}

//函数功能：对灰度图像进行二值化
//grayImg：灰度图，单通道
//binaryImg：二值图，单通道
//th：二值化阈值，高于此值，255，低于此值0
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_Threshold(Mat grayImg, Mat& binaryImg, int th)
{
	if (grayImg.data == NULL) return SUB_IMAGE_MATCH_FAIL;
	if (th < 0) th = 0;
	if (th > 255) th = 255;

	const int rows = grayImg.rows;
	const int cols = grayImg.cols;
	const int pixel = rows*cols;
	const int pixel_radio = pixel / 16;
	const int pixel_remain = pixel % 16;
	binaryImg.create(rows, cols, CV_8UC1);

	const uchar* inData = grayImg.ptr<uchar>(0);
	uchar* outData = binaryImg.ptr<uchar>(0);

#ifdef SSE2
	__m128i _th = _mm_set1_epi8(th + 128);
	__m128i _const128 = _mm_set1_epi8(0x80);
	__m128i _temp;

	for (int j = pixel_radio; j > 0; j--)
	{
		_temp = _mm_cmpgt_epi8(_mm_add_epi8(_mm_loadu_si128((__m128i *)inData), _const128), _th);
		_mm_storeu_si128((__m128i *)outData, _temp);
		inData = inData + 16;
		outData = outData + 16;
	}

	for (int j = pixel_remain; j > 0; j--)
	{
		if (*inData>th) *outData = 255;
		else *outData = 0;
		inData++;
		outData++;
	}
#else
	for (int j = 0; j < pixel; j++)
	{
		if (*inData>th) *outData = 255;
		else *outData = 0;
		inData++;
		outData++;
	}
#endif // SSE2

	if (binaryImg.data == NULL)return SUB_IMAGE_MATCH_FAIL;
	return SUB_IMAGE_MATCH_OK;
}

//函数功能：对灰度图像计算直方图
//grayImg：灰度图，单通道
//hist：直方图
//hist_len：直方图的亮度等级，直方图数组的长度
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_CalcHist(Mat grayImg, int* hist, int hist_len)
{
	if (grayImg.data == NULL) return SUB_IMAGE_MATCH_FAIL;
	if (hist_len != 256)return SUB_IMAGE_MATCH_FAIL;
	const int rows = grayImg.rows;
	const int cols = grayImg.cols;
	const int pixie = rows*cols;

	const uchar* Data = grayImg.ptr<uchar>(0);

	//直方图清零
	for (int i = 0; i < hist_len; i++)
	{
		hist[i] = 0;
	}

	//计算直方图
	for (int i = pixie; i > 0; i--)
	{
		int pixVal = *Data;
		hist[pixVal]++; 
		Data++;
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
	if (grayImg.data == NULL) return SUB_IMAGE_MATCH_FAIL;
	if (subImg.data == NULL) return SUB_IMAGE_MATCH_FAIL;
	if (subImg.rows > grayImg.rows || subImg.cols > grayImg.cols) return SUB_IMAGE_MATCH_FAIL;

	const int rows = grayImg.rows;
	const int cols = grayImg.cols;
	const int sub_rows = subImg.rows;
	const int sub_cols = subImg.cols;
	const int rows_diff = rows - sub_rows;
	const int cols_diff = cols - sub_cols;

	int min = 0x4fffffff;
	__m128i temp;

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i < rows_diff; i++)
	{
		for (int j = 0; j < cols_diff; j++)
		{
			int total_diff = 0;
			const uchar* Data = grayImg.ptr<uchar>(0) + i * cols + j;
			uchar* subData = subImg.ptr<uchar>(0);
			//遍历模板图上的每一个像素
			for (int sub_i = 0; sub_i < sub_rows; sub_i++)
			{
				int sub_j = 0;
				for (sub_j = 0; sub_j < sub_cols - 16; sub_j += 16)
				{
					temp = _mm_sad_epu8(_mm_loadu_si128((__m128i *)Data), _mm_loadu_si128((__m128i *)subData));
					total_diff = total_diff + _mm_extract_epi16(temp, 0) + _mm_extract_epi16(temp, 4);

					Data = Data + 16;
					subData = subData + 16;
				}

				for (sub_j; sub_j < sub_cols; sub_j ++)
				{
					total_diff += abs(*Data - *subData);
					Data++;
					subData++;
				}
				Data += cols_diff;
			}
			if (total_diff < min)
			{
				min = total_diff;
				*x = j;
				*y = i;
			}
		}
	}
	return SUB_IMAGE_MATCH_OK;
}

//函数功能：利用色彩进行子图匹配
//colorImg：彩色图，三通道
//subImg：模板子图，三通道
//x：最佳匹配子图左上角x坐标
//y：最佳匹配子图左上角y坐标
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int* x, int* y)
{
	if (colorImg.data == NULL) return SUB_IMAGE_MATCH_FAIL;
	if (subImg.data == NULL) return SUB_IMAGE_MATCH_FAIL;
	if (subImg.rows > colorImg.rows || subImg.cols > colorImg.cols) return SUB_IMAGE_MATCH_FAIL;

	const int rows = colorImg.rows;
	const int cols = colorImg.cols * 3;
	const int sub_rows = subImg.rows;
	const int sub_cols = subImg.cols * 3;

	const int row_diff = rows - sub_rows;
	const int cols_diff = cols - sub_cols;
	const int sub_cols_new = sub_cols - 16;

	int min = 0x4fffffff;
	__m128i temp;

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i < row_diff; i++)
	{
		for (int j = 0; j < cols_diff; j++)
		{
			int total_diff = 0;
			const uchar* Data = colorImg.ptr<uchar>(0) + i * cols + j;
			uchar* subData = subImg.ptr<uchar>(0);
			//遍历模板图上的每一个像素
			for (int sub_i = 0; sub_i < sub_rows; sub_i++)
			{
				int sub_j = 0;
				for (sub_j = 0; sub_j < sub_cols_new; sub_j += 16)
				{
					temp = _mm_sad_epu8(_mm_loadu_si128((__m128i *)Data), _mm_loadu_si128((__m128i *)subData));
					total_diff = total_diff + _mm_extract_epi16(temp, 0) + _mm_extract_epi16(temp, 4);

					Data = Data + 16;
					subData = subData + 16;
				}

				for (sub_j; sub_j < sub_cols; sub_j++)
				{
					total_diff += abs(*Data - *subData);
					Data++;
					subData++;
				}
				Data += cols_diff;
			}
			if (total_diff < min)
			{
				min = total_diff;
				*x = j / 3;
				*y = i;
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
	if (grayImg.data == NULL) return SUB_IMAGE_MATCH_FAIL;
	if (subImg.data == NULL) return SUB_IMAGE_MATCH_FAIL;
	if (subImg.rows > grayImg.rows || subImg.cols > grayImg.cols) return SUB_IMAGE_MATCH_FAIL;

	int rows = grayImg.rows;
	int cols = grayImg.cols;
	int sub_rows = subImg.rows;
	int sub_cols = subImg.cols;
	int sub_pixel = sub_rows*sub_cols;

	float max = 0;

	float total_diff = 0;
	long int SS = 0, TT = 0, ST = 0;

#ifdef SSE2
	__m128i _temp, _s, _t, _ss, _st, _ss1, _ss2;
	__m128i _zero = _mm_setzero_si128();
	//TT
	const uchar* subDataTT = subImg.ptr<uchar>(0);
	for (int sub_i = 0; sub_i < sub_pixel; sub_i++)
	{
		TT += *subDataTT**subDataTT;
		subDataTT++;
	}

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i < rows - sub_rows; i++)
	{
		for (int j = 0; j < cols - sub_cols; j++)
		{
			total_diff = 0, SS = 0, ST = 0;
			const uchar* Data = grayImg.ptr<uchar>(0) + i * cols + j;
			const uchar* subData = subImg.ptr<uchar>(0);
			//遍历模板图上的每一个像素
			for (int sub_i = 0; sub_i < sub_rows; sub_i++)
			{

				int sub_j = 0;
				for (sub_j = 0; sub_j < sub_cols - 8; sub_j += 8)
				{
					_s = _mm_unpacklo_epi8(_mm_loadu_si128((__m128i *)Data), _zero);
					_t = _mm_unpacklo_epi8(_mm_loadu_si128((__m128i *)subData), _zero);

					_ss = _mm_mullo_epi16(_s, _s);
					SS += _mm_extract_epi16(_ss, 0) + _mm_extract_epi16(_ss, 1) + _mm_extract_epi16(_ss, 2) + _mm_extract_epi16(_ss, 3)
						+ _mm_extract_epi16(_ss, 4) + _mm_extract_epi16(_ss, 5) + _mm_extract_epi16(_ss, 6) + _mm_extract_epi16(_ss, 7);

					_st = _mm_mullo_epi16(_s, _t);
					ST += _mm_extract_epi16(_st, 0) + _mm_extract_epi16(_st, 1) + _mm_extract_epi16(_st, 2) + _mm_extract_epi16(_st, 3)
						+ _mm_extract_epi16(_st, 4) + _mm_extract_epi16(_st, 5) + _mm_extract_epi16(_st, 6) + _mm_extract_epi16(_st, 7);
					Data += 8;
					subData += 8;
				}
				for (sub_j; sub_j < sub_cols; sub_j++)
				{
					SS += *Data * *Data;
					ST += *subData* *Data;
					Data++;
					subData++;
				}
				Data += (cols - sub_cols);
			}
			total_diff = ST / sqrt(SS) / sqrt(TT);
			if (total_diff > max)
			{
				max = total_diff;
				*x = j;
				*y = i;
			}
		}
	}
#else
	//TT
	const uchar* subDataTT = subImg.ptr<uchar>(0);
	for (int sub_i = 0; sub_i < sub_pixel; sub_i++)
	{
		TT += *subDataTT**subDataTT;
		subDataTT++;
	}

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i < rows - sub_rows; i++)
	{
		for (int j = 0; j < cols - sub_cols; j++)
		{
			total_diff = 0, SS = 0, ST = 0;
			const uchar* Data = grayImg.ptr<uchar>(0) + i * cols + j;
			const uchar* subData = subImg.ptr<uchar>(0);
			//遍历模板图上的每一个像素
			for (int sub_i = 0; sub_i < sub_rows; sub_i++)
			{
				for (int sub_j = 0; sub_j < sub_cols; sub_j++)
				{
					SS += *Data * *Data;
					ST += *subData* *Data;
					Data++;
					subData++;
				}
				Data += (cols - sub_cols);
			}
			total_diff = ST / sqrt(SS) / sqrt(TT);
			if (total_diff > max)
			{
				max = total_diff;
				*x = j;
				*y = i;
			}
		}
	}
#endif // !SSE2

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
	if (grayImg.data == NULL) return SUB_IMAGE_MATCH_FAIL;
	if (subImg.data == NULL) return SUB_IMAGE_MATCH_FAIL;
	if (subImg.rows > grayImg.rows || subImg.cols > grayImg.cols) return(SUB_IMAGE_MATCH_FAIL);

	Mat gradImg_x, gradImg_y, angleImg, magImg;
	if (ustc_CalcGrad(grayImg, gradImg_x, gradImg_y) == SUB_IMAGE_MATCH_FAIL) return SUB_IMAGE_MATCH_FAIL;
	if (ustc_CalcAngleMag(gradImg_x, gradImg_y, angleImg, magImg) == SUB_IMAGE_MATCH_FAIL) return SUB_IMAGE_MATCH_FAIL;

	Mat subImg_x, subImg_y, subAngleImg, subMagImg;
	if (ustc_CalcGrad(subImg, subImg_x, subImg_y) == SUB_IMAGE_MATCH_FAIL) return SUB_IMAGE_MATCH_FAIL;
	if (ustc_CalcAngleMag(subImg_x, subImg_y, subAngleImg, subMagImg) == SUB_IMAGE_MATCH_FAIL) return SUB_IMAGE_MATCH_FAIL;

	const int rows = angleImg.rows;
	const int cols = angleImg.cols;
	const int sub_rows = subAngleImg.rows;
	const int sub_cols = subAngleImg.cols;

	const int row_diff = rows - sub_rows;
	const int cols_diff = cols - sub_cols;

	int min = 0x4fffffff;

	int temp;

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 1; i < row_diff - 1; i++)
	{
		for (int j = 1; j < cols_diff - 1; j++)
		{
			int total_diff = 0;
			const float* Data = angleImg.ptr<float>(0) + i * cols + j +1;
			const float* subData = subAngleImg.ptr<float>(0) + sub_cols +1;
			//遍历模板图上的每一个像素
			for (int sub_i = 1; sub_i < sub_rows - 1; sub_i++)
			{

				for (int sub_j = 1; sub_j < sub_cols - 1; sub_j++)
				{
					temp = (int)(*Data - *subData);
					total_diff += (temp >> 31)*(temp << 1) + temp;
					Data++;
					subData++;
				}
				Data += 2;
				subData += 2;
				Data += cols_diff;
			}
			if (total_diff < min)
			{
				min = total_diff;
				*x = j;
				*y = i - 1;
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
	if (grayImg.data == NULL) return SUB_IMAGE_MATCH_FAIL;
	if (subImg.data == NULL) return SUB_IMAGE_MATCH_FAIL;
	if (subImg.rows > grayImg.rows || subImg.cols > grayImg.cols) return(SUB_IMAGE_MATCH_FAIL);

	Mat gradImg_x, gradImg_y, angleImg, magImg;
	if (ustc_CalcGrad(grayImg, gradImg_x, gradImg_y) == SUB_IMAGE_MATCH_FAIL) return SUB_IMAGE_MATCH_FAIL;
	if (ustc_CalcAngleMag(gradImg_x, gradImg_y, angleImg, magImg) == SUB_IMAGE_MATCH_FAIL) return SUB_IMAGE_MATCH_FAIL;

	Mat subImg_x, subImg_y, subAngleImg, subMagImg;
	if (ustc_CalcGrad(subImg, subImg_x, subImg_y) == SUB_IMAGE_MATCH_FAIL) return SUB_IMAGE_MATCH_FAIL;
	if (ustc_CalcAngleMag(subImg_x, subImg_y, subAngleImg, subMagImg) == SUB_IMAGE_MATCH_FAIL) return SUB_IMAGE_MATCH_FAIL;

	const int rows = magImg.rows;
	const int cols = magImg.cols;
	const int sub_rows = subMagImg.rows;
	const int sub_cols = subMagImg.cols;

	const int row_diff = rows - sub_rows;
	const int cols_diff = cols - sub_cols;

	int min = 0x4fffffff;

	int temp;

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 1; i < row_diff - 1; i++)
	{
		for (int j = 1; j < cols_diff - 1; j++)
		{
			int total_diff = 0;
			const float* Data = magImg.ptr<float>(0) + i * cols + j;
			const float* subData = subMagImg.ptr<float>(0);
			//遍历模板图上的每一个像素
			for (int sub_i = 1; sub_i < sub_rows - 1; sub_i++)
			{

				for (int sub_j = 1; sub_j < sub_cols - 1; sub_j++)
				{
					temp = (int)(*Data - *subData);
					total_diff += (temp >> 31)*(temp << 1) + temp;
					Data++;
					subData++;
				}
				Data += 2;
				subData += 2;
				Data += cols_diff;
			}
			if (total_diff < min)
			{
				min = total_diff;
				*x = j;
				*y = i;
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
	if (grayImg.data == NULL) return SUB_IMAGE_MATCH_FAIL;
	if (subImg.data == NULL) return SUB_IMAGE_MATCH_FAIL;
	if (subImg.rows > grayImg.rows || subImg.cols > grayImg.cols) return SUB_IMAGE_MATCH_FAIL;

	int gray_rows = grayImg.rows;
	int gray_cols = grayImg.cols;
	int sub_rows = subImg.rows;
	int sub_cols = subImg.cols;

	int sub_pixie = sub_rows*sub_cols;
	int gray_pixie = gray_rows*gray_cols;

	int sub_hist[256] = { 0 };
	ustc_CalcHist(subImg, sub_hist, 256);

	int temp;
	int count;
	int total_diff = 0;
	int min = 0x4fffffff;
	int value;

	for (int i = 0; i < gray_rows - sub_rows; i++)
	{
		for (int j = 0; j < gray_cols - sub_cols; j++)
		{
			int gray_hist[256] = { 0 };
			const uchar* Data = grayImg.ptr<uchar>(0) + i*gray_cols + j;
			for (int sub_i = 0; sub_i < sub_rows; sub_i++)
			{
				for (int sub_j = 0; sub_j < sub_cols; sub_j++)
				{
					value = *Data;
					gray_hist[value]++;
					Data++;
				}
				Data += gray_cols - sub_cols;
			}

			total_diff = 0;
			for (count = 0; count < 256; count++)
			{
				temp = gray_hist[count] - sub_hist[count];
				total_diff += (temp >> 31)*(temp << 1) + temp;
			}
			if (total_diff < min)
			{
				min = total_diff;
				*x = j;
				*y = i;
			}
		}
	}

	if (min == 0x4fffffff)return SUB_IMAGE_MATCH_FAIL;

	return SUB_IMAGE_MATCH_OK;
}

