#include"opencv2/opencv.hpp"
#include"cv.h"
#include"math.h"
#include"SubImageMatch.h"
#include <time.h>
#include "emmintrin.h"

using namespace cv;
using namespace std;
float tanlab[512][512];
float sqrtlab[512][512];
float abslab[512];
uchar firstflag = 0;
uchar absflag = 0;


int ustc_ConvertBgr2Gray(Mat bgrImg, Mat& grayImg) {
	uchar* ptr_c = bgrImg.data;
	uchar* ptr_g = grayImg.data;
	int b = 0.114 * 1024;
	int g = 0.587 * 1024;
	int r = 0.299 * 1024;
	int wid = bgrImg.cols;
	int len = bgrImg.rows;
	int size = len*wid;
	for (int i = 0; i < size; i++) {
		int c = 0;
		c += *(ptr_c+3*i) * b;
		c += *(ptr_c+3*i+1) * g;
		c += *(ptr_c+3*i+2) * r;
		c >>= 10;
		*(ptr_g+i) = c;
	}
	return 1;
}

int ustc_CalcGrad(Mat grayImg, Mat& gradImg_x, Mat& gradImg_y) {
	
	int wid = grayImg.cols;
	int len = grayImg.rows;
	uchar* ptr_g = grayImg.data + wid + 1;
	float* ptr_x = (float*)(gradImg_x.data);
	float* ptr_y = (float*)(gradImg_y.data);
	
	int size = len*wid-wid-1;
	for (int i = wid + 1; i < size; i++) {
		int c = 0;
		c -= *((ptr_g)-wid - 1);
		c += *((ptr_g)-wid + 1);
		c -= (*((ptr_g)-1)) << 1;
		c += (*((ptr_g)+1)) << 1;
		c -= *((ptr_g)+wid - 1);
		c += *((ptr_g)+wid + 1);
		int d = 0;
		d += *((ptr_g)-wid - 1);
		d += (*((ptr_g)-wid)) << 1;
		d += *((ptr_g)-wid + 1);
		d -= *((ptr_g)+wid - 1);
		d -= (*((ptr_g)+wid)) << 1;
		d -= *((ptr_g++) + wid + 1);
		
		*(ptr_x + i) = c;
		*(ptr_y + i) = d;
		/**(ptr_x + i) = abs(c);
		*(ptr_y + i) = abs(d);*/
	}
	return 1;
}

int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y, Mat& angleImg, Mat& magImg) {
	int wid = gradImg_x.cols;
	int len = gradImg_x.rows;
	float* ptr_x = (float*)(gradImg_x.data);
	float* ptr_y = (float*)(gradImg_y.data);
	float* ptr_a = (float*)(angleImg.data);
	float* ptr_m = (float*)(magImg.data);
	float x, y;
	int xi, yi;
	/*float tanlab[256][256];*/
	if (firstflag == 0) {
		for (int i = 0; i < 512; i++) {
			for (int j = 0; j < 512; j++) {
				tanlab[i][j] = atan2(i - 256, j - 256)*40.74 + 128;
				float a = (i - 256) *(i - 256) + (j - 256) * (j - 256);
				float b = a;
				unsigned int e = *(unsigned int *)&b;
				e = (e + 0x3f76cf62) >> 1;
				b = *(float *)&e;
				sqrtlab[i][j] = (b + a / b)*0.5;
				
			}
		}
		firstflag = 1;
	}
	
	int size = len*wid - wid - 1;
	for (int i = wid + 1; i < size; i++) {
		xi = *(ptr_x + i);
		yi = *(ptr_y + i);
		xi >>= 2;
		yi >>= 2;
		xi += 256;
		yi += 256;
		
		//*(ptr_m + i) = sqrt(x*x + y*y);
		//73ms
		*(ptr_m + i) = sqrtlab[xi][yi]*4;
		//106ms
		/*float a = xi*xi + yi * yi;
		float b = a;
		unsigned int e = *(unsigned int *)&b;
		e = (e + 0x3f76cf62) >> 1;
		b = *(float *)&e;
		*(ptr_m + i) = (b + a / b)*0.5;*/
		//116ms
		/*float a = x*x + y * y;
		float b = a;
		float ahalf = 0.5*b;
		unsigned int e = *(unsigned int *)&b;
		e = 0x5f3759df - (e >> 1);
		b = *(float *)&e;
		*(ptr_m + i) =a*b*(1.5f-ahalf*b*b);*/
		//399ms
		//*(ptr_m + i) = sqrt(*(ptr_x + i)**(ptr_x + i) + *(ptr_y + i)**(ptr_y + i));
			/**(ptr_a + i) = atan2(x, y);*/
			/*xi = x;
			yi = y;*/
			
			
		*(ptr_a + i) = tanlab[xi][yi];
		
	}
	return 1;
}

int ustc_Threshold(Mat grayImg, Mat& binaryImg, int th) {
	int width = grayImg.cols;
	int height = grayImg.rows;
	int size = (width*height)>>4;
	int err = (width*height) - (size << 4);
	uchar th1 = th+128;
	uchar cmp[16] = { th1,th1,th1,th1,th1,th1,th1,th1,th1,th1,th1,th1,th1,th1,th1,th1 };
	uchar _128[16] = { 128,128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128 };
	uchar* ptr_g = (grayImg.data);
	uchar* ptr_b = (binaryImg.data);
	__m128i *ptr_cg = (__m128i *)ptr_g;
	__m128i *ptr_cb = (__m128i *)ptr_b;
	__m128i *ptr_cth = (__m128i *)cmp;
	__m128i *ptr_128 = (__m128i *)_128;
	__m128i dst_th= _mm_loadu_si128(ptr_cth);
	__m128i temp,temp2,temp3;
	temp3 = _mm_loadu_si128(ptr_128);
	for (int i = 0; i < size; i++) {
		temp= _mm_load_si128(ptr_cg++);
		temp2 = _mm_adds_epu8(temp, temp3);
		__m128i dst = _mm_cmpgt_epi8(temp2,dst_th);

		_mm_store_si128(ptr_cb++, dst);
	}
	for (int i = 0; i < err; i++) {
		*(ptr_b + i + (size >> 4)) = 255 * (*(ptr_g + i + (size >> 4))>th);
	}
	/*for (int i = 0; i < size; i++) {
		*(ptr_b + i ) = 255 * (*(ptr_g + i)>th);
	}*/




	return 1;
}






int ustc_CalcHist(Mat grayImg, int* hist, int hist_len) {

	int width = grayImg.cols;
	int height = grayImg.rows;
	uchar* ptr_g = (grayImg.data);
	int size = width * height;
	//直方图清零
	for (int i = 0; i < hist_len; i++){
		hist[i] = 0;
	}

	//计算直方图
	
	for (int i = 0; i < size; i++) {
		int pixVal = *(ptr_g+i);
		hist[pixVal]++;
	}
	return 1;
}



int  ustc_SubImgMatch_gray(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return -1;
	}
	if (absflag == 0) {
		absflag = 1;
		for (int i = 0; i < 512; i++) {
			abslab[i] = abs(i - 256);
		}
	}
	uchar* ptr_g = grayImg.data;
	uchar* ptr_s = subImg.data;
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int delta_height = height - sub_height;
	int delta_width = width - sub_width;
	int i_min=0, j_min=0;
	int min_diff=0x7fffffff;

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i < delta_height; i++)
	{
		for (int j = 0; j < delta_width; j++)
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
					int bigImg_pix = ptr_g[row_index * width + col_index];
					//模板图上的像素
					int template_pix = ptr_s[x * sub_width + y];

					total_diff += abslab[bigImg_pix - template_pix + 256];
				}
			}
			i_min = i*(total_diff < min_diff) + i_min*(total_diff >= min_diff);
			j_min = j*(total_diff < min_diff) + j_min*(total_diff >= min_diff);
			min_diff = total_diff*(total_diff < min_diff) + min_diff*(total_diff >= min_diff);
		}
		
	}
	*x = i_min;
	*y = j_min;

	return 1;
}

int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int* x, int* y) {
	if (NULL == colorImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return -1;
	}
	if (absflag == 0) {
		absflag = 1;
		for (int i = 0; i < 512; i++) {
			abslab[i] = abs(i - 256);
		}
	}
	uchar* ptr_c = colorImg.data;
	uchar* ptr_s = subImg.data;
	int width = colorImg.cols;
	int height = colorImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int delta_height = height - sub_height;
	int delta_width = width - sub_width;
	int i_min=0, j_min=0;
	int min_diff = 0x7fffffff;

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i < delta_height; i++)
	{
		for (int j = 0; j < delta_width; j++)
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
					int bigImg_pix = ptr_c[3 * (row_index * width + col_index)];
					//模板图上的像素
					int template_pix = ptr_s[3 * (x * sub_width + y)];
					total_diff += abslab[bigImg_pix - template_pix+256];

					 bigImg_pix = ptr_c[3 * (row_index * width + col_index) + 1];
					//模板图上的像素
					 template_pix = ptr_s[3 * (x * sub_width + y)+1];
					total_diff += abslab[bigImg_pix - template_pix + 256];

					 bigImg_pix = ptr_c[3 * (row_index * width + col_index) + 2];
					//模板图上的像素
					 template_pix = ptr_s[3 * (x * sub_width + y) + 2];
					total_diff += abslab[bigImg_pix - template_pix + 256];

				}
			}
			
				i_min = i*(total_diff < min_diff)+ i_min*(total_diff >= min_diff);
				j_min = j*(total_diff < min_diff)+ j_min*(total_diff >= min_diff);
				min_diff = total_diff*(total_diff < min_diff)+ min_diff*(total_diff >= min_diff);
			
		}
		*x = i_min;
		*y = j_min;
	}

	return 1;
}


int ustc_SubImgMatch_corr(Mat grayImg, Mat subImg, int* x, int* y) {
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return -1;
	}
	uchar* ptr_g = grayImg.data;
	uchar* ptr_s = subImg.data;
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int delta_height = height - sub_height;
	int delta_width = width - sub_width;
	int i_max = 0, j_max = 0;
	int min_diff = 0x7fffffff;

	//遍历大图每一个像素，注意行列的起始、终止坐标
	int xy = 0;
	int xx = 0;
	int yy = 0;
	float inv_x = 0;
	float inv_y = 0;
	float crltn = 0;
	float crltn_max = 0;
	for (int x = 0; x < sub_height; x++)
	{
		for (int y = 0; y < sub_width; y++)
		{
			int template_pix = ptr_s[y * sub_width + x];
			yy += template_pix*template_pix;
			float b = yy;
			float ahalf = 0.5*b;
			unsigned int e = *(unsigned int *)&b;
			e = 0x5f3759df - (e >> 1);
			b = *(float *)&e;
			inv_y = b*(1.5f - ahalf*b*b);
		}
	}
	for (int i = 0; i < delta_height; i++)
	{
		for (int j = 0; j < delta_width; j++)
		{
			xy = 0;
			xx = 0;
			
			//遍历模板图上的每一个像素
			for (int x = 0; x < sub_height; x++)
			{
				for (int y = 0; y < sub_width; y++)
				{
					//大图上的像素位置
					int row_index = i + x;
					int col_index = j + y;
					int bigImg_pix = ptr_g[row_index * width + col_index];
					//模板图上的像素
					int template_pix = ptr_s[x * sub_width + y];

					xy += bigImg_pix*template_pix;
					xx += bigImg_pix*bigImg_pix;
					
				}
				
			}
			float b = xx;
			float ahalf = 0.5*b;
			unsigned int e = *(unsigned int *)&b;
			e = 0x5f3759df - (e >> 1);
			b = *(float *)&e;
			inv_x = b*(1.5f - ahalf*b*b);

			crltn = inv_x*xy*inv_y;
			if (crltn > crltn_max) {
				i_max = i;
				j_max = j;
				crltn_max = crltn;
			}
			/*i_max = i*(crltn > crltn_max) + i_max*(crltn <= crltn_max);
			j_max = j*(crltn > crltn_max) + j_max*(crltn <= crltn_max);
			crltn_max = crltn*(crltn > crltn_max) + crltn_max*(crltn <= crltn_max);*/
		}
		
	}
	*x = i_max;
	*y = j_max;

	return 1;
}

int ustc_SubImgMatch_angle(Mat grayImg, Mat subImg, int* x, int* y) {
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return -1;
	}
	if (firstflag == 0) {
		for (int i = 0; i < 512; i++) {
			for (int j = 0; j < 512; j++) {
				tanlab[i][j] = atan2(i - 256, j - 256)*40.74 + 128;
				float a = (i - 256) *(i - 256) + (j - 256) * (j - 256);
				float b = a;
				unsigned int e = *(unsigned int *)&b;
				e = (e + 0x3f76cf62) >> 1;
				b = *(float *)&e;
				sqrtlab[i][j] = (b + a / b)*0.5;

			}
		}
		firstflag = 1;
	}
	if (absflag == 0) {
		absflag = 1;
		for (int i = 0; i < 512; i++) {
			abslab[i] = abs(i - 256);
		}
	}
	
	uchar* ptr_g = grayImg.data;
	uchar* ptr_s = subImg.data;
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	Mat sub_aImg(sub_height, sub_width, CV_8UC1);
	Mat aImg(height, width, CV_8UC1);
	int delta_height = height - sub_height;
	int delta_width = width - sub_width;
	int sub_width_o = sub_width;
	int sub_height_o = sub_height;
	sub_width -= 1;
	sub_height -= 1;
	int i_min = 0, j_min = 0;
	int min_diff = 0x7fffffff;
	int wid = sub_aImg.cols;
	int len = sub_aImg.rows;
	int size = wid*len - wid - 1;
	ptr_s += wid + 1;
	
	for (int i = wid + 1; i < size; i++) {
		int c = 0;
		c -= *((ptr_s)-wid - 1);
		c += *((ptr_s)-wid + 1);
		c -= (*((ptr_s)-1)) << 1;
		c += (*((ptr_s)+1)) << 1;
		c -= *((ptr_s)+wid - 1);
		c += *((ptr_s)+wid + 1);
		int d = 0;
		d += *((ptr_s)-wid - 1);
		d += (*((ptr_s)-wid)) << 1;
		d += *((ptr_s)-wid + 1);
		d -= *((ptr_s)+wid - 1);
		d -= (*((ptr_s)+wid)) << 1;
		d -= *((ptr_s++) + wid + 1);

		
		c >>= 2;
		d >>= 2;
		c += 256;
		d += 256;
		sub_aImg.data[i] = tanlab[c][d];
	}

	wid = aImg.cols;
	len = aImg.rows;
	size = wid*len - wid - 1;
	ptr_g += wid + 1;
	for (int i = wid + 1; i < size; i++) {
		int c = 0;
		c -= *((ptr_g)-wid - 1);
		c += *((ptr_g)-wid + 1);
		c -= (*((ptr_g)-1)) << 1;
		c += (*((ptr_g)+1)) << 1;
		c -= *((ptr_g)+wid - 1);
		c += *((ptr_g)+wid + 1);
		int d = 0;
		d += *((ptr_g)-wid - 1);
		d += (*((ptr_g)-wid)) << 1;
		d += *((ptr_g)-wid + 1);
		d -= *((ptr_g)+wid - 1);
		d -= (*((ptr_g)+wid)) << 1;
		d -= *((ptr_g++) + wid + 1);


		c >>= 2;
		d >>= 2;
		c += 256;
		d += 256;
		aImg.data[i] = tanlab[c][d];
	}
	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i < delta_height; i++)
	{
		for (int j = 0; j < delta_width; j++)
		{
			int total_diff = 0;
			//遍历模板图上的每一个像素
			for (int x = 1; x < sub_height; x++)
			{
				for (int y = 1; y < sub_width; y++)
				{
					//大图上的像素位置
					int row_index = i + x;
					int col_index = j + y;
					int bigImg_pix = aImg.data[row_index * width + col_index];
					//模板图上的像素
					int template_pix = sub_aImg.data[x * (sub_width_o) + y];

					total_diff += abslab[bigImg_pix - template_pix + 256];
				}
			}
			i_min = i*(total_diff < min_diff) + i_min*(total_diff >= min_diff);
			j_min = j*(total_diff < min_diff) + j_min*(total_diff >= min_diff);
			min_diff = total_diff*(total_diff < min_diff) + min_diff*(total_diff >= min_diff);
		}

	}
	*x = i_min;
	*y = j_min;

	return 1;
}

int ustc_SubImgMatch_mag(Mat grayImg, Mat subImg, int* x, int* y) {
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return -1;
	}
	if (firstflag == 0) {
		for (int i = 0; i < 512; i++) {
			for (int j = 0; j < 512; j++) {
				tanlab[i][j] = atan2(i - 256, j - 256)*40.74 + 128;
				float a = (i - 256) *(i - 256) + (j - 256) * (j - 256);
				float b = a;
				unsigned int e = *(unsigned int *)&b;
				e = (e + 0x3f76cf62) >> 1;
				b = *(float *)&e;
				sqrtlab[i][j] = (b + a / b)*0.5;

			}
		}
		firstflag = 1;
	}
	
	if (absflag == 0) {
		absflag = 1;
		for (int i = 0; i < 512; i++) {
			abslab[i] = abs(i - 256);
		}
	}
	/*float _tanlab[512][512];
	float _sqrtlab[512][512];*/
	float _abslab[512];
	for (int i = 0; i < 512; i++) {
		_abslab[i] = abslab[i];
		/*for (int j = 0; j < 512; j++) {
			_tanlab[i][j] = tanlab[i][j];
			_sqrtlab[i][j] = sqrtlab[i][j];
		}*/
	}

	uchar* ptr_g = grayImg.data;
	uchar* ptr_s = subImg.data;
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	Mat sub_aImg(sub_height, sub_width, CV_8UC1);
	Mat aImg(height, width, CV_8UC1);
	int delta_height = height - sub_height;
	int delta_width = width - sub_width;
	int sub_width_o = sub_width;
	int sub_height_o = sub_height;
	sub_width -= 1;
	sub_height -= 1;
	int i_min = 0, j_min = 0;
	int min_diff = 0x7fffffff;
	int wid = sub_aImg.cols;
	int len = sub_aImg.rows;
	int size = wid*len - wid - 1;
	ptr_s += wid + 1;

	for (int i = wid + 1; i < size; i++) {
		int c = 0;
		c -= *((ptr_s)-wid - 1);
		c += *((ptr_s)-wid + 1);
		c -= (*((ptr_s)-1)) << 1;
		c += (*((ptr_s)+1)) << 1;
		c -= *((ptr_s)+wid - 1);
		c += *((ptr_s)+wid + 1);
		int d = 0;
		d += *((ptr_s)-wid - 1);
		d += (*((ptr_s)-wid)) << 1;
		d += *((ptr_s)-wid + 1);
		d -= *((ptr_s)+wid - 1);
		d -= (*((ptr_s)+wid)) << 1;
		d -= *((ptr_s++) + wid + 1);


		c >>= 2;
		d >>= 2;
		sub_aImg.data[i] = sqrtlab[c+256][d+256];
	}

	wid = aImg.cols;
	len = aImg.rows;
	size = wid*len - wid - 1;
	ptr_g += wid + 1;
	for (int i = wid + 1; i < size; i++) {
		int c = 0;
		c -= *((ptr_g)-wid - 1);
		c += *((ptr_g)-wid + 1);
		c -= (*((ptr_g)-1)) << 1;
		c += (*((ptr_g)+1)) << 1;
		c -= *((ptr_g)+wid - 1);
		c += *((ptr_g)+wid + 1);
		int d = 0;
		d += *((ptr_g)-wid - 1);
		d += (*((ptr_g)-wid)) << 1;
		d += *((ptr_g)-wid + 1);
		d -= *((ptr_g)+wid - 1);
		d -= (*((ptr_g)+wid)) << 1;
		d -= *((ptr_g++) + wid + 1);


		c >>= 2;
		d >>= 2;
		aImg.data[i] = sqrtlab[c+256][d+256];
	}
	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i < delta_height; i++)
	{
		for (int j = 0; j < delta_width; j++)
		{
			int total_diff = 0;
			//遍历模板图上的每一个像素
			for (int x = 1; x < sub_height; x++)
			{
				for (int y = 1; y < sub_width; y++)
				{
					//大图上的像素位置
					/*int row_index = i + x;
					int col_index = j + y;*/
					int bigImg_pix = aImg.data[(i + x )* width + (j + y)];
					//模板图上的像素
					int template_pix = sub_aImg.data[x * (sub_width_o)+y];

					total_diff += _abslab[bigImg_pix - template_pix + 256];
				}
			}
			i_min = i*(total_diff < min_diff) + i_min*(total_diff >= min_diff);
			j_min = j*(total_diff < min_diff) + j_min*(total_diff >= min_diff);
			min_diff = total_diff*(total_diff < min_diff) + min_diff*(total_diff >= min_diff);
		}

	}
	*x = i_min;
	*y = j_min;

	return 1;
}


int ustc_SubImgMatch_hist(Mat grayImg, Mat subImg, int* x, int* y) {
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return -1;
	}
	if (firstflag == 0) {
		for (int i = 0; i < 512; i++) {
			for (int j = 0; j < 512; j++) {
				tanlab[i][j] = atan2(i - 256, j - 256)*40.74 + 128;
				float a = (i - 256) *(i - 256) + (j - 256) * (j - 256);
				float b = a;
				unsigned int e = *(unsigned int *)&b;
				e = (e + 0x3f76cf62) >> 1;
				b = *(float *)&e;
				sqrtlab[i][j] = (b + a / b)*0.5;

			}
		}
		firstflag = 1;
	}
	if (absflag == 0) {
		absflag = 1;
		for (int i = 0; i < 512; i++) {
			abslab[i] = abs(i - 256);
		}
	}

	uchar* ptr_g = grayImg.data;
	uchar* ptr_s = subImg.data;
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int hist[256];
	int sub_hist[256];
	int delta_height = height - sub_height;
	int delta_width = width - sub_width;
	int i_min = 0, j_min = 0;
	int min_diff = 0x7fffffff;

	//直方图清零
	for (int i = 0; i < 256; i++) {
		hist[i] = 0;
		sub_hist[i] = 0;
	}

	//计算直方图
	int size = sub_width*sub_height;
	for (int i = 0; i < size; i++) {
		int pixVal = *(ptr_s + i);
		sub_hist[pixVal]++;
	}
	
	for (int i = 0; i < delta_height; i++)
	{
		for (int i = 0; i < 256; i++) {
			hist[i] = 0;
		}


		for (int x = 0; x < sub_height; x++)
		{
			for (int y = 0; y < sub_width; y++)
			{
				
				int bigImg_pix = grayImg.data[(i + x)* width + y];
				hist[bigImg_pix] += 1;
			}
		}


		for (int j = 1; j < delta_width; j++)
		{
			int total_diff = 0;
			for (int x = 0; x < sub_height; x++) {
				int bigImg_pix = grayImg.data[(i + x)* width + j - 1];
				hist[bigImg_pix] -= 1;
				bigImg_pix = grayImg.data[(i + x)* width + j + sub_width - 1];
				hist[bigImg_pix] += 1;
			}
			for (int x = 0; x < 256; x++) {
				total_diff += abslab[hist[x] - sub_hist[x] + 256];
			}
			i_min = i*(total_diff < min_diff) + i_min*(total_diff >= min_diff);
			j_min = j*(total_diff < min_diff) + j_min*(total_diff >= min_diff);
			min_diff = total_diff*(total_diff < min_diff) + min_diff*(total_diff >= min_diff);
		}

	}
	*x = i_min;
	*y = j_min;

	return 1;
}
