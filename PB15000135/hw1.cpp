#include "SubImageMatch.h"
//函数功能：将bgr图像转化成灰度图像
//bgrImg：彩色图，像素排列顺序是bgr
//grayImg：灰度图，单通道
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL
int ustc_ConvertBgr2Gray(Mat bgrImg, Mat& grayImg) {
	if (NULL == bgrImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (bgrImg.channels() != 3 || grayImg.channels() != 1)
	{
		cout << "image's channels doesn't match" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = bgrImg.cols;
	int height = bgrImg.rows;
	if (bgrImg.rows != grayImg.rows || bgrImg.cols != grayImg.cols) {
		cout << "images' size doesn't  match" << endl;
		grayImg.create(height, width, CV_8UC1);
	}
	uchar *p = bgrImg.data;
	uchar *q = grayImg.data;
	uchar *end = q + width*height;
	while (q<end) {
		*(q++) = (uchar)((*(p) * 114 + *(p + 1) * 587 + *(p + 2) * 229) >> 10);
		p += 3;
	}
	return SUB_IMAGE_MATCH_OK;
}

//函数功能：对灰度图像计算直方图
//grayImg：灰度图，单通道
//hist：直方图
//hist_len：直方图的亮度等级，直方图数组的长度
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL
int ustc_CalcHist(Mat grayImg, int* hist, int hist_len) {
	if (NULL == grayImg.data || NULL == hist)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	//直方图清零
	if (hist_len > 256 || hist_len < 0) {
		hist = new int[256];
	}
	for (int i = 0; i < hist_len; i++)
		hist[i] = 0;
	//计算直方图
	uchar*p = grayImg.data;
	uchar*q = p + width*height;
	while (p < q) {
		hist[*p] += 1;
		p++;
	}
	return SUB_IMAGE_MATCH_OK;
}

//函数功能：根据灰度图像计算梯度图像
//grayImg：灰度图，单通道
//gradImg_x：水平方向梯度，浮点类型图像，CV32FC1
//gradImg_y：垂直方向梯度，浮点类型图像，CV32FC1
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL
int ustc_CalcGrad(Mat grayImg, Mat& gradImg_x, Mat& gradImg_y) {
	if (NULL == grayImg.data || gradImg_x.data == NULL || gradImg_y.data == NULL)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (grayImg.channels() != 1 || gradImg_x.channels() != 1 || gradImg_y.channels() != 1)
	{
		cout << "images' channels doesn't match" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int height = grayImg.rows;
	int width = grayImg.cols;
	if (!(height == gradImg_x.rows&&width == gradImg_x.cols))
	{
		cout << "images' size doesn't  match" << endl;
		gradImg_x.create(height, width, CV_32FC1);
	}
	if (!(height == gradImg_y.rows&&width == gradImg_y.cols))
	{
		cout << "images' size doesn't  match" << endl;
		gradImg_y.create(height, width, CV_32FC1);
	}
	int temp0 = (height - 1)*width;
	for (int i = 0; i < width; i++) {
		((float*)gradImg_x.data)[i] = 0;
		((float*)gradImg_x.data)[temp0 + i] = 0;
		((float*)gradImg_y.data)[i] = 0;
		((float*)gradImg_y.data)[temp0 + i] = 0;//对上下边框赋值为0；
	}
	for (int row_i = 1; row_i < height - 1; row_i++)
	{
		int temp_0 = row_i*width;
		((float*)gradImg_x.data)[temp_0] = 0;
		((float*)gradImg_x.data)[temp_0 + width - 1] = 0;
		((float*)gradImg_y.data)[temp_0] = 0;//对1到height-2行的左右边框赋值为0；
		((float*)gradImg_y.data)[temp_0 + width - 1] = 0;
		for (int col_j = 1; col_j < width - 1; col_j++)
		{
			int temp_1 = temp_0 + col_j + 1;
			int temp_2 = temp_0 + col_j - 1;
			int grad_x =
				grayImg.data[temp_1 - width]
				+ (grayImg.data[temp_1] << 1)
				+ grayImg.data[temp_1 + width]
				- grayImg.data[temp_2 - width]
				- (grayImg.data[temp_2] << 1)
				- grayImg.data[temp_2 + width];
			((float*)gradImg_x.data)[temp_0 + col_j] = grad_x;
			int temp_3 = temp_0 - width + col_j;
			int temp_4 = temp_0 + width + col_j;
			int grad_y = -grayImg.data[temp_3 - 1]
				- (grayImg.data[temp_3] << 1)
				- grayImg.data[temp_3 + 1]
				+ grayImg.data[temp_4 - 1]
				+ (grayImg.data[temp_4] << 1)
				+ grayImg.data[temp_4 + 1];
			((float*)gradImg_y.data)[temp_0 + col_j] = grad_y;
		}
	}
	return  SUB_IMAGE_MATCH_OK;
}

//函数功能：对灰度图像进行二值化
//grayImg：灰度图，单通道
//binaryImg：二值图，单通道
//th：二值化阈值，高于此值，255，低于此值0
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL
int ustc_Threshold(Mat grayImg, Mat& binaryImg, int th) {
	if (NULL == grayImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (th < 0 || th>255) {
		cout << "th is wrong" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	if (!(height == binaryImg.rows&&width == binaryImg.cols))
	{
		cout << "images' size doesn't  match" << endl;
		binaryImg.create(height, width, CV_32FC1);
	}
	uchar *p = binaryImg.data;
	uchar *q = grayImg.data;
	uchar *end = p + width*height;
	while (p < end) {
		*(p++) = *(q++) > th ? 255 : 0;
	}
	return SUB_IMAGE_MATCH_OK;
}

//函数功能：利用亮度进行子图匹配
//grayImg：灰度图，单通道
//subImg：模板子图，单通道
//x：最佳匹配子图左上角x坐标
//y：最佳匹配子图左上角y坐标
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL
int ustc_SubImgMatch_gray(Mat grayImg, Mat subImg, int* x, int* y) {
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (grayImg.channels() != 1 && subImg.channels() != 1) {
		cout << "images' channels doesn't match" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	if (width < sub_width || height < sub_height) {
		cout << "images size is wrong";
		return SUB_IMAGE_MATCH_FAIL;
	}
	int end_row = height - sub_height;
	int end_col = height - sub_width;
	int max = 2147483647;
	int total_diff = 0;
	for (int i = 0; i < end_row; i++)
	{
		for (int j = 0; j < end_col; j++)
		{
			//遍历模板图上的每一个像素
			int temp0 = 0;
			for (int ii = 0; ii < sub_height; ii++)
			{
				int temp1 = (ii + i)*width;
				for (int jj = 0; jj < sub_width; jj++)
				{
					//大图上的像素位置
					int bigImg_pix = grayImg.data[temp1 + jj + j];
					//模板图上的像素
					int template_pix = subImg.data[temp0 + jj];
					int temp = bigImg_pix - template_pix;
					total_diff += temp > 0 ? temp : -temp;
				}
				temp0 = temp0 + sub_width;
			}
			//存储当前像素位置的匹配误差
			if (total_diff < max) {
				max = total_diff;
				*x = i;
				*y = j;
			}
			total_diff = 0;
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
int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int* x, int* y) {
	if (NULL == colorImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (colorImg.channels() != 3 && subImg.channels() != 3) {
		cout << "images' channels doesn't match" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = colorImg.cols;
	int height = colorImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	if (width < sub_width || height < sub_height) {
		cout << "images size is wrong";
		return SUB_IMAGE_MATCH_FAIL;
	}
	int end_row = height - sub_height;
	int end_col = width - sub_width;
	int max = 2147483647;
	uchar *p;
	uchar *q;
	int t1, t2, t3;
	for (int i = 0; i < end_row; i++) {
		for (int j = 0; j < end_col; j++) {
			int diff = 0;
			int temp3 = 0;
			for (int ii = 0; ii < sub_height; ii++) {
				int temp4 = (i + ii)*width;
				for (int jj = 0; jj < sub_width; jj++) {
					int temp_0 = 3 * (temp4 + j + jj);
					int temp_1 = 3 * (temp3 + jj);
					p = &colorImg.data[temp_0];
					q = &subImg.data[temp_1];
					t1 = *p - *q;
					t2 = *(p + 1) - *(q + 1);
					t3 = *(p + 2) - *(q + 2);
					diff += (t1 > 0 ? t1 : -t1) + (t2 > 0 ? t2 : -t2) + (t3 > 0 ? t3 : -t3);
				}
				temp3 = temp3 + sub_width;
			}
			if (diff < max) {
				max = diff;
				*x = i;
				*y = j;
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
int ustc_SubImgMatch_hist(Mat grayImg, Mat subImg, int* x, int* y) {
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (grayImg.channels() != 1 && subImg.channels() != 1) {
		cout << "images' channels doesn't match" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	if (width < sub_width || height < sub_height) {
		cout << "images size is wrong";
		return SUB_IMAGE_MATCH_FAIL;
	}
	int end_row = height - sub_height;
	int end_col = width - sub_width;
	int max = 2147483646;
	int sub_Array[256] = { 0 };
	uchar*p = subImg.data;
	uchar*q = p + sub_width*sub_height;
	while (p < q) {
		sub_Array[*p] += 1;
		p++;
	}
	int diff = 0;
	int temp = 0;
	int temp1 = 0;
	for (int i = 0; i < end_row; i++) {
		int gray_Array[256] = { 0 };
		for (int ii = 0; ii < sub_height; ii++) {
			temp1 = (ii + i)*width;
			for (int jj = 0; jj < sub_width; jj++) {
				gray_Array[grayImg.data[temp1 + jj]] += 1;
			}
		}
		for (int k = 0; k < 256; k++) {
			temp = gray_Array[k] - sub_Array[k];
			diff += temp > 0 ? temp : -temp;
		}
		if (diff < max) {
			max = diff;
			*x = i;
			*y = 0;
		}
		diff = 0;
		for (int j = 0; j < end_col; j++) {
			int temp3 = i*width;
			for (int ii = 0; ii < sub_height; ii++) {
				gray_Array[grayImg.data[temp3 + j + sub_width - 1]] += 1;
				gray_Array[grayImg.data[temp3 + j]] -= 1;
				temp3 += width;
			}
			for (int k = 0; k < 256; k++) {
				temp = gray_Array[k] - sub_Array[k];
				diff += temp > 0 ? temp : -temp;
			}
			if (diff < max) {
				max = diff;
				*x = i;
				*y = j;
			}
			diff = 0;
		}
	}
}

//函数功能：利用亮度相关性进行子图匹配
//grayImg：灰度图，单通道
//subImg：模板子图，单通道
//x：最佳匹配子图左上角x坐标
//y：最佳匹配子图左上角y坐标
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL
int ustc_SubImgMatch_corr(Mat grayImg, Mat subImg, int* x, int* y) {
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (grayImg.channels() != 1 && subImg.channels() != 1) {
		cout << "images' channels doesn't match" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	if (width < sub_width || height < sub_height) {
		cout << "images size is wrong";
		return SUB_IMAGE_MATCH_FAIL;
	}
	int end_row = height - sub_height;
	int end_col = width - sub_width;
	float max = 0;
	long long int S = 0;
	int temp0 = 0;
	for (int ii = 0; ii < sub_height; ii++) {
		for (int jj = 0; jj < sub_width; jj++) {
			S += subImg.data[temp0 + jj] * subImg.data[temp0 + jj];
		}//由于S方不变，故先求S方
		temp0 += sub_width;
	}
	for (int i = 0; i < end_row; i++) {
		float diff = 0;
		long long int ST = 0, T = 0;
		for (int ii = 0; ii < sub_height; ii++) {
			int temp1 = (i + ii)*width;
			int temp3 = ii*sub_width;
			for (int jj = 0; jj < sub_width; jj++)
			{
				int temp2 = temp1 + jj;
				T += grayImg.data[temp2] * grayImg.data[temp2];
				ST = ST + grayImg.data[temp2] * subImg.data[temp3 + jj];
			}//每行开始前计算右侧的一块的T方
		}
		max = ST / sqrt(S*T);
	// max = ST*fastInvSqrt(S*T);
		if (diff > max) {
			max = diff;
			*x = i;
			*y = 0;
		}
		ST = 0;
		for (int j = 0; j < end_col; j++) {
			int temp5 = 0;
			for (int ii = 0; ii < sub_height; ii++) {
				int temp4 = (i + ii)*width + j;
				T = T + grayImg.data[temp4 + sub_width - 1] * grayImg.data[temp4 + sub_width - 1] - grayImg.data[temp4] * grayImg.data[temp4];
				for (int jj = 0; jj < sub_width; jj++) {
					ST = ST + grayImg.data[temp4 + jj] * subImg.data[temp5 + jj];
				}
				temp5 += sub_width;
			}
			max = ST / sqrt(S*T);
			// max = ST*fastInvSqrt(S*T);
			if (diff > max) {
				max = diff;
				*x = i;
				*y = j;
			}
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
int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y, Mat& angleImg, Mat& magImg) {
	if (NULL == gradImg_x.data || NULL == gradImg_y.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (gradImg_x.channels() != 1 || gradImg_x.channels() != 1)
	{
		cout << "gradImgs' channels doesn't match" << endl;;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int height = gradImg_x.rows;
	int width = gradImg_x.cols;
	int temp_0 = 0;
	int temp1;
	float temp = 0;
	for (int row_i = 0; row_i < height; row_i++)
	{
		for (int col_j = 0; col_j < width; col_j++)
		{
			temp1 = temp_0 + col_j;
			float grad_x = ((float*)gradImg_x.data)[temp1];
			float grad_y = ((float*)gradImg_y.data)[temp1];
			temp = atan2(grad_y, grad_x);
			((float*)angleImg.data)[temp1] = (temp > 0 ? temp : temp + 6.28294)*57.295;
			((float*)magImg.data)[temp1] = sqrt(grad_y*grad_y + grad_x*grad_x);
		}
		temp_0 += width;
	}
	return SUB_IMAGE_MATCH_OK;
}

//函数功能：利用幅值进行子图匹配
//grayImg：灰度图，单通道
//subImg：模板子图，单通道
//x：最佳匹配子图左上角x坐标
//y：最佳匹配子图左上角y坐标
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL
int ustc_CalcMag(Mat gradImg_x, Mat gradImg_y, Mat& magImg)
{
	if (NULL == gradImg_x.data || NULL == gradImg_y.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (gradImg_x.channels() != 1 || gradImg_x.channels() != 1)
	{
		cout << "gradImgs' channels doesn't match";
		return SUB_IMAGE_MATCH_FAIL;
	}
	int height = gradImg_x.rows;
	int width = gradImg_x.cols;
	int temp_0 = 0;
	int temp1;
	for (int row_i = 0; row_i < height; row_i++)
	{
		for (int col_j = 0; col_j < width; col_j++)
		{
			temp1 = temp_0 + col_j;
			float grad_x = ((float*)gradImg_x.data)[temp1];
			float grad_y = ((float*)gradImg_y.data)[temp1];
			((float*)magImg.data)[temp1] = sqrt(grad_y*grad_y + grad_x*grad_x);
		}
		temp_0 += width;
	}
	return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_mag(Mat grayImg, Mat subImg, int* x, int* y) {
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	unsigned int width = grayImg.cols;
	unsigned int height = grayImg.rows;
	unsigned int sub_width = subImg.cols;
	unsigned int sub_height = subImg.rows;
	unsigned int end_row = height - sub_height;
	unsigned int end_col = width - sub_width;
	if (height < sub_height || width < sub_width) {
		cout << "images' size is wrong" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	Mat grayImgx;
	grayImgx.create(height, width, CV_32FC1);
	Mat grayImgy;
	grayImgy.create(height, width, CV_32FC1);
	Mat grayImgMag;
	grayImgMag.create(height, width, CV_32FC1);
	Mat subImgx;
	subImgx.create(sub_height, sub_width, CV_32FC1);
	Mat subImgy;
	subImgy.create(sub_height, sub_width, CV_32FC1);
	Mat subImgMag;
	subImgMag.create(sub_height, sub_width, CV_32FC1);//建立两匹配图的6个梯度矩阵
	ustc_CalcGrad(grayImg, grayImgx, grayImgy);
	ustc_CalcGrad(subImg, subImgx, subImgy);
	ustc_CalcMag(subImgx, subImgy, subImgMag);
	ustc_CalcMag(grayImgx, grayImgy, grayImgMag);
	unsigned int max = 2147483647;
	unsigned int diff = 0;
	int temp2 = 0;
	unsigned int h = sub_height - 1;
	unsigned int w = sub_width - 1;
	for (unsigned int i = 0; i < end_row; i++)
	{
		for (unsigned int j = 0; j < end_col; j++)
		{
			int temp1 = sub_width;
			for (unsigned int ii = 1; ii < h; ii++)
			{
				unsigned int temp0 = (i + ii)*width;
				for (int jj = 1; jj < w; jj++)
				{
					temp2 = j + jj;
					int t = ((float*)grayImgMag.data)[temp0 + temp2] - ((float*)subImgMag.data)[temp1 + jj];
					diff += (t > 0 ? t : -t);
				}
				temp1 += sub_width;
			}
			if (max > diff)
			{
				max = diff;
				*x = i;
				*y = j;
			}
			diff = 0;
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
int ustc_CalcAng(Mat gradImg_x, Mat gradImg_y, Mat&angleImg)
{
	if (NULL == gradImg_x.data || NULL == gradImg_y.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (gradImg_x.channels() != 1 || gradImg_x.channels() != 1)
	{
		cout << "gradImgs' channels doesn't match";
		return SUB_IMAGE_MATCH_FAIL;
	}
	int height = gradImg_x.rows;
	int width = gradImg_x.cols;
	int temp_0 = 0;
	int temp1;
	int temp = 0;
	for (int row_i = 0; row_i < height; row_i++)
	{
		for (int col_j = 0; col_j < width; col_j++)
		{
			temp1 = temp_0 + col_j;
			float grad_x = ((float*)gradImg_x.data)[temp1];
			float grad_y = ((float*)gradImg_y.data)[temp1];
			temp = atan2(grad_y, grad_x);
			((float*)angleImg.data)[temp1] = (temp > 0 ? temp : temp + 6.28294)*57.295;
		}
		temp_0 += width;
	}
	return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_angle(Mat grayImg, Mat subImg, int* x, int* y) {
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	unsigned int width = grayImg.cols;
	unsigned int height = grayImg.rows;
	unsigned int sub_width = subImg.cols;
	unsigned int sub_height = subImg.rows;
	if (height < sub_height || width < sub_width) {
		cout << "images' size is wrong" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	unsigned int end_row = height - sub_height;
	unsigned int end_col = width - sub_width;
	Mat grayImgx;
	grayImgx.create(height, width, CV_32FC1);
	Mat grayImgy;
	grayImgy.create(height, width, CV_32FC1);
	Mat grayImgAng;
	grayImgAng.create(height, width, CV_32FC1);
	Mat subImgx;
	subImgx.create(sub_height, sub_width, CV_32FC1);
	Mat subImgy;
	subImgy.create(sub_height, sub_width, CV_32FC1);
	Mat subImgAng;
	subImgAng.create(sub_height, sub_width, CV_32FC1);//建立两匹配图的6个梯度矩阵
	ustc_CalcGrad(grayImg, grayImgx, grayImgy);
	ustc_CalcGrad(subImg, subImgx, subImgy);
	ustc_CalcAng(subImgx, subImgy, subImgAng);
	ustc_CalcAng(grayImgx, grayImgy, grayImgAng);
	float max = 2147483647;
	float diff = 0;
	unsigned int temp2 = 0;
	float t;
	unsigned int h = sub_height - 1;
	unsigned int w = sub_width - 1;
	for (unsigned int i = 0; i < end_row; i++)
	{
		for (unsigned int j = 0; j < end_col; j++)
		{
			unsigned int temp1 = sub_width;
			for (unsigned int ii = 1; ii < h; ii++)
			{
				unsigned int temp0 = (i + ii)*width;
				for (unsigned int jj = 1; jj < w; jj++)
				{
					temp2 = j + jj;
					t = ((float*)grayImgAng.data)[temp0 + temp2] - ((float*)subImgAng.data)[temp1 + jj];
					t = (t > 0 ? t : -t);
					diff += (t < 180 ? t : 360 - t);
				}
				temp1 += sub_width;
			}
			if (max > diff)
			{
				max = diff;
				*x = i;
				*y = j;
			}
			diff = 0;
		}
	}
	return SUB_IMAGE_MATCH_OK;
}

