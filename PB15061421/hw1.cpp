
#include"SubImageMatch.h"

int ustc_Bgr2Gray(Mat colorImg, Mat &grayImg)
{
	if (NULL == colorImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int n, b, g, r = 0;
	int width = colorImg.cols;
	int height = colorImg.rows;

	for (int row_i = 0; row_i < height; row_i++)
	{
		for (int col_j = 0; col_j < width; col_j += 1)
		{
			n = row_i * width + col_j;
			b = colorImg.data[3 * n + 0];
			g = colorImg.data[3 * n + 1];
			r = colorImg.data[3 * n + 2];

			int grayVal = (b * 117 + g * 601 + r * 234) >> 10;
			grayImg.data[row_i * width + col_j] = grayVal;
		}
	}

	#ifdef IMG_SHOW
     namedWindow("grayImg", 0);
	imshow("grayImg", grayImg);
	waitKey();
	#endif
	return SUB_IMAGE_MATCH_OK;
}

int ustc_Threshold(Mat grayImg, Mat &binaryImg, int th)
{
	if (NULL == grayImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (th > 255 || th < 0) {
		cout << "th is too large." << endl;
		return SUB_IMAGE_MATCH_FAIL;

	}
	int height = grayImg.rows;
	int width = grayImg.cols;


	int binary_th = th;

	for (int row_i = 0; row_i < height; row_i++)
	{
		int temp0 = row_i * width;
		for (int col_j = 0; col_j < width; col_j++)
		{
			int temp1 = temp0 + col_j;
			int pixVal = grayImg.data[temp1];
			int dstVal = 0;
			dstVal = ((binary_th - pixVal) >> 31) & 1 * 255;
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

int ustc_CalcGrad(Mat grayImg, Mat& gradImg_x, Mat& gradImg_y)
{

	if (NULL == grayImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (grayImg.cols != gradImg_x.cols || grayImg.rows != gradImg_x.rows) {
		cout << "images are not matched." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (grayImg.cols != gradImg_y.cols || grayImg.rows != gradImg_y.rows) {
		cout << "images are not matched." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	gradImg_x.setTo(0);
	gradImg_y.setTo(0);

	//计算x方向梯度图
	for (int row_i = 1; row_i < height - 1; row_i++)
	{
		for (int col_j = 1; col_j < width - 1; col_j += 1)
		{
			int n = (row_i)* width + col_j;
			int grad_x =
				grayImg.data[n - width + 1]
				+ 2 * grayImg.data[n + 1]
				+ grayImg.data[n + width + 1]
				- grayImg.data[n - width - 1]
				- 2 * grayImg.data[n - 1]
				- grayImg.data[n + width - 1];

			((float*)gradImg_x.data)[n] = grad_x;

			int grad_y =
				-grayImg.data[n - width - 1]
				- 2 * grayImg.data[n - width]
				- grayImg.data[n - width + 1]
				+ grayImg.data[n + width - 1]
				+ 2 * grayImg.data[n + width]
				+ grayImg.data[n + width + 1];

			((float*)gradImg_y.data)[n] = grad_y;
		}
	}


	#ifdef IMG_SHOW
	Mat gradImg_x_8U(height, width, CV_8UC1);
	Mat gradImg_y_8U(height, width, CV_8UC1);
	//为了方便观察，直接取绝对值
	for (int row_i = 0; row_i < height; row_i++)
	{
	for (int col_j = 0; col_j < width; col_j += 1)
	{
	int val = ((float*)gradImg_x.data)[row_i * width + col_j];
	gradImg_x_8U.data[row_i * width + col_j] = abs(val);
	}
	}
	for (int row_i = 0; row_i < height; row_i++)
	{
	for (int col_j = 0; col_j < width; col_j += 1)
	{
	int val = ((float*)gradImg_y.data)[row_i * width + col_j];
	gradImg_y_8U.data[row_i * width + col_j] = abs(val);
	}
	}


	namedWindow("gradImg_x_8U", 0);
	imshow("gradImg_x_8U", gradImg_x_8U);
	namedWindow("gradImg_y_8U", 0);
	imshow("gradImg_y_8U", gradImg_x_8U);
	waitKey();
	#endif
	return SUB_IMAGE_MATCH_OK;
}

int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y, Mat& angleImg, Mat& magImg)
{
	
	if (NULL == gradImg_x.data || NULL == gradImg_y.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (gradImg_x.cols != angleImg.cols || gradImg_x.rows != angleImg.rows) {
		cout << "images are not matched." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (gradImg_x.cols != magImg.cols || gradImg_x.rows != magImg.rows) {
		cout << "images are not matched." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	float tg[180] = {0};
	for (int i = 0; i < 89; i++) {
		tg[i] = tan(i*3.14159f / 180);
	}
	for (int i = 91; i < 180; i++) {
		tg[i] = tan(i*3.14159f / 180);
	}
	int width = gradImg_x.cols;
	int height = gradImg_x.rows;
	angleImg.setTo(0);
	magImg.setTo(0);
	//计算角度图
		for (int row_i = 1; row_i < height - 1; row_i++)
		{
			for (int col_j = 1; col_j < width - 1; col_j += 1)
			{
				float grad_x = ((float*)gradImg_x.data)[row_i * width + col_j];
				float grad_y = ((float*)gradImg_y.data)[row_i * width + col_j];

				float k = grad_y / grad_x;
				int i = 0;
				if (k = 0) {
					if (grad_x < 0)
						i = 180;
					else
						i = 0;
				}
				else if(k>0){
					for (i = 0; i < 91; i++) {
						if (i = 90)
							break;
						if (k > tg[i])
							i++;
						else if (k <= tg[i])
							break;
					}
				}
				else if (k < 0) {
					for (i = 91; i <= 180; i++)
						if (k > tg[i])
							i++;
						else if (k <= tg[i])
							break;
				}
				if (grad_y < 0) {
					i = i + 180;
				}
				int angle = i;
				//float angle = atan2(grad_y, grad_x);
				float mag = sqrt(grad_x*grad_x + grad_y*grad_y);
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
	for (int col_j = 0; col_j < width; col_j += 1)
	{
	float angle = ((float*)angleImg.data)[row_i * width + col_j];
	//为了能在8U上显示，缩小到0-180之间
	angle /= 2;
	angleImg_8U.data[row_i * width + col_j] = angle;
	}
	}
	Mat magImg_8U(height, width, CV_8UC1);
	for (int row_i = 0; row_i < height; row_i++)
	{
	for (int col_j = 0; col_j < width; col_j += 1)
	{
	int val = ((float*)magImg.data)[row_i * width + col_j];
	magImg_8U.data[row_i * width + col_j] = val;
	}
	}
	namedWindow("angleImg", 0);
	imshow("angleImg", angleImg_8U);
	namedWindow("magImg_8U", 0);
	imshow("magImg_8U", magImg_8U);
	waitKey();
	#endif
	return SUB_IMAGE_MATCH_OK;
}

int ustc_CalcHist(Mat grayImg, int* hist, int hist_len)
{
	if (NULL == grayImg.data || NULL == hist)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (hist_len > 256) {
		cout << "Histlen is too large." << endl;
		return SUB_IMAGE_MATCH_FAIL;
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
	return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_gray(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (subImg.rows > grayImg.rows || subImg.cols > grayImg.cols) {
		cout << "subimage is too large." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
//	int start = clock();
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int total_diff = 0;
	int row_index;
	int col_index;
	int bigImg_pix;
	int template_pix;

	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg=Mat::zeros(height - sub_height, width - sub_width, CV_32FC1);
	searchImg.setTo(FLT_MAX);
	//匹配误差初始化
	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int row = 0; row < height - sub_height; row++)

	{
		for (int col = 0; col < width - sub_width; col++)
		{
			total_diff = 0;
			//遍历模板图上的每一个像素
			for (int sub_row = 0; sub_row < sub_height; sub_row++)
			{
                 row_index = row + sub_row;
				for (int sub_col = 0; sub_col < sub_width; sub_col++)
				{
					//大图上的像素位置
					
					 col_index = col + sub_col;
					 bigImg_pix = grayImg.data[row_index * width + col_index];
					//模板图上的像素
					 template_pix = subImg.data[sub_row * sub_width + sub_col];
					total_diff += (bigImg_pix - template_pix)*(2 *!(((bigImg_pix - template_pix) >> 31) & 1)- 1);
				}
				//存储当前像素位置的匹配误差
				((float*)searchImg.data)[row * (width - sub_width) + col] = total_diff;
			}
			
		}
	}

	float tmp = 1000; *x = 0; *y = 0;
	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++) {
			if (((float*)searchImg.data)[i*(width - sub_width)+j] < tmp){
				tmp = ((float*)searchImg.data)[i*(width - sub_width) + j];
				*x = j;
				*y = i;
			}
		
	}

}
	//int end = clock();
	//cout << (end - start) << endl;
	return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int* x, int* y) {
	if (NULL == colorImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (subImg.rows > colorImg.rows || subImg.cols > colorImg.cols) {
		cout << "subimage is too large." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
//	int start = clock();
	int width = colorImg.cols;
	int height = colorImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg = Mat::zeros(height - sub_height, width - sub_width, CV_32FC3);
	searchImg.setTo(FLT_MAX);
	//匹配误差初始化
	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int row = 0; row < height - sub_height; row++)

	{
		for (int col = 0; col < width - sub_width; col++)
		{
			int total_diff_b= 0;
			int total_diff_g = 0;
			int total_diff_r = 0;
			//遍历模板图上的每一个像素
			for (int sub_row = 0; sub_row < sub_height; sub_row++)
			{
                    int row_index = row + sub_row;
				for (int sub_col = 0; sub_col < sub_width; sub_col++)
				{
					//大图上的像素位置
					
					int col_index = col + sub_col;
					int placeindex = 3 * (row_index * width + col_index);
					int sub_place = 3 * (sub_row*sub_width + sub_col);
					int bigImg_pix_b = colorImg.data[placeindex];
					//模板图上的像素
					int template_pix_b= subImg.data[sub_place];
					total_diff_b += (bigImg_pix_b - template_pix_b)*(2 * !(((bigImg_pix_b - template_pix_b) >> 31) & 1) - 1);
					int bigImg_pix_g = colorImg.data[placeindex +1];
					//模板图上的像素
					int template_pix_g = subImg.data[sub_place +1];
					total_diff_g += (bigImg_pix_g - template_pix_g)*(2 * !(((bigImg_pix_g - template_pix_g) >> 31) & 1) - 1);
					int bigImg_pix_r = colorImg.data[placeindex +2];
					//模板图上的像素
					int template_pix_r = subImg.data[sub_place +2];
					total_diff_r += (bigImg_pix_r - template_pix_r)*(2 * !(((bigImg_pix_r - template_pix_r) >> 31) & 1) - 1);

				}
				//存储当前像素位置的匹配误差
				((float*)searchImg.data)[3*(row * (width - sub_width) + col)] = total_diff_b;
				((float*)searchImg.data)[3 * (row * (width - sub_width) + col)+1] = total_diff_g;
				((float*)searchImg.data)[3 * (row * (width - sub_width) + col)+2] = total_diff_r;
			}

		}
	}

	float tmp = 1000; *x = 0; *y = 0;
	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++) {
			float bgrplus= ((float*)searchImg.data)[3*(i*(width - sub_width) + j)]
				+ ((float*)searchImg.data)[3*(i*(width - sub_width) + j)+1]
				+ ((float*)searchImg.data)[3*(i*(width - sub_width) + j)+2];
			if (bgrplus< tmp) {
				tmp = bgrplus;
				*x = j;
				*y = i;
			}

		}

	}
        //int end = clock();
	//cout << (end - start) << endl;
	return SUB_IMAGE_MATCH_OK;

}

int ustc_SubImgMatch_corr(Mat grayImg, Mat subImg, int* x, int* y) {
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (subImg.rows > grayImg.rows || subImg.cols > grayImg.cols) {
		cout << "subimage is too large." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	//int start = clock();
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	//该图用于记录每一个像素位置的R(i,j)
	Mat searchImg = Mat::zeros(height - sub_height, width - sub_width, CV_32FC1);
	searchImg.setTo(FLT_MAX);
	//匹配误差初始化
	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int row = 0; row < height - sub_height; row++)

	{
		for (int col = 0; col < width - sub_width; col++)
		{
			float R = 0;
			float ST = 0;
			float SS = 0;
			float TT = 0;
			//遍历模板图上的每一个像素
			for (int sub_row = 0; sub_row < sub_height; sub_row++)
			{
				for (int sub_col = 0; sub_col < sub_width; sub_col++)
				{
					//大图上的像素位置
					int row_index = row + sub_row;
					int col_index = col + sub_col;
					int bigImg_pix = grayImg.data[row_index * width + col_index];
					//模板图上的像素
					int template_pix = subImg.data[sub_row * sub_width + sub_col];
					ST += bigImg_pix*template_pix;
					TT += bigImg_pix*bigImg_pix;
					SS += template_pix*template_pix;
				}
				R = ST / sqrt(TT*SS);
				//存储当前像素位置的匹配误差
				((float*)searchImg.data)[row * (width - sub_width) + col] = R;
			}

		}
	}

	float tmp = 0.8; *x = 0; *y = 0;
	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++) {
			if (((float*)searchImg.data)[i*(width - sub_width) + j] > tmp) {
				tmp = ((float*)searchImg.data)[i*(width - sub_width) + j];
				*x = j;
				*y = i;
			}

		}

	}
	
	//int end = clock();
	//cout << (end - start) << endl;
	return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_angle(Mat grayImg, Mat subImg, int* x, int* y) {
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (subImg.rows > grayImg.rows || subImg.cols > grayImg.cols) {
		cout << "subimage is too large." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	//int start = clock();
	int sub_height = subImg.rows;
	int sub_width = subImg.cols;
	Mat sub_gradImg_x(sub_height, sub_width, CV_32FC1);
	Mat sub_gradImg_y(sub_height, sub_width, CV_32FC1);
	Mat sub_angleImg(sub_height, sub_width, CV_32FC1);
	Mat sub_magImg(sub_height, sub_width, CV_32FC1);
	ustc_CalcGrad(subImg, sub_gradImg_x, sub_gradImg_y);
	ustc_CalcAngleMag(sub_gradImg_x, sub_gradImg_y, sub_angleImg, sub_magImg);
	
	int height = grayImg.rows;
	int width = grayImg.cols;
	Mat gradImg_x(height, width, CV_32FC1);
	Mat gradImg_y(height, width, CV_32FC1);
	Mat angleImg(height, width, CV_32FC1);
	Mat magImg(height, width, CV_32FC1);
	ustc_CalcGrad(grayImg, gradImg_x, gradImg_y);
	ustc_CalcAngleMag(gradImg_x, gradImg_y,angleImg,magImg);
	Mat searchImg(height - sub_height, width - sub_width, CV_32FC1);
	searchImg.setTo(FLT_MAX);
	for (int row = 1; row < height - sub_height - 1; row++)

	{
		for (int col = 1; col < width - sub_width - 1; col++)
		{
			float total_diff = 0;
			//遍历模板图上的每一个像素
			for (int sub_row = 1; sub_row < sub_height - 1; sub_row++)
			{
				for (int sub_col = 1; sub_col < sub_width - 1; sub_col++)
				{
					//大图上的像素位置
					int row_index = row + sub_row;
					int col_index = col + sub_col;
					int bigImg_pix = ((float*)angleImg.data)[row_index * width + col_index];
					//模板图上的像素
					int template_pix = ((float*)sub_angleImg.data)[sub_row * sub_width + sub_col];
					total_diff += (bigImg_pix - template_pix)*(2 * !(((bigImg_pix - template_pix) >> 31) & 1) - 1);
				}
				//存储当前像素位置的匹配误差
				((float*)searchImg.data)[row * (width - sub_width) + col] = total_diff;
			}

		}
	}

	float tmp = 100; *x = 0; *y = 0;
	for (int i = 1; i < height - sub_height - 1; i++)
	{
		for (int j = 1; j < width - sub_width - 1; j++) {
			if (((float*)searchImg.data)[i*(width - sub_width) + j] < tmp) {
				tmp = ((float*)searchImg.data)[i*(width - sub_width) + j];
				*x = j;
				*y = i;
			}

		}

	}
//	int end = clock();
	//cout << end - start << endl;
	return SUB_IMAGE_MATCH_OK;
    
	
}

int ustc_SubImgMatch_mag(Mat grayImg, Mat subImg, int* x, int* y) {
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (subImg.rows > grayImg.rows || subImg.cols > grayImg.cols) {
		cout << "subimage is too large." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	//int start = clock();
	int sub_height = subImg.rows;
	int sub_width = subImg.cols;
	Mat sub_gradImg_x(sub_height, sub_width, CV_32FC1);
	Mat sub_gradImg_y(sub_height, sub_width, CV_32FC1);
	Mat sub_angleImg(sub_height, sub_width, CV_32FC1);
	Mat sub_magImg(sub_height, sub_width, CV_32FC1);
	ustc_CalcGrad(subImg, sub_gradImg_x, sub_gradImg_y);
	ustc_CalcAngleMag(sub_gradImg_x, sub_gradImg_y, sub_angleImg, sub_magImg);
	int height = grayImg.rows;
	int width = grayImg.cols;
	Mat gradImg_x(height, width, CV_32FC1);
	Mat gradImg_y(height, width, CV_32FC1);
	Mat angleImg(height, width, CV_32FC1);
	Mat magImg(height, width, CV_32FC1);

	ustc_CalcGrad(grayImg, gradImg_x, gradImg_y);
	ustc_CalcAngleMag(gradImg_x, gradImg_y, angleImg, magImg);
	Mat searchImg(height - sub_height, width - sub_width, CV_32FC1);
	searchImg.setTo(FLT_MAX);
	for (int row = 1; row < height - sub_height-1; row++)

	{
		for (int col = 1; col < width - sub_width-1; col++)
		{
			float total_diff = 0;
			//遍历模板图上的每一个像素
			for (int sub_row = 1; sub_row < sub_height-1; sub_row++)
			{
				for (int sub_col = 1; sub_col < sub_width-1; sub_col++)
				{
					//大图上的像素位置
					int row_index = row + sub_row;
					int col_index = col + sub_col;
					float bigImg_pix = ((float*)magImg.data)[row_index * width + col_index];
					//模板图上的像素
					float template_pix = ((float*)sub_magImg.data)[sub_row * sub_width + sub_col];
					int tmpdiffer =(int)( bigImg_pix - template_pix);
					int antisign = !((tmpdiffer >> 31) & 1);
					total_diff += (bigImg_pix - template_pix)*(2 * antisign - 1);
				}
				//存储当前像素位置的匹配误差
				((float*)searchImg.data)[row * (width - sub_width) + col] = total_diff;
			}

		}
	}

	float tmp = 10; *x = 0; *y = 0;
	for (int i = 1; i < height - sub_height-1; i++)
	{
		for (int j = 1; j < width - sub_width-1; j++) {
			if (((float*)searchImg.data)[i*(width - sub_width) + j] < tmp) {
				tmp = ((float*)searchImg.data)[i*(width - sub_width) + j];
				*x = j;
				*y = i;
			}

		}

	}
	//int end = clock();
	//cout << end - start << endl;
	return SUB_IMAGE_MATCH_OK;

}

int ustc_SubImgMatch_hist(Mat grayImg, Mat subImg, int* x, int* y) {
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (subImg.rows > grayImg.rows || subImg.cols > grayImg.cols) {
		cout << "subimage is too large." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	//int start = clock();
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int hist_len = 256;
	int sub_hist[256] = { 0 };
	int tmp_hist[256] = { 0 };
	ustc_CalcHist(subImg, sub_hist, 256);
	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg = Mat::zeros(height - sub_height, width - sub_width, CV_32FC1);
	searchImg.setTo(FLT_MAX);
	//匹配误差初始化
	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int row = 0; row < height - sub_height; row++)

	{
		for (int col = 0; col < width - sub_width; col++)
		{
			int total_diff = 0;
			//直方图清零
			for (int i = 0; i < hist_len; i++)
			{
				tmp_hist[i] = 0;
			}
			//计算直方图
			for (int row_i = 0; row_i < sub_height; row_i++)
			{
				for (int col_j = 0; col_j < sub_width; col_j += 1)
				{
					int rowindex = row + row_i;
					int colindex = col + col_j;
					int pixVal = grayImg.data[rowindex * width + colindex];
					tmp_hist[pixVal]++;
				}
			}
			for (int num = 0; num < hist_len; num++) {
				int subpix = sub_hist[num];
				int tmppix = tmp_hist[num];
				int tmpdiffer = subpix - tmppix;
				int antisign = !((tmpdiffer >> 31) & 1);
				total_diff += (subpix - tmppix)*(2 * antisign - 1);
			}
			((float*)searchImg.data)[row * (width - sub_width) + col] = total_diff;
		}

	}

	float tmp = 1000; *x = 0; *y = 0;
	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++) {
			if (((float*)searchImg.data)[i*(width - sub_width) + j] < tmp) {
				tmp = ((float*)searchImg.data)[i*(width - sub_width) + j];
				*x = j;
				*y = i;
			}

		}


	}
	//int end = clock();
	//cout << end - start << endl;
	return SUB_IMAGE_MATCH_OK;
}






