#include"SubImageMatch.h"

int ustc_ConvertBgr2Gray(Mat bgrImg, Mat& grayImg) {
	int i, psize;
	uchar *Oimagep, *temp;

	if (NULL == grayImg.data )
	{
		printf("image is NULL.");
		return SUB_IMAGE_MATCH_FAIL;
	}

	temp = bgrImg.data;
	Oimagep = grayImg.data;
	 psize = bgrImg.rows * bgrImg.cols;

	for (i = 0; i < psize; i++) {
		*(Oimagep + i)= (*temp * 15 + *(temp + 1) * 75 + *(temp + 2) * 38) >> 7;
		temp = temp + 3;
	}
	return SUB_IMAGE_MATCH_OK;
}//函数功能：将bgr图像转化成灰度图像

int ustc_CalcGrad(Mat grayImg, Mat& gradImg_x, Mat& gradImg_y) {
	int i, j, row, col;
	uchar *grayImgp;
	float *gradImg_xp = (float*)gradImg_x.data;
	float *gradImg_yp = (float*)gradImg_y.data;

	//防御
	if (NULL == grayImg.data || NULL == gradImg_x.data || NULL == gradImg_y.data)
	{
		printf("image is NULL.");
		return SUB_IMAGE_MATCH_FAIL;
	}

	grayImgp = grayImg.data;
	row = grayImg.rows;
	col = grayImg.cols;
	for (i = 1; i < row - 1; i++) {
		for (j = 1; j < col - 1; j++) {
			gradImg_xp[i*row + j] = grayImgp[(i - 1)*row + j + 1] + grayImgp[i*row + j + 1] * 2 + grayImgp[(i + 1)*row + j + 1] - grayImgp[(i - 1)*row + j - 1] - grayImgp[i*row + j - 1] * 2 - grayImgp[(i + 1)*row + j - 1];
			gradImg_yp[i*row + j] = grayImgp[(i + 1)*row + j - 1] + grayImgp[(i + 1)*row + j] * 2 + grayImgp[(i + 1)*row + j + 1] - grayImgp[(i - 1)*row + j - 1] - grayImgp[(i - 1)*row + j] * 2 - grayImgp[(i - 1)*row + j + 1];
		}
	}
	return SUB_IMAGE_MATCH_OK;
}//函数功能：根据灰度图像计算梯度图像

int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y, Mat& angleImg, Mat& magImg) {
	if (NULL == angleImg.data || NULL == gradImg_x.data || NULL == gradImg_y.data||NULL==magImg.data)
	{
		printf("image is NULL.");
		return SUB_IMAGE_MATCH_FAIL;
	}

	static float atan_table[231] = {
		-89.50178839, -89.49741833,	-89.49297094,	-89.48844413,	-89.48383577,	-89.47914363,	-89.47436539,	-89.46949868,	-89.46454101,	-89.45948981,	-89.45434241,	-89.44909602,	-89.44374777,	-89.43829467,	-89.43273359,	-89.4270613,	-89.42127443,	-89.41536948,	-89.40934279,	-89.40319055,	-89.39690881,	-89.39049342,	-89.38394009,	-89.37724431,	-89.37040139,	-89.36340642,	-89.35625429,	-89.34893962,	-89.34145682,	-89.33380003,	-89.3259631, -89.31793961,	-89.3097228,	-89.30130562,	-89.29268063,	-89.28384005,	-89.2747757,	-89.26547897,	-89.2559408,	-89.24615167,	-89.23610154,	-89.22577984,	-89.2151754,	-89.20427645,	-89.19307054,	-89.18154454,	-89.16968451,	-89.15747574,	-89.1449026,	-89.13194855,	-89.118596,	-89.10482629,	-89.09061955,
		-89.07595465,	-89.06080905,	-89.04515875,	-89.02897807,	-89.0122396,	-88.99491399,	-88.97696981,	-88.95837332,	-88.93908831,	-88.91907581,	-88.89829388,	-88.87669729,	-88.85423716,	-88.83086067,	-88.80651058,	-88.78112476,	-88.75463573,	-88.72696998,	-88.69804733,	-88.66778015,	-88.63607247,	-88.60281897,	-88.56790382,	-88.53119929,	-88.49256424,	-88.4518423,	-88.40885973,	-88.36342296,	-88.31531568,	-88.26429541,	-88.21008939,	-88.15238973,	-88.09084757,	-88.02506599,	-87.95459151,	-87.8789036,	-87.79740184,	-87.70938996,	-87.61405597,	-87.51044708,	-87.3974378,	-87.27368901,	-87.13759477,	-86.9872125,	-86.82016988,	-86.63353934,	-86.42366563,	-86.18592517,	-85.91438322,	-85.60129465,	-85.23635831,	-84.80557109,	-84.28940686,
		-83.65980825,	-82.87498365,	-81.86989765,	-80.53767779,	-78.69006753,	-75.96375653,	-71.56505118,	-63.43494882,	-45,	  0,	45, 63.43494882, 71.56505118,	     75.96375653,	78.69006753,	80.53767779,	81.86989765,	82.87498365,	83.65980825,	84.28940686,	84.80557109,	85.23635831,	85.60129465,	85.91438322,	86.18592517,	86.42366563,	86.63353934,	86.82016988,	86.9872125,	87.13759477,	87.27368901,	87.3974378,	87.51044708,	87.61405597,	87.70938996,	87.79740184,	87.8789036, 87.95459151,	88.02506599,	88.09084757,	88.15238973,	88.21008939,	88.26429541,	88.31531568,	88.36342296,	88.40885973,	88.4518423,	88.49256424,	88.53119929,	88.56790382,	88.60281897,	88.63607247,	88.66778015,	88.69804733,	88.72696998,	88.75463573,
		88.78112476,	88.80651058,	88.83086067,	88.85423716,	88.87669729,	88.89829388,	88.91907581,	88.93908831,	88.95837332,	88.97696981,	88.99491399,	89.0122396,	89.02897807,	89.04515875,	89.06080905,	89.07595465,	89.09061955,	89.10482629,	89.118596,	89.13194855,	89.1449026,	89.15747574,	89.16968451,	89.18154454,	89.19307054,	89.2042764,	89.2151754, 89.22577984,	89.23610154,	89.24615167,	89.2559408,	89.26547897,	89.2747757,	89.28384005,	89.29268063,	89.30130562,	89.3097228,	89.31793961,	89.3259631,	89.33380003,	89.34145682,	89.34893962,	89.35625429,	89.36340642,	89.37040139,	89.37724431,	89.38394009,	89.39049342,	89.39690881,	89.40319055,	89.40934279,	89.41536948,	89.42127443,	89.4270613,	89.43273359,	89.43829467,
		89.44374777,	89.44909602,	89.45434241,	89.45948981,	89.46454101,	89.46949868,	89.47436539,	89.47914363,	89.48383577,	89.48844413,	89.49297094,	89.49741833,	89.50178839
	};//最后设为静态数组



	int i, psize;
	float x, y, tg, Theta;
	float *gradImg_xp = (float*)gradImg_x.data;
	float *gradImg_yp = (float*)gradImg_y.data;
	float *angleImgp = (float*)angleImg.data;
	float *magImgp = (float*)magImg.data;
	psize = gradImg_x.rows * gradImg_x.cols;
	
	for (i = 0; i < psize; i++) {
		y = gradImg_yp[i];
		x = gradImg_xp[i];
		
		if (x && y) {
			tg = y / x;
		}
		else {
			tg = 0;
		}
		
		magImgp[i] = FastSqrt(x*x + y*y);// 计算幅值
		Theta = (atan_table[(int)(tg)+1+115] - atan_table[(int)(tg)+115])*(tg - (int)tg) + atan_table[(int)(tg)+115];//计算辐角
		
		if (x && y) {
			angleImgp[i] = Theta;
		}
		else if (x && ! y) {
			angleImgp[i] = Theta + 360;
		}
		else {
			angleImgp[i] = Theta + 180;
		}
}

	return SUB_IMAGE_MATCH_OK;
}//函数功能：根据灰度图像计算角度和幅度图

int ustc_Threshold(Mat grayImg, Mat& binaryImg, int th) {
	int i, psize;
	uchar *grayImgp = grayImg.data;
	uchar *binaryImgp = binaryImg.data;

	//防御
	if (NULL == grayImg.data || NULL == binaryImg.data){
		printf("image is NULL.");
		return SUB_IMAGE_MATCH_FAIL;
	}

	psize = grayImg.rows * grayImg.cols;

	for (i = 0; i < psize; i++) {
		binaryImgp[i] = (grayImgp[i] >= th) * 255;
	}
	return 1;
}//函数功能：对灰度图像进行二值化

int ustc_CalcHist(Mat grayImg, int* hist, int hist_len) {
	int i, psize;
	uchar* grayImgp = grayImg.data;

	//防御
	if (NULL == grayImg.data || NULL == hist)
	{
		printf("image is NULL.");
		return -1;
	}

	//直方图清零
	for (int i = 0; i < hist_len; i++)
	{
		hist[i] = 0;
	}
	//计算直方图
	psize = grayImg.rows * grayImg.cols;
	for (i = 0; i < psize; i++) {
		if (grayImgp[i] <= hist_len) {
			hist[grayImgp[i]]++;
		}
	}
	return SUB_IMAGE_MATCH_OK;
}//函数功能：对灰度图像计算直方图

int ustc_SubImgMatch_gray(Mat grayImg, Mat subImg, int* x, int* y) {
	if (NULL == grayImg.data || NULL == subImg.data) {
		printf("img is NULL.");
		return SUB_IMAGE_MATCH_FAIL;
	}
	
	int row = grayImg.rows;
	int col = grayImg.cols;
	int sub_row = subImg.rows;
	int sub_col = subImg.cols;

	if (sub_row > row || sub_col > col) {
		printf("subimage fail");
		return SUB_IMAGE_MATCH_FAIL;
	}

	int min_diff = 0x4fffffff;
	int bigImg_pix;
	int template_pix;
	int total_diff = 0;
	int yy;
	int big_temp;
	int sub_temp;
	uchar *big_p=grayImg.data, *sub_p=subImg.data;
	//遍历大图的每一个像素
	for (int i = row - sub_row; i>=0 ; --i) {
		for (int j = col - sub_col; j>=0 ; --j) {
			total_diff = 0;
			//遍历模板上的每一个像素


			for (int xx = 0; xx<sub_row; ++xx) {

				yy = 0;
				for (; yy < sub_col - 4; yy += 4) {
					//大图上的像素位置
					//row_index = i + xx;
					//col_index = j + yy;
					big_temp = (i+xx)*col + j + yy;
					sub_temp = xx*sub_col + yy;
					bigImg_pix = big_p[big_temp];
					//模板上的像素位置
					template_pix = sub_p[ sub_temp];

					total_diff += ((bigImg_pix - template_pix) ^ ((bigImg_pix - template_pix) >> 31)) - ((bigImg_pix - template_pix) >> 31);


					bigImg_pix = big_p[big_temp+ 1];
					//模板上的像素位置
					template_pix = sub_p[sub_temp + 1];

					total_diff += ((bigImg_pix - template_pix) ^ ((bigImg_pix - template_pix) >> 31)) - ((bigImg_pix - template_pix) >> 31);


					bigImg_pix = big_p[big_temp+ 2];
					//模板上的像素位置
					template_pix = sub_p[sub_temp + 2];

					total_diff += ((bigImg_pix - template_pix) ^ ((bigImg_pix - template_pix) >> 31)) - ((bigImg_pix - template_pix) >> 31);


					bigImg_pix = big_p[big_temp + 3];
					//模板上的像素位置
					template_pix = sub_p[sub_temp + 3];

					total_diff += ((bigImg_pix - template_pix) ^ ((bigImg_pix - template_pix) >> 31)) - ((bigImg_pix - template_pix) >> 31);

				}
				for (; yy < sub_col; yy += 1) {
					//大图上的像素位置
					

					bigImg_pix = big_p[(i + xx)*col + j+yy];
					//模板上的像素位置
					template_pix = sub_p[xx*sub_col + yy];

					total_diff += ((bigImg_pix - template_pix) ^ ((bigImg_pix - template_pix) >> 31)) - ((bigImg_pix - template_pix) >> 31);

				}
				//比较匹配误差
				if (total_diff < min_diff) {
					*x = i;
					*y = j;
					min_diff = total_diff;
				}
			}
		}
	}
	return SUB_IMAGE_MATCH_OK;
}//函数功能：利用亮度进行子图匹配

int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int* x, int* y) {
	if (NULL == colorImg.data || NULL == subImg.data) {
		printf("img is NULL.");
		return SUB_IMAGE_MATCH_FAIL;
	}
	int xx, yy, i, j;
	int row = colorImg.rows;
	int col = colorImg.cols;
	int sub_row = subImg.rows;
	int sub_col = subImg.cols;

	if (sub_row > row || sub_col > col) {
		printf("subimage fail");
		return SUB_IMAGE_MATCH_FAIL;
	}

	int min_diff = 0x4fffffff;
	int big_b, big_g,big_r, big_bgr;
	int sub_b,sub_g, sub_r, sub_bgr;
	int total_diff;
	int big_temp;
	int sub_temp;
	uchar* sub_p, *big_p;
	sub_p = subImg.data;
	big_p = colorImg.data;
	//遍历大图的每一个像素
	for ( i = row - sub_row; i >= 0; --i) {

		for (j = col - sub_col; j>= 0; --j) {
			total_diff = 0;
			//遍历模板上的每一个像素

			for (xx = 0; xx < sub_row; ++xx) {
				yy = 0;
				for (; yy < sub_col- 4; yy+=4) {
					//大图上的像素位置
					//row_index = i + xx;
					//col_index = j + yy;

					
					
                    big_temp = 3 * ((i+xx)*col + j+yy);

                    sub_temp = 3 * (xx*sub_col + yy);

                   
					sub_b = sub_p[sub_temp + 0];
					sub_g = sub_p[sub_temp + 1];
					sub_r = sub_p[sub_temp + 2];

					big_b = big_p[big_temp + 0];
					big_g = big_p[big_temp + 1];
					big_r = big_p[big_temp + 2];
		
					total_diff += ((((big_b - sub_b) ^ ((big_b - sub_b) >> 31)) - ((big_b - sub_b) >> 31)) + (((big_g - sub_g) ^ ((big_g - sub_g) >> 31)) - ((big_g - sub_g) >> 31)) + (((big_r - sub_r) ^ ((big_r - sub_r) >> 31)) - ((big_r - sub_r) >> 31)));
		
					sub_b = sub_p[sub_temp + 3];
					sub_g = sub_p[sub_temp + 4];
					sub_r = sub_p[sub_temp + 5];

					big_b = big_p[big_temp + 3];
					big_g = big_p[big_temp + 4];
					big_r = big_p[big_temp + 5];

					total_diff += ((((big_b - sub_b) ^ ((big_b - sub_b) >> 31)) - ((big_b - sub_b) >> 31)) + (((big_g - sub_g) ^ ((big_g - sub_g) >> 31)) - ((big_g - sub_g) >> 31)) + (((big_r - sub_r) ^ ((big_r - sub_r) >> 31)) - ((big_r - sub_r) >> 31)));

					sub_b = sub_p[sub_temp + 6];
					sub_g = sub_p[sub_temp + 7];
					sub_r = sub_p[sub_temp + 8];

					big_b = big_p[big_temp + 6];
					big_g = big_p[big_temp + 7];
					big_r = big_p[big_temp + 8];

					total_diff += ((((big_b - sub_b) ^ ((big_b - sub_b) >> 31)) - ((big_b - sub_b) >> 31)) + (((big_g - sub_g) ^ ((big_g - sub_g) >> 31)) - ((big_g - sub_g) >> 31)) + (((big_r - sub_r) ^ ((big_r - sub_r) >> 31)) - ((big_r - sub_r) >> 31)));

					sub_b = sub_p[sub_temp + 9];
					sub_g = sub_p[sub_temp + 10];
					sub_r = sub_p[sub_temp + 11];

					big_b = big_p[big_temp + 9];
					big_g = big_p[big_temp + 10];
					big_r = big_p[big_temp + 11];

					total_diff += ((((big_b - sub_b) ^ ((big_b - sub_b) >> 31)) - ((big_b - sub_b) >> 31)) + (((big_g - sub_g) ^ ((big_g - sub_g) >> 31)) - ((big_g - sub_g) >> 31)) + (((big_r - sub_r) ^ ((big_r - sub_r) >> 31)) - ((big_r - sub_r) >> 31)));

					for (; yy < sub_col ; yy += 1) {
						//大图上的像素位置
						//row_index = i + xx;
						//col_index = j + yy;



						big_temp = 3 * ((i + xx)*col + j + yy);

						sub_temp = 3 * (xx*sub_col + yy);


						sub_b = sub_p[sub_temp + 0];
						sub_g = sub_p[sub_temp + 1];
						sub_r = sub_p[sub_temp + 2];

						big_b = big_p[big_temp + 0];
						big_g = big_p[big_temp + 1];
						big_r = big_p[big_temp + 2];

						total_diff += ((((big_b - sub_b) ^ ((big_b - sub_b) >> 31)) - ((big_b - sub_b) >> 31)) + (((big_g - sub_g) ^ ((big_g - sub_g) >> 31)) - ((big_g - sub_g) >> 31)) + (((big_r - sub_r) ^ ((big_r - sub_r) >> 31)) - ((big_r - sub_r) >> 31)));
					}
				}
			}
			//比较匹配误差
			if (total_diff < min_diff) {
				*x = i;
				*y = j;
				min_diff = total_diff;
			}
		}
	}
	return SUB_IMAGE_MATCH_OK;
}//函数功能：利用色彩进行子图匹配

int ustc_SubImgMatch_corr(Mat grayImg, Mat subImg, int* x, int* y) {
	if (NULL == grayImg.data || NULL == subImg.data) {
		printf("img is NULL.");
		return SUB_IMAGE_MATCH_FAIL;
	}
	
	int row = grayImg.rows;
	int col = grayImg.cols;
	int sub_row = subImg.rows;
	int sub_col = subImg.cols;

	if (sub_row > row || sub_col > col) {
		printf("subimage fail");
		return SUB_IMAGE_MATCH_FAIL;
	}

	float min_corr = 0x4fffffff;
	float corr;
	float ST;
	float S2;
	float T2;
	int row_index;
	int col_index;
	int bigImg_pix;
	int template_pix;
	//遍历大图的每一个像素
	for (int i = 0; i < row - sub_row; i++) {
		for (int j = 0; j < col - sub_col; j++) {
			ST = 0;
			S2 = 0;
			T2 = 0;
			//遍历模板上的每一个像素
			for (int xx = 0; xx < sub_row; xx++) {
				for (int yy = 0; yy < sub_col; yy++) {
					//大图上的像素位置
					row_index = i + xx;
				    col_index = j + yy;
					bigImg_pix = grayImg.data[row_index*col + col_index];
					//模板上的像素位置
					template_pix = subImg.data[xx*sub_col + yy];
					ST += (bigImg_pix*template_pix);
					S2 += (bigImg_pix*bigImg_pix);
					T2 += (template_pix*template_pix);
					
				}
			}
			
			corr =  ST/sqrt(S2*T2);
			//比较匹配误差
			if (corr < min_corr) {
				*x = i;
				*y = j;
				min_corr = corr;
			}
		}
	}
	return SUB_IMAGE_MATCH_OK;
}//函数功能：利用亮度相关性进行子图匹配

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
